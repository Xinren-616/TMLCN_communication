"""
ExpoComm: Exponential Topology-enabled Scalable Communication.

Ported from the EPyMARL-based implementation (ICLR 2025) to the MAGIC trainer
interface so it can be compared directly with CommNet/MAGIC/CTAP baselines.

Key idea: instead of full O(n²) communication, each agent i only receives
messages from a fixed sparse set of peers determined by the exponential
topology — agent i receives from agents at offsets {0, 1, 2, 4, 8, ...} mod n.
This gives O(n log n) total edges while maintaining small graph diameter.

Reference: Li et al., "Exponential Topology-enabled Scalable Communication
in Multi-Agent Reinforcement Learning", ICLR 2025.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from action_utils import select_action, translate_action


def _build_expo_topology(n_agents: int) -> torch.Tensor:
    """
    Build the exponential topology adjacency matrix.

    For agent i, it receives from agents at positions:
        (i + 0) % n, (i + 1) % n, (i + 2) % n, (i + 4) % n, ...
    until we exhaust powers of 2 that are < n.

    Returns: [n, n] float tensor, entry [i, j] = 1 if agent i receives from j.
    """
    offsets = [0]
    k = 1
    while k < n_agents:
        offsets.append(k)
        k *= 2

    adj = torch.zeros(n_agents, n_agents)
    for i in range(n_agents):
        for off in offsets:
            j = (i + off) % n_agents
            if j != i:  # exclude self-loop (self handled separately)
                adj[i, j] = 1.0
    return adj


class ExpoCommNet(nn.Module):
    """
    ExpoComm ported to the MAGIC trainer interface.

    Architecture (per step):
      1. obs → fc1 → ReLU → h  (observation encoder)
      2. GRU(h, hidden) → hidden'
      3. Exponential-topology message passing:
           key/query/value attention over topk neighbors
           → aggregated message m_i
      4. [hidden', m_i] → fc2 → action logits
      5. Centralized value head: mean(hidden') → value

    comm_enabled=False (--comm_mask_zero) disables message passing entirely.
    """

    def __init__(self, args, num_inputs):
        super(ExpoCommNet, self).__init__()
        self.args = args
        self.nagents = args.nagents
        self.hid_size = args.hid_size
        self.comm_passes = 1
        self.recurrent = False

        self.comm_enabled = not getattr(args, 'comm_mask_zero', False)

        # ── observation encoder ──
        self.fc1 = nn.Linear(num_inputs, args.hid_size)
        self.rnn = nn.GRUCell(args.hid_size, args.hid_size)

        # ── attention-based message aggregation ──
        attn_dim = max(16, args.hid_size // 4)
        self.attn_dim = attn_dim
        if self.comm_enabled:
            self.msg_k = nn.Linear(args.hid_size, attn_dim)
            self.msg_q = nn.Linear(args.hid_size, attn_dim)
            self.msg_v = nn.Linear(args.hid_size, args.hid_size)

        # ── policy head ──
        in_dim = args.hid_size * 2 if self.comm_enabled else args.hid_size
        self.fc2 = nn.Sequential(
            nn.Linear(in_dim, args.hid_size),
            nn.ReLU(),
        )

        # ── action heads ──
        self.continuous = args.continuous
        if self.continuous:
            self.action_mean = nn.Linear(args.hid_size, args.dim_actions)
            self.action_log_std = nn.Parameter(torch.zeros(1, args.dim_actions))
        else:
            self.heads = nn.ModuleList(
                [nn.Linear(args.hid_size, o) for o in args.naction_heads]
            )

        self.value_head = nn.Linear(args.hid_size, 1)
        self.tanh = nn.Tanh()

        # ── pre-compute exponential topology ──
        # adj[i, j] = 1 means agent i receives from agent j
        self.register_buffer('expo_adj', _build_expo_topology(args.nagents))
        # topk = number of peers each agent receives from
        self.topk = int(self.expo_adj[0].sum().item())

        # ── comm stats ──
        self.comm_bits_stats = {
            "round1": {
                "total_elements": 0,
                "active_channels": 0,
                "entropy": 0.0,
                "max_entropy": 0.0,
                "effective_bandwidth": 0.0,
                "message_similarity": 0.0,
                "total_comm_num": 0,
            }
        }

        self.double()

    # ── helpers ──────────────────────────────────────────────────────────────

    def init_hidden(self, batch_size):
        h = torch.zeros(batch_size * self.nagents, self.hid_size)
        return (h, h)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)

    def get_agent_mask(self, batch_size, info):
        n = self.nagents
        if "alive_mask" in info:
            alive = torch.tensor(info["alive_mask"], dtype=torch.double)
            alive = alive.view(batch_size, n)
        else:
            alive = torch.ones(batch_size, n, dtype=torch.double)
        return alive.sum().item(), alive

    # ── forward ──────────────────────────────────────────────────────────────

    def forward(self, x, info={}):
        """
        x: [state_tensor, prev_hidden]
        Returns: (action, value, hidden_state)
        """
        state = x[0]          # [batch, n, obs_size]
        prev_h = x[1]         # [batch*n, hid_size] or tuple
        if isinstance(prev_h, tuple):
            prev_h = prev_h[0]

        batch_size = state.shape[0]
        n = self.nagents
        device = state.device

        # flatten for GRU: [batch*n, obs_size]
        state_flat = state.view(batch_size * n, -1)
        prev_h_flat = prev_h.view(batch_size * n, self.hid_size)

        # 1. encode + GRU
        x_enc = F.relu(self.fc1(state_flat))          # [batch*n, hid]
        h = self.rnn(x_enc, prev_h_flat)              # [batch*n, hid]
        h_3d = h.view(batch_size, n, self.hid_size)   # [batch, n, hid]

        # 2. exponential-topology message passing
        if self.comm_enabled:
            # apply pruning mask on top of expo topology if provided
            adj = self.expo_adj.to(device)  # [n, n]
            pruning_mask = info.get("pruning_mask", None)
            if pruning_mask is not None:
                if not isinstance(pruning_mask, torch.Tensor):
                    pruning_mask = torch.tensor(pruning_mask, dtype=torch.float64)
                adj = adj * pruning_mask.to(device=device, dtype=torch.float64)

            # attention over neighbors
            # k, q, v: [batch, n, attn_dim/hid]
            k = self.msg_k(h_3d)   # [batch, n, attn_dim]
            q = self.msg_q(h_3d)   # [batch, n, attn_dim]
            v = self.msg_v(h_3d)   # [batch, n, hid]

            # for each receiver i, gather sender j's k/v weighted by adj[i,j]
            # adj: [n, n] → [1, n, n] broadcast over batch
            adj_b = adj.unsqueeze(0)  # [1, n, n]

            # attn_scores[b, i, j] = q[b,i] · k[b,j] / sqrt(d), masked by adj
            attn_scores = torch.bmm(
                q,                          # [batch, n, attn_dim]
                k.transpose(1, 2)           # [batch, attn_dim, n]
            ) / math.sqrt(self.attn_dim)    # [batch, n, n]

            # mask out non-neighbors with -inf
            mask_inf = (adj_b == 0).expand(batch_size, -1, -1)
            attn_scores = attn_scores.masked_fill(mask_inf, -1e9)

            # if all neighbors pruned for an agent, fall back to zeros
            has_neighbor = adj_b.sum(dim=-1, keepdim=True) > 0  # [1, n, 1]
            attn_w = torch.softmax(attn_scores, dim=-1)          # [batch, n, n]
            attn_w = attn_w * adj_b                              # zero non-neighbors
            attn_w = attn_w * has_neighbor.float()               # zero if no neighbors

            # aggregate: [batch, n, hid]
            msg = torch.bmm(attn_w, v)  # [batch, n, hid]

            # combine hidden + message
            combined = torch.cat([h_3d, msg], dim=-1)  # [batch, n, hid*2]
            h_out = self.fc2(combined)                  # [batch, n, hid]

            self._update_comm_stats(adj, attn_w, h_3d, batch_size, n)
        else:
            h_out = self.fc2(h_3d) if self.fc2[0].in_features == self.hid_size \
                    else self.tanh(h_3d)
            self._update_comm_stats_no_comm(n)

        if isinstance(info, dict):
            info["comm_stats"] = self.comm_bits_stats

        # 3. value (centralized: mean over agents)
        value = self.value_head(h_out)  # [batch, n, 1]

        # 4. action
        if self.continuous:
            action_mean = self.action_mean(h_out)
            action_log_std = self.action_log_std.expand_as(action_mean)
            action_std = torch.exp(action_log_std)
            action = (action_mean, action_log_std, action_std)
        else:
            action = [F.log_softmax(head(h_out), dim=-1) for head in self.heads]

        return action, value, h_out.view(batch_size * n, self.hid_size)

    def _update_comm_stats(self, adj, attn_w, h_3d, batch_size, n):
        total = int(adj.sum().item())  # actual edges in expo topology
        active = total  # all expo edges are always active (fixed topology)

        # entropy of attention distribution per agent (averaged over batch)
        avg_attn = attn_w.mean(dim=0).detach().cpu()  # [n, n]
        row_sum = avg_attn.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        p = avg_attn / row_sum
        ent = -(p * (p + 1e-8).log()).sum().item()
        max_ent = n * math.log(n) if n > 1 else 1.0

        # message similarity
        h_norm = F.normalize(h_3d.mean(dim=0).float(), dim=-1)  # [n, hid]
        sim = (h_norm @ h_norm.T).mean().item()

        self.comm_bits_stats["round1"]["total_elements"] = n * (n - 1)
        self.comm_bits_stats["round1"]["active_channels"] = active
        self.comm_bits_stats["round1"]["total_comm_num"] = total
        self.comm_bits_stats["round1"]["entropy"] = ent
        self.comm_bits_stats["round1"]["max_entropy"] = max_ent
        self.comm_bits_stats["round1"]["effective_bandwidth"] = ent / max_ent if max_ent > 0 else 0.0
        self.comm_bits_stats["round1"]["message_similarity"] = float(sim)

    def _update_comm_stats_no_comm(self, n):
        self.comm_bits_stats["round1"]["total_elements"] = n * (n - 1)
        self.comm_bits_stats["round1"]["active_channels"] = 0
        self.comm_bits_stats["round1"]["total_comm_num"] = 0
