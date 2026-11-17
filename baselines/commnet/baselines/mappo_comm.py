"""
MAPPO (Multi-Agent PPO) with optional full communication layer.

Architecture:
  - Actor: obs encoder → optional comm aggregation (mean-pool) → action head
  - Critic: centralized (all agents' obs concatenated) → value head

comm_enabled=True  → MAPPO + FullComm (full connected comm, prunable via mask)
comm_enabled=False → pure MAPPO (no communication, set via --comm_mask_zero)

Compatible with existing trainer.py + Shapley/CTAP infrastructure.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from action_utils import select_action, translate_action


class MAPPOCommNet(nn.Module):
    """
    MAPPO with optional communication, compatible with the MAGIC trainer interface.

    When comm_mask_zero=True (--comm_mask_zero flag), communication is disabled
    entirely, giving pure MAPPO. Otherwise full communication is used, optionally
    pruned by a pruning_mask from CTAP.
    """

    def __init__(self, args, num_inputs):
        super(MAPPOCommNet, self).__init__()
        self.args = args
        self.nagents = args.nagents
        self.hid_size = args.hid_size
        self.comm_passes = 1
        self.recurrent = False

        # comm_enabled: False when --comm_mask_zero is set (pure MAPPO no-comm)
        self.comm_enabled = not getattr(args, 'comm_mask_zero', False)

        # ── observation encoder (actor) ──
        self.encoder = nn.Linear(num_inputs, args.hid_size)

        # ── communication aggregation (only used when comm_enabled) ──
        if self.comm_enabled:
            self.comm_encoder = nn.Linear(args.hid_size, args.hid_size)
            self.comm_decoder = nn.Linear(args.hid_size * 2, args.hid_size)

        # ── centralized critic: takes all agents' obs concatenated ──
        self.critic_encoder = nn.Linear(num_inputs * args.nagents, args.hid_size)
        self.value_head = nn.Linear(args.hid_size, 1)

        # ── action heads ──
        self.continuous = args.continuous
        if self.continuous:
            self.action_mean = nn.Linear(args.hid_size, args.dim_actions)
            self.action_log_std = nn.Parameter(torch.zeros(1, args.dim_actions))
        else:
            self.heads = nn.ModuleList(
                [nn.Linear(args.hid_size, o) for o in args.naction_heads]
            )

        self.tanh = nn.Tanh()

        # ── communication mask (off-diagonal) ──
        self.comm_mask = (
            torch.ones(self.nagents, self.nagents)
            - torch.eye(self.nagents, self.nagents)
        )

        # ── comm stats (same format as other baselines) ──
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
        state = x[0]  # [batch, n, obs_size]
        batch_size = state.shape[0]
        n = self.nagents
        device = state.device

        # 1. encode observations
        h = self.tanh(self.encoder(state))  # [batch, n, hid_size]

        # 2. optional communication aggregation
        if self.comm_enabled:
            # encode messages
            msg = self.tanh(self.comm_encoder(h))  # [batch, n, hid_size]

            # build comm mask: [1, n, n, 1]
            mask = self.comm_mask.to(device).view(1, n, n, 1)

            # apply pruning mask if provided
            pruning_mask = info.get("pruning_mask", None)
            if pruning_mask is not None:
                if not isinstance(pruning_mask, torch.Tensor):
                    pruning_mask = torch.tensor(pruning_mask, dtype=torch.float64)
                mask = mask * pruning_mask.to(device=device, dtype=torch.float64).view(1, n, n, 1)

            # msg_broadcast[b, receiver, sender, :] = msg[b, sender, :]
            msg_broadcast = msg.unsqueeze(1).expand(-1, n, -1, -1)  # [batch, n, n, hid]
            msg_masked = msg_broadcast * mask.double()

            # mean aggregation over senders
            n_senders = mask.squeeze(-1).sum(dim=-1, keepdim=True).clamp(min=1)
            context = msg_masked.sum(dim=2) / n_senders.squeeze(-1).unsqueeze(-1)
            context = self.tanh(context)  # [batch, n, hid_size]

            # combine own encoding with received context
            combined = torch.cat([h, context], dim=-1)  # [batch, n, hid_size*2]
            h_out = self.tanh(self.comm_decoder(combined))  # [batch, n, hid_size]

            self._update_comm_stats(mask, msg, batch_size, n)
        else:
            h_out = h
            self._update_comm_stats_no_comm(n)

        if isinstance(info, dict):
            info["comm_stats"] = self.comm_bits_stats

        # 3. centralized critic: use all agents' raw observations
        # state: [batch, n, obs_size] → [batch, n * obs_size]
        state_flat = state.view(batch_size, -1)  # [batch, n * obs_size]
        v_enc = self.tanh(self.critic_encoder(state_flat))  # [batch, hid_size]
        # expand to per-agent: [batch, n, 1]
        value = self.value_head(v_enc).unsqueeze(1).expand(-1, n, -1)  # [batch, n, 1]

        # 4. action
        if self.continuous:
            action_mean = self.action_mean(h_out)
            action_log_std = self.action_log_std.expand_as(action_mean)
            action_std = torch.exp(action_log_std)
            action = (action_mean, action_log_std, action_std)
        else:
            action = [F.log_softmax(head(h_out), dim=-1) for head in self.heads]

        return action, value, h_out

    def _update_comm_stats(self, mask, msg, batch_size, n):
        avg_mask = mask.squeeze(-1).squeeze(0).cpu()  # [n, n]
        total = n * (n - 1)
        active = (avg_mask > 0).float().sum().item()

        msg_norm = F.normalize(msg.mean(dim=0).float(), dim=-1)  # [n, hid]
        sim = (msg_norm @ msg_norm.T).mean().item()

        max_ent = np.log(n) if n > 1 else 1.0
        self.comm_bits_stats["round1"]["total_elements"] = int(total)
        self.comm_bits_stats["round1"]["active_channels"] = int(active)
        self.comm_bits_stats["round1"]["message_similarity"] = float(sim)
        self.comm_bits_stats["round1"]["total_comm_num"] = int(active)
        self.comm_bits_stats["round1"]["max_entropy"] = max_ent

    def _update_comm_stats_no_comm(self, n):
        self.comm_bits_stats["round1"]["total_elements"] = n * (n - 1)
        self.comm_bits_stats["round1"]["active_channels"] = 0
        self.comm_bits_stats["round1"]["total_comm_num"] = 0
