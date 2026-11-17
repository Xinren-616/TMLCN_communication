"""
Multi-Agent Transformer (MAT) baseline for communication analysis.

Simplified MAT: uses a Transformer encoder to aggregate observations from all
agents (acting as the communication channel), then each agent decodes its own
action conditioned on the global context.  The attention weights serve as the
communication topology for Shapley analysis.

Reference: Wen et al., "Multi-Agent Transformer" (NeurIPS 2022).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from action_utils import select_action, translate_action


class MATCommNet(nn.Module):
    """
    MAT-style communication model compatible with the MAGIC trainer interface.

    Communication topology: the cross-agent attention matrix (averaged over
    heads) is used as the edge weight for Shapley analysis.
    """

    def __init__(self, args, num_inputs):
        super(MATCommNet, self).__init__()
        self.args = args
        self.nagents = args.nagents
        self.hid_size = args.hid_size
        self.comm_passes = 1
        self.recurrent = False  # MAT uses attention, not RNN

        # ── observation encoder ──
        self.encoder = nn.Linear(num_inputs, args.hid_size)

        # ── Transformer encoder (self-attention over agents) ──
        nhead = max(1, args.hid_size // 64)  # ensure divisibility
        # hid_size must be divisible by nhead
        while args.hid_size % nhead != 0 and nhead > 1:
            nhead -= 1
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=args.hid_size,
            nhead=nhead,
            dim_feedforward=args.hid_size * 2,
            dropout=0.0,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.nhead = nhead

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

        # ── communication mask (for Shapley masking) ──
        self.comm_mask = (
            torch.ones(self.nagents, self.nagents)
            - torch.eye(self.nagents, self.nagents)
        )

        # ── comm stats (same format as CommNet) ──
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

        # store last attention weights for topology extraction
        self._last_attn_weights = None  # [batch, nhead, n, n]

        # The global default tensor type is DoubleTensor; cast all parameters
        # to double so they match the input activations.
        self.double()

    # ── helpers ──────────────────────────────────────────────────────────────

    def init_hidden(self, batch_size):
        # MAT is not recurrent; return dummy hidden state
        h = torch.zeros(batch_size * self.nagents, self.hid_size)
        return (h, h)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)

    def get_agent_mask(self, batch_size, info):
        n = self.nagents
        if "alive_mask" in info:
            alive = torch.tensor(info["alive_mask"], dtype=torch.double)
            alive = alive.view(batch_size, 1, n, 1)
        else:
            alive = torch.ones(batch_size, 1, n, 1, dtype=torch.double)
        num_alive = alive.sum().item()
        return num_alive, alive

    # ── forward ──────────────────────────────────────────────────────────────

    def forward(self, x, info={}):
        """
        x: [state_tensor, prev_hidden]  (prev_hidden ignored for MAT)
        Returns: (action, value, hidden_state)
        """
        state = x[0]  # [batch, n, obs_size]
        batch_size = state.shape[0]
        n = self.nagents

        # encode observations
        h = self.tanh(self.encoder(state))  # [batch, n, hid_size]

        # apply pruning mask if provided (zero out rows/cols in attention)
        pruning_mask = info.get("pruning_mask", None)
        attn_mask = None
        if pruning_mask is not None:
            if not isinstance(pruning_mask, torch.Tensor):
                pruning_mask = torch.tensor(pruning_mask, dtype=h.dtype)
            # attn_mask: additive mask, -inf blocks attention
            # shape [n, n]; 0 = keep, -inf = block; dtype must match h
            block = (pruning_mask == 0).to(dtype=h.dtype) * (-1e9)
            attn_mask = block.to(h.device)

        # Transformer encoder: [batch, n, hid] → [batch, n, hid]
        # We need attention weights; use manual multi-head attention
        h_out, attn_w = self._transformer_with_attn(h, attn_mask)
        self._last_attn_weights = attn_w.detach()  # [batch, nhead, n, n]

        # update comm stats
        self._update_comm_stats(attn_w, batch_size, n)
        if isinstance(info, dict):
            info["comm_stats"] = self.comm_bits_stats

        # value and action
        value = self.value_head(h_out)  # [batch, n, 1]

        if self.continuous:
            action_mean = self.action_mean(h_out)
            action_log_std = self.action_log_std.expand_as(action_mean)
            action_std = torch.exp(action_log_std)
            action = (action_mean, action_log_std, action_std)
        else:
            action = [F.log_softmax(head(h_out), dim=-1) for head in self.heads]

        return action, value, h_out

    def _transformer_with_attn(self, h, attn_mask=None):
        """Run one self-attention layer and return (output, attn_weights)."""
        batch, n, d = h.shape

        # TransformerEncoderLayer was created with batch_first=True, so its
        # internal MultiheadAttention also expects [batch, seq, d] — do NOT
        # transpose to seq-first here.
        layer = self.transformer.layers[0]
        attn = layer.self_attn

        # attn_mask for batch_first MHA must be [n, n] or
        # [batch*nhead, n, n].  A plain [n, n] 2-D mask is correct.
        out, attn_w = attn(h, h, h,
                           attn_mask=attn_mask,
                           need_weights=True,
                           average_attn_weights=False)
        # attn_w: [batch, nhead, n, n]

        # feed-forward part of the encoder layer
        out2 = layer.norm1(out + h)
        out3 = layer.linear2(F.relu(layer.linear1(out2)))
        out3 = layer.norm2(out3 + out2)

        return out3, attn_w

    def _update_comm_stats(self, attn_w, batch_size, n):
        """Update comm_bits_stats from attention weights."""
        # average over heads and batch
        avg_attn = attn_w.mean(dim=(0, 1)).detach().cpu()  # [n, n]
        # remove diagonal (self-attention)
        mask = (1 - torch.eye(n)).to(avg_attn.device)
        avg_attn = avg_attn * mask

        active = (avg_attn > 0.05).float().sum().item()
        total = n * (n - 1)

        # entropy of attention distribution per agent
        row_sum = avg_attn.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        p = avg_attn / row_sum
        ent = -(p * (p + 1e-8).log()).sum().item()
        max_ent = n * np.log(n) if n > 1 else 1.0

        self.comm_bits_stats["round1"]["total_elements"] = int(total)
        self.comm_bits_stats["round1"]["active_channels"] = int(active)
        self.comm_bits_stats["round1"]["entropy"] = ent
        self.comm_bits_stats["round1"]["max_entropy"] = max_ent
        self.comm_bits_stats["round1"]["effective_bandwidth"] = ent / max_ent if max_ent > 0 else 0.0
        self.comm_bits_stats["round1"]["total_comm_num"] = int(total)
