"""
BVME (Belief-Value Message Encoder) baseline for communication analysis.

Each agent encodes its observation into a latent belief via a VAE-style encoder,
then broadcasts the belief as a message.  Receivers aggregate incoming beliefs
and use them to condition their policy.  The KL divergence between the prior
and posterior belief serves as a regulariser.

The communication topology is derived from the message norm (how much each
sender's message changes the receiver's hidden state).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from action_utils import select_action, translate_action


class BVMECommNet(nn.Module):
    """
    BVME communication model compatible with the MAGIC trainer interface.

    Architecture:
      1. Observation encoder  → hidden h_i
      2. Belief encoder (VAE) → mu_i, logvar_i  (message = reparameterised z_i)
      3. Message aggregation  → context c_i = mean of received z_j (j≠i)
      4. Policy decoder       → action / value from (h_i + c_i)
    """

    def __init__(self, args, num_inputs):
        super(BVMECommNet, self).__init__()
        self.args = args
        self.nagents = args.nagents
        self.hid_size = args.hid_size
        self.msg_size = args.hid_size // 2  # latent message dimension
        self.comm_passes = 1
        self.recurrent = False

        # ── observation encoder ──
        self.encoder = nn.Linear(num_inputs, args.hid_size)

        # ── belief encoder (VAE) ──
        self.belief_mu = nn.Linear(args.hid_size, self.msg_size)
        self.belief_logvar = nn.Linear(args.hid_size, self.msg_size)

        # ── message decoder (project received messages to hid_size) ──
        self.msg_decoder = nn.Linear(self.msg_size, args.hid_size)

        # ── policy network ──
        self.policy_fc = nn.Linear(args.hid_size * 2, args.hid_size)

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

        # ── communication mask ──
        self.comm_mask = (
            torch.ones(self.nagents, self.nagents)
            - torch.eye(self.nagents, self.nagents)
        )

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

        # KL loss accumulated during forward (used by trainer if desired)
        self.kl_loss = 0.0

        # Cast all parameters to double to match the global DoubleTensor default.
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

    def _reparameterise(self, mu, logvar):
        if self.training:
            std = (0.5 * logvar).exp()
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    # ── forward ──────────────────────────────────────────────────────────────

    def forward(self, x, info={}):
        state = x[0]  # [batch, n, obs_size]
        batch_size = state.shape[0]
        n = self.nagents
        device = state.device

        # 1. encode observations
        h = self.tanh(self.encoder(state))  # [batch, n, hid_size]

        # 2. belief encoding
        mu = self.belief_mu(h)       # [batch, n, msg_size]
        logvar = self.belief_logvar(h)
        z = self._reparameterise(mu, logvar)  # [batch, n, msg_size]

        # KL divergence: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        self.kl_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1).mean()

        # 3. message aggregation with optional pruning mask
        # z: [batch, n, msg_size]  →  broadcast to [batch, n, n, msg_size]
        z_broadcast = z.unsqueeze(1).expand(-1, n, -1, -1)  # sender dim

        # comm_mask: [n, n], 0 on diagonal
        mask = self.comm_mask.to(device).view(1, n, n, 1)

        # pruning mask
        pruning_mask = info.get("pruning_mask", None)
        if pruning_mask is not None:
            if not isinstance(pruning_mask, torch.Tensor):
                pruning_mask = torch.tensor(pruning_mask, dtype=torch.float64)
            mask = mask * pruning_mask.to(device=device, dtype=torch.float64).view(1, n, n, 1)

        # apply mask: z_broadcast[b, receiver, sender, :]
        z_masked = z_broadcast * mask.double()

        # compute edge activity for topology
        edge_activity = z_masked.norm(dim=-1).detach()  # [batch, n, n]

        # aggregate: mean over senders (excluding self)
        n_senders = mask.squeeze(-1).sum(dim=-1, keepdim=True).clamp(min=1)
        context_msg = z_masked.sum(dim=2) / n_senders.squeeze(-1).unsqueeze(-1)
        context = self.tanh(self.msg_decoder(context_msg))  # [batch, n, hid_size]

        # 4. policy
        combined = torch.cat([h, context], dim=-1)  # [batch, n, hid_size*2]
        h_out = self.tanh(self.policy_fc(combined))  # [batch, n, hid_size]

        # update comm stats
        self._update_comm_stats(edge_activity, z, batch_size, n)
        if isinstance(info, dict):
            info["comm_stats"] = self.comm_bits_stats

        # value and action
        value = self.value_head(h_out)

        if self.continuous:
            action_mean = self.action_mean(h_out)
            action_log_std = self.action_log_std.expand_as(action_mean)
            action_std = torch.exp(action_log_std)
            action = (action_mean, action_log_std, action_std)
        else:
            action = [F.log_softmax(head(h_out), dim=-1) for head in self.heads]

        return action, value, h_out

    def _update_comm_stats(self, edge_activity, z, batch_size, n):
        avg_act = edge_activity.mean(dim=0).cpu()  # [n, n]
        diag_mask = 1 - torch.eye(n, device=avg_act.device)
        avg_act = avg_act * diag_mask

        threshold = avg_act.max().item() * 0.1 + 1e-8
        active = (avg_act > threshold).float().sum().item()
        total = n * (n - 1)

        # message similarity (cosine between agent messages)
        z_norm = F.normalize(z.mean(dim=0).float(), dim=-1)  # [n, msg_size]
        sim = (z_norm @ z_norm.T).mean().item()

        self.comm_bits_stats["round1"]["total_elements"] = int(total)
        self.comm_bits_stats["round1"]["active_channels"] = int(active)
        self.comm_bits_stats["round1"]["message_similarity"] = float(sim)
        self.comm_bits_stats["round1"]["total_comm_num"] = int(total)
        max_ent = np.log(n) if n > 1 else 1.0
        self.comm_bits_stats["round1"]["max_entropy"] = max_ent
