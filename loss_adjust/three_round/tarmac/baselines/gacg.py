import torch
import torch.nn.functional as F
from torch import nn

from comm import CommNetMLP


class GACGNetMLP(CommNetMLP):
    """
    GACG baseline:
    - Stochastic coordination graph edges inferred from agent pairs.
    - Group-aware dependence estimated from short observation history.
    - Group distance regularization term exposed to trainer.
    """

    def __init__(self, args, num_inputs):
        super(GACGNetMLP, self).__init__(args, num_inputs)
        self.history_k = int(getattr(args, "gacg_history_k", 4))
        self.num_groups = int(getattr(args, "gacg_num_groups", 2))
        self.group_var_scale = float(getattr(args, "gacg_group_var_scale", 0.5))
        self.edge_logvar_min = float(getattr(args, "gacg_edge_logvar_min", -5.0))
        self.edge_logvar_max = float(getattr(args, "gacg_edge_logvar_max", 1.0))

        self.group_divider = nn.Linear(num_inputs, self.num_groups)
        self.edge_pair = nn.Linear(self.hid_size * 2, self.hid_size)
        self.edge_mu = nn.Linear(self.hid_size, 1)
        self.edge_logvar = nn.Linear(self.hid_size, 1)

        self._obs_history = []

    def reset_episode_state(self):
        self._obs_history = []

    def _build_group_context(self, obs):
        self._obs_history.append(obs.detach())
        if len(self._obs_history) > self.history_k:
            self._obs_history = self._obs_history[-self.history_k:]
        return torch.stack(self._obs_history, dim=0).mean(dim=0)

    def _sample_coordination_edges(self, comm_base, same_group):
        batch_size, n, _ = comm_base.shape
        sender = comm_base.unsqueeze(2).expand(-1, n, n, self.hid_size)
        receiver = comm_base.unsqueeze(1).expand(-1, n, n, self.hid_size)
        pair_feat = torch.cat([sender, receiver], dim=-1)
        pair_h = torch.tanh(self.edge_pair(pair_feat))

        mu = torch.sigmoid(self.edge_mu(pair_h)).squeeze(-1)
        logvar = torch.clamp(self.edge_logvar(pair_h).squeeze(-1), self.edge_logvar_min, self.edge_logvar_max)
        var = torch.exp(logvar)

        # Same-group edges are encouraged to be more certain (smaller variance).
        var_scale = (1.0 - self.group_var_scale * same_group).clamp(min=0.1)
        var = var * var_scale

        std = torch.sqrt(var + 1e-8)
        edge = torch.sigmoid(mu + std * torch.randn_like(std))

        eye = torch.eye(n, device=edge.device, dtype=edge.dtype).unsqueeze(0)
        edge = edge * (1.0 - eye)
        return edge

    def _compute_group_distance_loss(self, features, group_probs):
        # features: [B, N, H], group_probs: [B, N, G]
        batch_size, n, _ = features.shape
        if n <= 1:
            return features.new_tensor(0.0)

        eps = 1e-8
        dist = torch.cdist(features, features, p=2)
        eye = torch.eye(n, device=features.device, dtype=features.dtype).unsqueeze(0)

        same = torch.matmul(group_probs, group_probs.transpose(1, 2)) * (1.0 - eye)
        diff = (1.0 - same) * (1.0 - eye)

        intra = (dist * same).sum(dim=(1, 2)) / (same.sum(dim=(1, 2)) + eps)
        inter = (dist * diff).sum(dim=(1, 2)) / (diff.sum(dim=(1, 2)) + eps)
        return (intra / (inter + eps)).mean()

    def forward(self, x, info={}):
        raw_obs = x[0] if self.args.recurrent else x
        x, hidden_state, cell_state = self.forward_state_encoder(x)

        batch_size = x.size(0)
        n = self.nagents
        num_agents_alive, agent_mask = self.get_agent_mask(batch_size, info)
        agent_mask_transpose = agent_mask.transpose(1, 2)

        group_context = self._build_group_context(raw_obs)
        group_logits = self.group_divider(group_context)
        group_probs = F.softmax(group_logits, dim=-1)
        same_group = torch.matmul(group_probs, group_probs.transpose(1, 2))

        # Hard attention - action whether an agent communicates or not
        if self.args.hard_attn:
            comm_action = torch.tensor(info['comm_action'])
            comm_action_mask = comm_action.expand(batch_size, n, n).unsqueeze(-1)
            agent_mask *= comm_action_mask.double()

        per_round_stats = {
            'round1': {'total_elements': 0, 'active_channels': 0.0, 'total_comm_num': 0, 'entropy': 0.0,
                       'max_entropy': 0.0, 'effective_bandwidth': 0.0, 'message_similarity': 0.0},
            'round2': {'total_elements': 0, 'active_channels': 0.0, 'total_comm_num': 0, 'entropy': 0.0,
                       'max_entropy': 0.0, 'effective_bandwidth': 0.0, 'message_similarity': 0.0},
            'round3': {'total_elements': 0, 'active_channels': 0.0, 'total_comm_num': 0, 'entropy': 0.0,
                       'max_entropy': 0.0, 'effective_bandwidth': 0.0, 'message_similarity': 0.0},
        }
        all_pass_stats = []

        for i in range(self.comm_passes):
            comm_base = hidden_state.view(batch_size, n, self.hid_size) if self.args.recurrent else hidden_state
            comm = comm_base.unsqueeze(-2).expand(-1, n, n, self.hid_size)

            edge = self._sample_coordination_edges(comm_base, same_group)
            if hasattr(self.args, 'comm_mode') and self.args.comm_mode == 'avg' and num_agents_alive > 1:
                edge = edge / (num_agents_alive - 1)
            comm = comm * edge.unsqueeze(-1)

            # Mask communication from dead/silent agents and to dead agents.
            comm = comm * agent_mask
            comm = comm * agent_mask_transpose

            comm_stats = self._calculate_comm_stats(comm, n, batch_size)
            all_pass_stats.append(comm_stats)
            if i < 3:
                round_name = f'round{i + 1}'
                for key in per_round_stats[round_name]:
                    per_round_stats[round_name][key] = float(comm_stats.get(key, 0.0))

            comm_sum = comm.sum(dim=1)
            c = self.C_modules[i](comm_sum)

            if self.args.recurrent:
                inp = (x + c).view(batch_size * n, self.hid_size)
                output = self.f_module(inp, (hidden_state, cell_state))
                hidden_state = output[0]
                cell_state = output[1]
            else:
                hidden_state = sum([x, self.f_modules[i](hidden_state), c])
                hidden_state = self.tanh(hidden_state)

        value_head = self.value_head(hidden_state)
        h = hidden_state.view(batch_size, n, self.hid_size)
        group_loss = self._compute_group_distance_loss(h, group_probs)

        if self.continuous:
            action_mean = self.action_mean(h)
            action_log_std = self.action_log_std.expand_as(action_mean)
            action_std = torch.exp(action_log_std)
            action = (action_mean, action_log_std, action_std)
        else:
            action = [F.log_softmax(head(h), dim=-1) for head in self.heads]

        pass_count = max(1, len(all_pass_stats))
        info['gacg_group_loss'] = group_loss
        info['comm_stats'] = {
            'round1': per_round_stats['round1'],
            'round2': per_round_stats['round2'],
            'round3': per_round_stats['round3'],
            'total_elements': sum(s.get('total_elements', 0.0) for s in all_pass_stats),
            'total_comm_num': sum(s.get('total_comm_num', 0.0) for s in all_pass_stats),
            'active_channels': sum(s.get('active_channels', 0.0) for s in all_pass_stats) / pass_count,
            'total_entropy': sum(s.get('entropy', 0.0) for s in all_pass_stats) / pass_count,
            'total_max_entropy': sum(s.get('max_entropy', 0.0) for s in all_pass_stats) / pass_count,
            'total_effective_bandwidth': sum(s.get('effective_bandwidth', 0.0) for s in all_pass_stats) / pass_count,
            'total_message_similarity': sum(s.get('message_similarity', 0.0) for s in all_pass_stats) / pass_count,
        }

        if self.args.recurrent:
            return action, value_head, (hidden_state.clone(), cell_state.clone())
        return action, value_head
