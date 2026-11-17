import math
import torch
import torch.nn.functional as F
from torch import nn

from comm import CommNetMLP


class BVMECommNetMLP(CommNetMLP):
    """
    BVME baseline on top of two-round CommNet:
    variational message encoding after communication aggregation.
    """

    def __init__(self, args, num_inputs):
        super(BVMECommNetMLP, self).__init__(args, num_inputs)
        self.bvme_mu = nn.Linear(args.hid_size, args.hid_size)
        self.bvme_logvar = nn.Linear(args.hid_size, args.hid_size)
        self.bvme_sigma0 = float(getattr(args, 'bvme_sigma0', 0.1))
        self.bvme_logvar_min = float(getattr(args, 'bvme_logvar_min', -5.0))
        self.bvme_logvar_max = float(getattr(args, 'bvme_logvar_max', 3.0))

    def _apply_bvme(self, message_tensor):
        """
        Args:
            message_tensor: [batch_size, n, hid_size]
        """
        mu = self.bvme_mu(message_tensor)
        logvar = self.bvme_logvar(message_tensor)
        logvar = torch.clamp(logvar, min=self.bvme_logvar_min, max=self.bvme_logvar_max)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + std * eps

        prior_var = max(self.bvme_sigma0 * self.bvme_sigma0, 1e-8)
        log_prior_var = math.log(prior_var)
        kl = 0.5 * (((std.pow(2) + mu.pow(2)) / prior_var) - 1.0 + log_prior_var - logvar)
        kl = kl.sum(dim=-1).mean()
        return z, kl

    def forward(self, x, info={}):
        x, hidden_state, cell_state = self.forward_state_encoder(x)

        batch_size = x.size(0)
        n = self.nagents
        num_agents_alive, agent_mask = self.get_agent_mask(batch_size, info)
        agent_mask_transpose = agent_mask.transpose(1, 2)

        if self.args.hard_attn:
            comm_action = torch.tensor(info['comm_action'])
            comm_action_mask = comm_action.expand(batch_size, n, n).unsqueeze(-1)
            agent_mask *= comm_action_mask.double()

        per_round_stats = {
            'round1': {'total_elements': 0, 'active_channels': 0.0, 'total_comm_num': 0, 'entropy': 0.0,
                       'max_entropy': 0.0, 'effective_bandwidth': 0.0, 'message_similarity': 0.0},
            'round2': {'total_elements': 0, 'active_channels': 0.0, 'total_comm_num': 0, 'entropy': 0.0,
                       'max_entropy': 0.0, 'effective_bandwidth': 0.0, 'message_similarity': 0.0},
        }
        all_pass_stats = []
        total_bvme_kl = None

        for i in range(self.comm_passes):
            comm = hidden_state.view(batch_size, n, self.hid_size) if self.args.recurrent else hidden_state
            comm = comm.unsqueeze(-2).expand(-1, n, n, self.hid_size)

            mask = self.comm_mask.view(1, n, n).expand(comm.shape[0], n, n).unsqueeze(-1)
            mask = mask.expand_as(comm)
            comm = comm * mask

            if hasattr(self.args, 'comm_mode') and self.args.comm_mode == 'avg' and num_agents_alive > 1:
                comm = comm / (num_agents_alive - 1)

            comm = comm * agent_mask
            comm = comm * agent_mask_transpose

            comm_stats = self._calculate_comm_stats(comm, n, batch_size)
            all_pass_stats.append(comm_stats)
            if i < 2:
                round_name = f'round{i + 1}'
                for key in per_round_stats[round_name]:
                    per_round_stats[round_name][key] = float(comm_stats.get(key, 0.0))

            comm_sum = comm.sum(dim=1)
            comm_sum, step_kl = self._apply_bvme(comm_sum)
            if total_bvme_kl is None:
                total_bvme_kl = step_kl
            else:
                total_bvme_kl = total_bvme_kl + step_kl

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

        if self.continuous:
            action_mean = self.action_mean(h)
            action_log_std = self.action_log_std.expand_as(action_mean)
            action_std = torch.exp(action_log_std)
            action = (action_mean, action_log_std, action_std)
        else:
            action = [F.log_softmax(head(h), dim=-1) for head in self.heads]

        if total_bvme_kl is None:
            total_bvme_kl = hidden_state.new_tensor(0.0)

        pass_count = max(1, len(all_pass_stats))
        if isinstance(info, dict):
            info['bvme_kl'] = total_bvme_kl / max(1, self.comm_passes)
            info['comm_stats'] = {
                'round1': per_round_stats['round1'],
                'round2': per_round_stats['round2'],
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
import math

import torch
import torch.nn.functional as F
from torch import nn

from comm import CommNetMLP


class BVMECommNetMLP(CommNetMLP):
    """
    BVME baseline on top of two-round CommNet:
    variational message encoding after communication aggregation.
    """

    def __init__(self, args, num_inputs):
        super(BVMECommNetMLP, self).__init__(args, num_inputs)
        self.bvme_mu = nn.Linear(args.hid_size, args.hid_size)
        self.bvme_logvar = nn.Linear(args.hid_size, args.hid_size)
        self.bvme_sigma0 = float(getattr(args, 'bvme_sigma0', 0.1))
        self.bvme_logvar_min = float(getattr(args, 'bvme_logvar_min', -5.0))
        self.bvme_logvar_max = float(getattr(args, 'bvme_logvar_max', 3.0))

    def _apply_bvme(self, message_tensor):
        """
        Args:
            message_tensor: [batch_size, n, hid_size]
        """
        mu = self.bvme_mu(message_tensor)
        logvar = self.bvme_logvar(message_tensor)
        logvar = torch.clamp(logvar, min=self.bvme_logvar_min, max=self.bvme_logvar_max)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + std * eps

        prior_var = max(self.bvme_sigma0 * self.bvme_sigma0, 1e-8)
        log_prior_var = math.log(prior_var)
        kl = 0.5 * (((std.pow(2) + mu.pow(2)) / prior_var) - 1.0 + log_prior_var - logvar)
        kl = kl.sum(dim=-1).mean()
        return z, kl

    def forward(self, x, info={}):
        x, hidden_state, cell_state = self.forward_state_encoder(x)

        batch_size = x.size(0)
        n = self.nagents
        num_agents_alive, agent_mask = self.get_agent_mask(batch_size, info)
        agent_mask_transpose = agent_mask.transpose(1, 2)

        if self.args.hard_attn:
            comm_action = torch.tensor(info['comm_action'])
            comm_action_mask = comm_action.expand(batch_size, n, n).unsqueeze(-1)
            agent_mask *= comm_action_mask.double()

        per_round_stats = {
            'round1': {'total_elements': 0, 'active_channels': 0.0, 'total_comm_num': 0, 'entropy': 0.0,
                       'max_entropy': 0.0, 'effective_bandwidth': 0.0, 'message_similarity': 0.0},
            'round2': {'total_elements': 0, 'active_channels': 0.0, 'total_comm_num': 0, 'entropy': 0.0,
                       'max_entropy': 0.0, 'effective_bandwidth': 0.0, 'message_similarity': 0.0},
        }
        all_pass_stats = []
        total_bvme_kl = None

        for i in range(self.comm_passes):
            comm = hidden_state.view(batch_size, n, self.hid_size) if self.args.recurrent else hidden_state
            comm = comm.unsqueeze(-2).expand(-1, n, n, self.hid_size)

            mask = self.comm_mask.view(1, n, n).expand(comm.shape[0], n, n).unsqueeze(-1)
            mask = mask.expand_as(comm)
            comm = comm * mask

            if hasattr(self.args, 'comm_mode') and self.args.comm_mode == 'avg' and num_agents_alive > 1:
                comm = comm / (num_agents_alive - 1)

            comm = comm * agent_mask
            comm = comm * agent_mask_transpose

            comm_stats = self._calculate_comm_stats(comm, n, batch_size)
            all_pass_stats.append(comm_stats)
            if i < 2:
                round_name = f'round{i + 1}'
                for key in per_round_stats[round_name]:
                    per_round_stats[round_name][key] = float(comm_stats.get(key, 0.0))

            comm_sum = comm.sum(dim=1)
            comm_sum, step_kl = self._apply_bvme(comm_sum)
            if total_bvme_kl is None:
                total_bvme_kl = step_kl
            else:
                total_bvme_kl = total_bvme_kl + step_kl

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

        if self.continuous:
            action_mean = self.action_mean(h)
            action_log_std = self.action_log_std.expand_as(action_mean)
            action_std = torch.exp(action_log_std)
            action = (action_mean, action_log_std, action_std)
        else:
            action = [F.log_softmax(head(h), dim=-1) for head in self.heads]

        if total_bvme_kl is None:
            total_bvme_kl = hidden_state.new_tensor(0.0)

        pass_count = max(1, len(all_pass_stats))
        if isinstance(info, dict):
            info['bvme_kl'] = total_bvme_kl / max(1, self.comm_passes)
            info['comm_stats'] = {
                'round1': per_round_stats['round1'],
                'round2': per_round_stats['round2'],
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
