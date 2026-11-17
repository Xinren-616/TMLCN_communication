import math

import torch
import torch.nn.functional as F
from torch import nn

from comm import CommNetMLP


class BVMECommNetMLP(CommNetMLP):
    """
    BVME baseline for one-round CommNet:
    variational message encoding on aggregated communication.
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

        if self.args.hard_attn:
            comm_action = torch.tensor(info['comm_action'])
            comm_action_mask = comm_action.expand(batch_size, n, n).unsqueeze(-1)
            agent_mask *= comm_action_mask.double()

        agent_mask_transpose = agent_mask.transpose(1, 2)

        comm = hidden_state.view(batch_size, n, self.hid_size) if self.args.recurrent else hidden_state
        comm = comm.unsqueeze(-2).expand(-1, n, n, self.hid_size)

        mask = self.comm_mask.view(1, n, n)
        mask = mask.expand(comm.shape[0], n, n).unsqueeze(-1)
        mask = mask.expand_as(comm)
        comm = comm * mask

        if hasattr(self.args, 'comm_mode') and self.args.comm_mode == 'avg' and num_agents_alive > 1:
            comm = comm / (num_agents_alive - 1)

        comm = comm * agent_mask
        comm = comm * agent_mask_transpose

        self._calculate_comm_stats(comm, n, batch_size, 'round1')
        if isinstance(info, dict):
            info['comm_stats'] = self.comm_bits_stats

        comm_sum = comm.sum(dim=1)
        comm_sum, bvme_kl = self._apply_bvme(comm_sum)
        c = self.C_modules[0](comm_sum)

        if self.args.recurrent:
            inp = (x + c).view(batch_size * n, self.hid_size)
            output = self.f_module(inp, (hidden_state, cell_state))
            hidden_state = output[0]
            cell_state = output[1]
        else:
            hidden_state = sum([x, self.f_modules[0](hidden_state), c])
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

        if isinstance(info, dict):
            info['bvme_kl'] = bvme_kl

        if self.args.recurrent:
            return action, value_head, (hidden_state.clone(), cell_state.clone())
        return action, value_head
