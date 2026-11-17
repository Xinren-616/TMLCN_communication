from collections import namedtuple
from inspect import getargspec
import numpy as np
import torch
from torch import optim
import torch.nn as nn
from utils import *
from action_utils import *
import itertools

Transition = namedtuple('Transition', ('state', 'action', 'action_out', 'value', 'episode_mask', 'episode_mini_mask', 'next_state', 'reward', 'misc'))


class Trainer(object):
    def __init__(self, args, policy_net, env):
        self.args = args
        self.policy_net = policy_net
        self.env = env
        self.display = False
        self.last_step = False
        self.optimizer = optim.RMSprop(policy_net.parameters(),
            lr = args.lrate, alpha=0.97, eps=1e-6)
        self.params = [p for p in self.policy_net.parameters()]

        self.batch_counter = 0
        
        # 确保nprocesses参数存在
        if not hasattr(self.args, 'nprocesses'):
            self.args.nprocesses = 1
        
        # 初始化batch级别通信统计 (简化版本)
        self.batch_comm_stats = {
            'round1': {
                'total_elements': 0,         # 传输的消息数(元素数)
                'active_channels': 0,        # 活跃通信通道数（边数）
                'entropy': 0.0,              # 通信信息熵
                'max_entropy': 0.0,          # 最大熵
                'effective_bandwidth': 0.0,  # 有效带宽利用率
                'message_similarity': 0.0,   # 消息相似度
                'total_comm_num': 0,         # 通信次数（累加型）
            },
            'round2': {
                'total_elements': 0,         # 传输的消息数(元素数)
                'active_channels': 0,        # 活跃通信通道数（边数）
                'entropy': 0.0,              # 通信信息熵
                'max_entropy': 0.0,          # 最大熵
                'effective_bandwidth': 0.0,  # 有效带宽利用率
                'message_similarity': 0.0,   # 消息相似度
                'total_comm_num': 0,         # 通信次数（累加型）
            },
            'total_elements': 0,             # 总传输消息数
            'total_entropy': 0.0,            # 总信息熵
            'total_max_entropy': 0.0,        # 总最大熵
            'total_effective_bandwidth': 0.0, # 总有效带宽利用率
            'total_message_similarity': 0.0,  # 总消息相似度
            'total_comm_num': 0,             # 总通信次数（累加型）
            'active_channels': 0,            # 总活跃通信通道数
            'episode_count': 0
        }

    def get_episode(self, epoch):
        episode = []
        reset_args = getargspec(self.env.reset).args
        if 'epoch' in reset_args:
            state = self.env.reset(epoch)
        else:
            state = self.env.reset()
        should_display = self.display and self.last_step

        if should_display:
            self.env.display()
        stat = dict()
        info = dict()
        info['epoch'] = epoch

        # 如果stats中有success信息，添加到info中
        if hasattr(self, 'stats') and 'success' in self.stats:
            info['success'] = self.stats['success']
            
        switch_t = -1

        prev_hid = torch.zeros(1, self.args.nagents, self.args.hid_size)
        
        # 初始化episode级别的通信统计累积字典
        episode_comm_stats = {
            'round1': {
                'total_elements': 0,         # 传输的消息数(元素数)
                'active_channels': 0,        # 活跃通信通道数（边数）
                'entropy': 0.0,              # 通信信息熵
                'max_entropy': 0.0,          # 最大熵
                'effective_bandwidth': 0.0,  # 有效带宽利用率
                'message_similarity': 0.0,   # 消息相似度
                'total_comm_num': 0,         # 通信次数（累加型）
            },
            'round2': {
                'total_elements': 0,         # 传输的消息数(元素数)
                'active_channels': 0,        # 活跃通信通道数（边数）
                'entropy': 0.0,              # 通信信息熵
                'max_entropy': 0.0,          # 最大熵
                'effective_bandwidth': 0.0,  # 有效带宽利用率
                'message_similarity': 0.0,   # 消息相似度
                'total_comm_num': 0,         # 通信次数（累加型）
            },
            'total_elements': 0,             # 总传输消息数
            'total_entropy': 0.0,            # 总信息熵
            'total_max_entropy': 0.0,        # 总最大熵
            'total_effective_bandwidth': 0.0, # 总有效带宽利用率
            'total_message_similarity': 0.0,  # 总消息相似度
            'total_comm_num': 0,             # 总通信次数（累加型）
            'active_channels': 0,            # 总活跃通信通道数
            'step_count': 0                   # 记录有效步数
        }

        for t in range(self.args.max_steps):
            misc = dict()

            if t == 0:
                prev_hid = self.policy_net.init_hidden(batch_size=state.shape[0])

            x = [state, prev_hid]
            action_out, value, prev_hid = self.policy_net(x, info)

            # 累加当前步骤的通信统计
            if 'comm_stats' in info and info['comm_stats'] is not None:
                episode_comm_stats['step_count'] += 1
                
                # 处理每轮通信的统计数据
                for key in ['round1', 'round2']:
                    if key in info['comm_stats']:
                        # 累加型指标
                        for subkey in ['total_elements', 'total_comm_num']:
                            if subkey in info['comm_stats'][key]:
                                episode_comm_stats[key][subkey] += info['comm_stats'][key][subkey]
                        
                        # 平均型指标 - 先累加后面再计算平均值
                        # entropy和max_entropy应该是平均型指标，因为它们表示每条消息的平均信息量，不应随步数增加而累加
                        for subkey in ['active_channels', 'effective_bandwidth', 'message_similarity', 'entropy', 'max_entropy']:
                            if subkey in info['comm_stats'][key]:
                                episode_comm_stats[key][subkey] += info['comm_stats'][key][subkey]
                
                # 累加总体指标
                if 'total_elements' in info['comm_stats']:
                    episode_comm_stats['total_elements'] += info['comm_stats']['total_elements']
                if 'total_comm_num' in info['comm_stats']:
                    episode_comm_stats['total_comm_num'] += info['comm_stats']['total_comm_num']
                
                # 平均型指标 - 先累加后面再计算平均值
                # 总熵和最大熵也应该是平均值，表示系统整体的平均信息含量
                if 'total_entropy' in info['comm_stats']:
                    episode_comm_stats['total_entropy'] += info['comm_stats']['total_entropy']
                if 'total_max_entropy' in info['comm_stats']:
                    episode_comm_stats['total_max_entropy'] += info['comm_stats']['total_max_entropy']
                if 'total_effective_bandwidth' in info['comm_stats']:
                    episode_comm_stats['total_effective_bandwidth'] += info['comm_stats']['total_effective_bandwidth']
                if 'total_message_similarity' in info['comm_stats']:
                    episode_comm_stats['total_message_similarity'] += info['comm_stats']['total_message_similarity']
                
                # 计算总活跃通道数 - 修正计算方式为两轮平均值
                if 'round1' in info['comm_stats'] and 'round2' in info['comm_stats']:
                    if 'active_channels' in info['comm_stats']['round1'] and 'active_channels' in info['comm_stats']['round2']:
                        # 修正计算方式为两轮平均
                        active_channels = (info['comm_stats']['round1']['active_channels'] + info['comm_stats']['round2']['active_channels']) / 2.0
                        episode_comm_stats['active_channels'] += active_channels
            #print('episode_comm_stats', episode_comm_stats)
            if (t + 1) % self.args.detach_gap == 0:
                prev_hid = (prev_hid[0].detach(), prev_hid[1].detach())

            action = select_action(self.args, action_out)
            action, actual = translate_action(self.args, self.env, action)
            next_state, reward, done, info = self.env.step(actual)

            if 'alive_mask' in info:
                misc['alive_mask'] = info['alive_mask'].reshape(reward.shape)
            else:
                misc['alive_mask'] = np.ones_like(reward)

            stat['reward'] = stat.get('reward', 0) + reward[:self.args.nfriendly]
            if hasattr(self.args, 'enemy_comm') and self.args.enemy_comm:
                stat['enemy_reward'] = stat.get('enemy_reward', 0) + reward[self.args.nfriendly:]

            done = done or t == self.args.max_steps - 1

            episode_mask = np.ones(reward.shape)
            episode_mini_mask = np.ones(reward.shape)

            if done:
                episode_mask = np.zeros(reward.shape)
            else:
                if 'is_completed' in info:
                    episode_mini_mask = 1 - info['is_completed'].reshape(-1)

            if should_display:
                self.env.display()

            trans = Transition(state, action, action_out, value, episode_mask, episode_mini_mask, next_state, reward, misc)
            episode.append(trans)
            state = next_state
            if done:
                break

        # 计算平均值
        if episode_comm_stats['step_count'] > 0:
            for key in ['round1', 'round2']:
                # 只对平均型指标计算平均值
                for subkey in ['active_channels', 'effective_bandwidth', 'message_similarity', 'entropy', 'max_entropy']:
                    if subkey in episode_comm_stats[key]:
                        episode_comm_stats[key][subkey] /= episode_comm_stats['step_count']
            
            # 只对平均型指标计算平均值
            for subkey in ['total_entropy', 'total_max_entropy', 'total_effective_bandwidth', 'total_message_similarity', 'active_channels']:
                if subkey in episode_comm_stats:
                    episode_comm_stats[subkey] /= episode_comm_stats['step_count']
            
        stat['num_steps'] = t + 1
        stat['steps_taken'] = stat['num_steps']

        if hasattr(self.env, 'reward_terminal'):
            reward = self.env.reward_terminal()

            episode[-1] = episode[-1]._replace(reward = episode[-1].reward + reward)
            stat['reward'] = stat.get('reward', 0) + reward[:self.args.nfriendly]
            if hasattr(self.args, 'enemy_comm') and self.args.enemy_comm:
                stat['enemy_reward'] = stat.get('enemy_reward', 0) + reward[self.args.nfriendly:]


        if hasattr(self.env, 'get_stat'):
            merge_stat(self.env.get_stat(), stat)
        return (episode, stat, episode_comm_stats)


    def compute_grad(self, batch):
        stat = dict()
        # num_actions: number of discrete actions in the action space
        num_actions = self.args.num_actions
        # dim_actions: number of action heads
        dim_actions = self.args.dim_actions

        n = self.args.nagents
        batch_size = len(batch.state)

        # rewards: [batch_size * n]
        rewards = torch.Tensor(batch.reward)
        # episode_mask: [batch_size * n]
        episode_masks = torch.Tensor(batch.episode_mask)
        # episode_mini_mask: [batch_size * n]
        episode_mini_masks = torch.Tensor(batch.episode_mini_mask)
        actions = torch.Tensor(batch.action)
        # actions: [batch_size * n * dim_actions] have been detached
        actions = actions.transpose(1, 2).view(-1, n, dim_actions)

        values = torch.cat(batch.value, dim=0)
        action_out = list(zip(*batch.action_out))
        action_out = [torch.cat(a, dim=0) for a in action_out]

        # alive_masks: [batch_size * n]
        alive_masks = torch.Tensor(np.concatenate([item['alive_mask'] for item in batch.misc])).view(-1)

        coop_returns = torch.Tensor(batch_size, n)
        ncoop_returns = torch.Tensor(batch_size, n)
        returns = torch.Tensor(batch_size, n)
        deltas = torch.Tensor(batch_size, n)
        advantages = torch.Tensor(batch_size, n)
        values = values.view(batch_size, n)

        prev_coop_return = 0
        prev_ncoop_return = 0
        prev_value = 0
        prev_advantage = 0

        for i in reversed(range(rewards.size(0))):
            coop_returns[i] = rewards[i] + self.args.gamma * prev_coop_return * episode_masks[i]
            ncoop_returns[i] = rewards[i] + self.args.gamma * prev_ncoop_return * episode_masks[i] * episode_mini_masks[i]

            prev_coop_return = coop_returns[i].clone()
            prev_ncoop_return = ncoop_returns[i].clone()

            returns[i] = (self.args.mean_ratio * coop_returns[i].mean()) \
                        + ((1 - self.args.mean_ratio) * ncoop_returns[i])


        for i in reversed(range(rewards.size(0))):
            advantages[i] = returns[i] - values.data[i]

        if self.args.normalize_rewards:
            advantages = (advantages - advantages.mean()) / advantages.std()
            
        # element of log_p_a: [(batch_size*n) * num_actions[i]]
        log_p_a = [action_out[i].view(-1, num_actions[i]) for i in range(dim_actions)]
        # actions: [(batch_size*n) * dim_actions]
        actions = actions.contiguous().view(-1, dim_actions)

        if self.args.advantages_per_action:
            # log_prob: [(batch_size*n) * dim_actions]
            log_prob = multinomials_log_densities(actions, log_p_a)
            # the log prob of each action head is multiplied by the advantage
            action_loss = -advantages.view(-1).unsqueeze(-1) * log_prob
            action_loss *= alive_masks.unsqueeze(-1)
        else:
            # log_prob: [(batch_size*n) * 1]
            log_prob = multinomials_log_density(actions, log_p_a)
            action_loss = -advantages.view(-1) * log_prob.squeeze()
            action_loss *= alive_masks

        action_loss = action_loss.sum()
        stat['action_loss'] = action_loss.item()

        # value loss term
        targets = returns
        value_loss = (values - targets).pow(2).view(-1)
        value_loss *= alive_masks
        value_loss = value_loss.sum()

        stat['value_loss'] = value_loss.item()
        loss = action_loss + self.args.value_coeff * value_loss

        # entropy regularization term
        entropy = 0
        for i in range(len(log_p_a)):
            entropy -= (log_p_a[i] * log_p_a[i].exp()).sum()
        stat['entropy'] = entropy.item()
        if self.args.entr > 0:
            loss -= self.args.entr * entropy

        loss.backward()

        return stat

    def run_batch(self, epoch):
        batch = []
        self.stats = dict()
        self.stats['num_episodes'] = 0

        # 初始化batch级别的通信统计累积字典
        batch_comm_stats = {
            'round1': {
                'total_elements': 0,         # 传输的消息数(元素数)
                'active_channels': 0,        # 活跃通信通道数（边数）
                'entropy': 0.0,              # 通信信息熵
                'max_entropy': 0.0,          # 最大熵
                'effective_bandwidth': 0.0,  # 有效带宽利用率
                'message_similarity': 0.0,   # 消息相似度
                'total_comm_num': 0,         # 通信次数（累加型）
            },
            'round2': {
                'total_elements': 0,         # 传输的消息数(元素数)
                'active_channels': 0,        # 活跃通信通道数（边数）
                'entropy': 0.0,              # 通信信息熵
                'max_entropy': 0.0,          # 最大熵
                'effective_bandwidth': 0.0,  # 有效带宽利用率
                'message_similarity': 0.0,   # 消息相似度
                'total_comm_num': 0,         # 通信次数（累加型）
            },
            'total_elements': 0,             # 总传输消息数
            'total_entropy': 0.0,            # 总信息熵
            'total_max_entropy': 0.0,        # 总最大熵
            'total_effective_bandwidth': 0.0, # 总有效带宽利用率
            'total_message_similarity': 0.0,  # 总消息相似度
            'total_comm_num': 0,             # 总通信次数（累加型）
            'active_channels': 0,            # 总活跃通信通道数
            'episode_count': 0                # 记录有效episode数
        }

        while len(batch) < self.args.batch_size:
            if self.args.batch_size - len(batch) <= self.args.max_steps:
                self.last_step = True
            episode, episode_stat, episode_comm_stats  = self.get_episode(epoch)
            merge_stat(episode_stat, self.stats)
            self.stats['num_episodes'] += 1

            # 累加通信统计
            if episode_comm_stats['step_count'] > 0:  # 只累加有效的episode
                batch_comm_stats['episode_count'] += 1
                
                # 处理每轮通信的统计数据
                for key in ['round1', 'round2']:
                    # 累加型指标
                    for subkey in ['total_elements', 'total_comm_num']:
                        if subkey in episode_comm_stats[key]:
                            batch_comm_stats[key][subkey] += episode_comm_stats[key][subkey]
                    
                    # 平均型指标 - 先累加后面再计算平均值
                    # 将熵作为平均型指标处理，因为熵表示消息的平均信息量
                    for subkey in ['active_channels', 'effective_bandwidth', 'message_similarity', 'entropy', 'max_entropy']:
                        if subkey in episode_comm_stats[key]:
                            batch_comm_stats[key][subkey] += episode_comm_stats[key][subkey]
                
                # 累加总体指标
                if 'total_elements' in episode_comm_stats:
                    batch_comm_stats['total_elements'] += episode_comm_stats['total_elements']
                if 'total_comm_num' in episode_comm_stats:
                    batch_comm_stats['total_comm_num'] += episode_comm_stats['total_comm_num']
                
                # 平均型指标 - 先累加后面再计算平均值
                for subkey in ['total_entropy', 'total_max_entropy', 'total_effective_bandwidth', 'total_message_similarity']:
                    if subkey in episode_comm_stats:
                        batch_comm_stats[subkey] += episode_comm_stats[subkey]
                    
                # 添加总活跃通道数的累加
                if 'active_channels' in episode_comm_stats:
                    batch_comm_stats['active_channels'] += episode_comm_stats['active_channels']
            
            batch += episode

        # 计算batch平均值
        if batch_comm_stats['episode_count'] > 0:
            for key in ['round1', 'round2']:
                # 只对平均型指标计算平均值
                for subkey in ['active_channels', 'effective_bandwidth', 'message_similarity', 'entropy', 'max_entropy']:
                    if subkey in batch_comm_stats[key]:
                        batch_comm_stats[key][subkey] /= batch_comm_stats['episode_count']
            
            # 只对平均型指标计算平均值
            for subkey in ['total_entropy', 'total_max_entropy', 'total_effective_bandwidth', 'total_message_similarity', 'active_channels']:
                if subkey in batch_comm_stats:
                    batch_comm_stats[subkey] /= batch_comm_stats['episode_count']
                
            # 累加型指标保持原样，如total_elements和total_comm_num等
            # 不需要除以episode_count
        self.last_step = False
        self.stats['num_steps'] = len(batch)
        batch = Transition(*zip(*batch))
        return batch, self.stats, batch_comm_stats

    def train_batch(self, epoch):
        batch, stat, batch_comm_stats  = self.run_batch(epoch)
        self.optimizer.zero_grad()

        s = self.compute_grad(batch)
        merge_stat(s, stat)
#         for name, param in self.policy_net.named_parameters():
#             print(name)
#             print(param.grad)
        for p in self.params:
            if p._grad is not None:
                p._grad.data /= stat['num_steps']
        self.optimizer.step()

        return stat, batch_comm_stats

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state):
        self.optimizer.load_state_dict(state)
