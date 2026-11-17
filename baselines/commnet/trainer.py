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
        
        # 初始化剪枝掩码
        self.current_pruning_mask = None
        
        # 初始化batch级别通信统计 (包含因果影响指标)
        self.batch_comm_stats = {
            'round1': {
                'total_elements': 0,         # 传输的消息数(元素数)
                'active_channels': 0,        # 活跃通信通道数（边数）
                'entropy': 0.0,              # 通信信息熵
                'max_entropy': 0.0,          # 最大熵
                'effective_bandwidth': 0.0,  # 有效带宽利用率
                'message_similarity': 0.0,   # 消息相似度
                'total_comm_num': 0,         # 通信次数（累加型）
                'causal_influence': 0.0,     # 因果影响（新增）
                'hidden_state_change': 0.0,  # 隐藏状态变化（新增）
                'message_impact_ratio': 0.0, # 消息影响效率比率（新增）
            },
            'episode_count': 0
        }
    
    def _update_ci_statistics(self, current_stats, samples_with, samples_without, comm_mask):
        """
        使用 Welford 算法在线更新 CI 统计量
        
        Args:
            current_stats: 当前统计量字典（如果是第一次则为 None）
            samples_with: [batch, n, n, hid_size]
            samples_without: [batch, n, n, hid_size]
            comm_mask: [batch, n, n] 或 None
            
        Returns:
            updated_stats: 更新后的统计量字典
        """
        import numpy as np
        
        # 解析维度
        if len(samples_with.shape) != 4:
            raise ValueError(f"Expected samples_with to have 4 dimensions [batch, n, n, hid_size], got shape {samples_with.shape}")
        
        batch_size = samples_with.shape[0]
        n = samples_with.shape[1]
        hid_size = samples_with.shape[3]
        
        # 初始化统计量
        if current_stats is None:
            current_stats = {
                'count': np.zeros((n, n)),
                'mean_with': np.zeros((n, n, hid_size)),
                'mean_without': np.zeros((n, n, hid_size)),
                'M2_with': np.zeros((n, n, hid_size)),
                'M2_without': np.zeros((n, n, hid_size))
            }
        
        # 将 tensor 转换为 numpy
        samples_with_np = samples_with.detach().cpu().numpy()
        samples_without_np = samples_without.detach().cpu().numpy()
        if comm_mask is not None:
            comm_mask_np = comm_mask.detach().cpu().numpy()
        else:
            diff = np.abs(samples_with_np - samples_without_np).sum(axis=-1)
            comm_mask_np = (diff > 1e-6).astype(float)
        
        # Welford 在线更新
        for b in range(batch_size):
            for i in range(n):
                for j in range(n):
                    if i != j and comm_mask_np[b, i, j] > 0.5:
                        # 更新计数
                        current_stats['count'][i, j] += 1
                        count = current_stats['count'][i, j]
                        
                        # 更新 with 的均值和 M2
                        sample_with = samples_with_np[b, i, j, :]
                        delta_with = sample_with - current_stats['mean_with'][i, j]
                        current_stats['mean_with'][i, j] += delta_with / count
                        delta2_with = sample_with - current_stats['mean_with'][i, j]
                        current_stats['M2_with'][i, j] += delta_with * delta2_with
                        
                        # 更新 without 的均值和 M2
                        sample_without = samples_without_np[b, i, j, :]
                        delta_without = sample_without - current_stats['mean_without'][i, j]
                        current_stats['mean_without'][i, j] += delta_without / count
                        delta2_without = sample_without - current_stats['mean_without'][i, j]
                        current_stats['M2_without'][i, j] += delta_without * delta2_without
        
        return current_stats
    
    def _merge_ci_statistics(self, stats1, stats2):
        """
        合并两个 CI 统计量（用于多进程模式下的汇总）
        使用并行 Welford 算法
        
        Args:
            stats1: 第一个统计量字典（可以为 None）
            stats2: 第二个统计量字典（可以为 None）
            
        Returns:
            merged_stats: 合并后的统计量
        """
        import numpy as np
        
        if stats1 is None:
            return stats2
        if stats2 is None:
            return stats1
        
        n = stats1['count'].shape[0]
        hid_size = stats1['mean_with'].shape[2]
        
        merged_stats = {
            'count': np.zeros((n, n)),
            'mean_with': np.zeros((n, n, hid_size)),
            'mean_without': np.zeros((n, n, hid_size)),
            'M2_with': np.zeros((n, n, hid_size)),
            'M2_without': np.zeros((n, n, hid_size))
        }
        
        # 合并统计量
        for i in range(n):
            for j in range(n):
                count1 = stats1['count'][i, j]
                count2 = stats2['count'][i, j]
                count_total = count1 + count2
                
                merged_stats['count'][i, j] = count_total
                
                if count_total == 0:
                    continue
                
                # 合并 with 的统计量
                mean1_with = stats1['mean_with'][i, j]
                mean2_with = stats2['mean_with'][i, j]
                M2_1_with = stats1['M2_with'][i, j]
                M2_2_with = stats2['M2_with'][i, j]
                
                delta_with = mean2_with - mean1_with
                merged_mean_with = (count1 * mean1_with + count2 * mean2_with) / count_total
                merged_M2_with = M2_1_with + M2_2_with + (delta_with ** 2) * count1 * count2 / count_total
                
                merged_stats['mean_with'][i, j] = merged_mean_with
                merged_stats['M2_with'][i, j] = merged_M2_with
                
                # 合并 without 的统计量
                mean1_without = stats1['mean_without'][i, j]
                mean2_without = stats2['mean_without'][i, j]
                M2_1_without = stats1['M2_without'][i, j]
                M2_2_without = stats2['M2_without'][i, j]
                
                delta_without = mean2_without - mean1_without
                merged_mean_without = (count1 * mean1_without + count2 * mean2_without) / count_total
                merged_M2_without = M2_1_without + M2_2_without + (delta_without ** 2) * count1 * count2 / count_total
                
                merged_stats['mean_without'][i, j] = merged_mean_without
                merged_stats['M2_without'][i, j] = merged_M2_without
        
        return merged_stats

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
        
        # 传递剪枝掩码到info中
        if hasattr(self, 'current_pruning_mask') and self.current_pruning_mask is not None:
            info['pruning_mask'] = self.current_pruning_mask
            
        switch_t = -1

        device = getattr(self.args, 'device', torch.device('cpu'))
        prev_hid = torch.zeros(1, self.args.nagents, self.args.hid_size, device=device)
        
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
                'causal_influence': 0.0,     # 因果影响（新增）
                'hidden_state_change': 0.0,  # 隐藏状态变化（新增）
                'message_impact_ratio': 0.0, # 消息影响效率比率（新增）
            },
            'step_count': 0                   # 记录有效步数
        }

        for t in range(self.args.max_steps):
            misc = dict()

            if t == 0:
                prev_hid = self.policy_net.init_hidden(batch_size=state.shape[0])
                if isinstance(prev_hid, (list, tuple)):
                    prev_hid = type(prev_hid)(h.to(device) for h in prev_hid)
                else:
                    prev_hid = prev_hid.to(device)

            x = [state.to(device), prev_hid]
            result = self.policy_net(x, info)
            # 处理GAComm返回4个值的情况
            if len(result) == 4:
                action_out, value, prev_hid, comm_density = result
            else:
                action_out, value, prev_hid = result

            # 累加当前步骤的通信统计
            if 'comm_stats' in info and info['comm_stats'] is not None:
                episode_comm_stats['step_count'] += 1
                
                # 处理round1通信的统计数据
                if 'round1' in info['comm_stats']:
                    # 累加型指标
                    for subkey in ['total_elements', 'total_comm_num']:
                        if subkey in info['comm_stats']['round1']:
                            episode_comm_stats['round1'][subkey] += info['comm_stats']['round1'][subkey]
                    
                    # 平均型指标 - 先累加后面再计算平均值（包括新增的因果影响指标）
                    for subkey in ['active_channels', 'effective_bandwidth', 'message_similarity', 'entropy', 'max_entropy',
                                   'causal_influence', 'hidden_state_change', 'message_impact_ratio']:
                        if subkey in info['comm_stats']['round1']:
                            episode_comm_stats['round1'][subkey] += info['comm_stats']['round1'][subkey]
            
            # 收集反事实样本供 epoch 级别 CI 计算（如果存在）
            if 'counterfactual_samples' in info and info['counterfactual_samples'] is not None:
                # 使用 Welford 算法在线更新统计量
                if 'ci_statistics' not in episode_comm_stats:
                    episode_comm_stats['ci_statistics'] = None
                    
                sample = info['counterfactual_samples']
                if sample is not None and 'with' in sample and 'without' in sample:
                    samples_with = sample['with']  # [1, n, n, hid_size]
                    samples_without = sample['without']
                    comm_mask = sample.get('comm_mask', None)
                    
                    # 在 worker 进程中更新统计量（Welford 算法）
                    episode_comm_stats['ci_statistics'] = self._update_ci_statistics(
                        episode_comm_stats['ci_statistics'],
                        samples_with, 
                        samples_without,
                        comm_mask
                    )

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
            state = next_state.to(device) if hasattr(next_state, 'to') else next_state
            if done:
                break

        # 计算平均值
        if episode_comm_stats['step_count'] > 0:
            # 只对平均型指标计算平均值（包括新增的因果影响指标）
            for subkey in ['active_channels', 'effective_bandwidth', 'message_similarity', 'entropy', 'max_entropy',
                           'causal_influence', 'hidden_state_change', 'message_impact_ratio']:
                if subkey in episode_comm_stats['round1']:
                    episode_comm_stats['round1'][subkey] /= episode_comm_stats['step_count']
            
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
        device = getattr(self.args, 'device', torch.device('cpu'))

        # rewards: [batch_size * n]
        rewards = torch.Tensor(batch.reward).to(device)
        # episode_mask: [batch_size * n]
        episode_masks = torch.Tensor(batch.episode_mask).to(device)
        # episode_mini_mask: [batch_size * n]
        episode_mini_masks = torch.Tensor(batch.episode_mini_mask).to(device)
        actions = torch.Tensor(batch.action).to(device)
        # actions: [batch_size * n * dim_actions] have been detached
        actions = actions.transpose(1, 2).view(-1, n, dim_actions)

        values = torch.cat(batch.value, dim=0).to(device)
        action_out = list(zip(*batch.action_out))
        action_out = [torch.cat(a, dim=0).to(device) for a in action_out]

        # alive_masks: [batch_size * n]
        alive_masks = torch.Tensor(np.concatenate([item['alive_mask'] for item in batch.misc])).view(-1).to(device)

        coop_returns = torch.zeros(batch_size, n, device=device)
        ncoop_returns = torch.zeros(batch_size, n, device=device)
        returns = torch.zeros(batch_size, n, device=device)
        deltas = torch.zeros(batch_size, n, device=device)
        advantages = torch.zeros(batch_size, n, device=device)
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
                'causal_influence': 0.0,     # 因果影响（新增）
                'hidden_state_change': 0.0,  # 隐藏状态变化（新增）
                'message_impact_ratio': 0.0, # 消息影响效率比率（新增）
            },
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
                
                # 处理round1通信的统计数据
                # 累加型指标
                for subkey in ['total_elements', 'total_comm_num']:
                    if subkey in episode_comm_stats['round1']:
                        batch_comm_stats['round1'][subkey] += episode_comm_stats['round1'][subkey]
                
                # 平均型指标 - 先累加后面再计算平均值（包括新增的因果影响指标）
                for subkey in ['active_channels', 'effective_bandwidth', 'message_similarity', 'entropy', 'max_entropy',
                               'causal_influence', 'hidden_state_change', 'message_impact_ratio']:
                    if subkey in episode_comm_stats['round1']:
                        batch_comm_stats['round1'][subkey] += episode_comm_stats['round1'][subkey]
                
                # 收集 CI 统计量
                if 'ci_statistics' in episode_comm_stats and episode_comm_stats['ci_statistics'] is not None:
                    if 'ci_statistics' not in batch_comm_stats:
                        batch_comm_stats['ci_statistics'] = None
                    # 合并统计量
                    batch_comm_stats['ci_statistics'] = self._merge_ci_statistics(
                        batch_comm_stats['ci_statistics'],
                        episode_comm_stats['ci_statistics']
                    )
            
            batch += episode

        # 计算batch平均值
        if batch_comm_stats['episode_count'] > 0:
            # 只对平均型指标计算平均值（包括新增的因果影响指标）
            for subkey in ['active_channels', 'effective_bandwidth', 'message_similarity', 'entropy', 'max_entropy',
                           'causal_influence', 'hidden_state_change', 'message_impact_ratio']:
                if subkey in batch_comm_stats['round1']:
                    batch_comm_stats['round1'][subkey] /= batch_comm_stats['episode_count']
            
            # 累加型指标保持原样，如total_elements和total_comm_num等
            # 不需要除以episode_count
        self.last_step = False
        self.stats['num_steps'] = len(batch)
        batch = Transition(*zip(*batch))
        return batch, self.stats, batch_comm_stats

    def train_batch(self, epoch, pruning_mask=None):
        # 存储pruning_mask以便在get_episode中使用
        self.current_pruning_mask = pruning_mask
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
