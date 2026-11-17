from collections import namedtuple
from inspect import getargspec
import numpy as np
import torch
from torch import optim
import torch.nn as nn
from utils import *
from action_utils import *

Transition = namedtuple('Transition', ('state', 'action', 'action_out', 'value', 'episode_mask', 'episode_mini_mask', 'next_state',
                                       'reward', 'misc'))


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
        
        # **固定学习率 - 绝不改变**
        # 保存初始学习率，用于验证
        self.fixed_lr = args.lrate
        self.scheduler = None  # 禁用学习率调度器
        
        print(f"[Trainer] 学习率固定为: {self.fixed_lr}")
        print(f"[Trainer] 学习率调度器: 禁用")
        # # 如果需要学习率衰减，使用更温和的策略：
        # self.scheduler = optim.lr_scheduler.StepLR(
        #     self.optimizer, 
        #     step_size=1000,  # 每1000个epoch
        #     gamma=0.9        # 只降低10%
        # )
        
        # 确保nprocesses参数存在
        if not hasattr(self.args, 'nprocesses'):
            self.args.nprocesses = 1
    
    def _update_ci_statistics(self, current_stats, samples_with, samples_without, comm_mask):
        """
        使用 Welford 算法在线更新 CI 统计量（只传递小的统计数据，不传递大的 tensor）
        
        Args:
            current_stats: 当前统计量字典（如果是第一次则为 None）
            samples_with: [batch, n, n, hid_size]
            samples_without: [batch, n, n, hid_size]
            comm_mask: [batch, n, n] 或 None
            
        Returns:
            updated_stats: 更新后的统计量字典（只包含小的数值数组）
        """
        import numpy as np
        
        # 正确解析维度：samples_with 应该是 [batch, n, n, hid_size]
        if len(samples_with.shape) != 4:
            raise ValueError(f"Expected samples_with to have 4 dimensions [batch, n, n, hid_size], got shape {samples_with.shape}")
        
        batch_size = samples_with.shape[0]
        n = samples_with.shape[1]
        hid_size = samples_with.shape[3]  # 最后一维是 hid_size
        
        # 初始化统计量
        if current_stats is None:
            current_stats = {
                'count': np.zeros((n, n)),
                'mean_with': np.zeros((n, n, hid_size)),
                'mean_without': np.zeros((n, n, hid_size)),
                'M2_with': np.zeros((n, n, hid_size)),
                'M2_without': np.zeros((n, n, hid_size))
            }
        
        # 将 tensor 转换为 numpy（在 worker 进程中）
        samples_with_np = samples_with.detach().cpu().numpy()
        samples_without_np = samples_without.detach().cpu().numpy()
        if comm_mask is not None:
            comm_mask_np = comm_mask.detach().cpu().numpy()
        else:
            # 如果没有 mask，通过差异判断
            diff = np.abs(samples_with_np - samples_without_np).sum(axis=-1)
            comm_mask_np = (diff > 1e-6).astype(float)
        
        # Welford 在线更新（只更新有通信的链路）
        for b in range(batch_size):
            for i in range(n):
                for j in range(n):
                    if i != j and comm_mask_np[b, i, j] > 0.5:
                        # 更新计数
                        current_stats['count'][i, j] += 1
                        count = current_stats['count'][i, j]
                        
                        # 更新 with 的均值和 M2
                        sample_with = samples_with_np[b, i, j, :]  # 形状: [hid_size]
                        delta_with = sample_with - current_stats['mean_with'][i, j]
                        current_stats['mean_with'][i, j] += delta_with / count
                        delta2_with = sample_with - current_stats['mean_with'][i, j]
                        current_stats['M2_with'][i, j] += delta_with * delta2_with
                        
                        # 更新 without 的均值和 M2
                        sample_without = samples_without_np[b, i, j, :]  # 形状: [hid_size]
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
        
        # 合并统计量（并行 Welford 算法）
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
            
        # 初始化通信统计字典 - 修改为只有一轮通信，新增因果影响指标
        self.batch_comm_stats = {
            'round1': {
                'total_elements': 0,         # 传输的消息数(元素数)
                'active_channels': 0,        # 活跃通信通道数（边数）
                'entropy': 0.0,              # 通信信息熵
                'max_entropy': 0.0,          # 最大熵
                'effective_bandwidth': 0.0,  # 有效带宽利用率
                'message_similarity': 0.0,   # 消息相似度
                'total_comm_num': 0,         # 通信次数
                'causal_influence': 0.0,     # 因果影响（新增）
                'hidden_state_change': 0.0,  # 隐藏状态变化（新增）
                'message_impact_ratio': 0.0, # 消息影响效率比率（新增）
                'pairwise_influence_matrix': [],  # 成对影响矩阵（新增）
            },
            'episode_count': 0               # 记录有效episode数
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
        
        # 初始化episode级别通信统计 - 修改为只有一轮通信，新增因果影响指标
        episode_comm_stats = {
            'round1': {
                'total_elements': 0,         # 传输的消息数(元素数)
                'active_channels': 0,        # 活跃通信通道数（边数）
                'entropy': 0.0,              # 通信信息熵
                'max_entropy': 0.0,          # 最大熵
                'effective_bandwidth': 0.0,  # 有效带宽利用率
                'message_similarity': 0.0,   # 消息相似度
                'total_comm_num': 0,         # 通信次数
                'causal_influence': 0.0,     # 因果影响（新增）
                'hidden_state_change': 0.0,  # 隐藏状态变化（新增）
                'message_impact_ratio': 0.0, # 消息影响效率比率（新增）
                'pairwise_influence_matrix': [],  # 成对影响矩阵（新增）
            },
            'step_count': 0                  # 记录有效步数
        }
            
        switch_t = -1

        prev_hid = torch.zeros(1, self.args.nagents, self.args.hid_size)

        for t in range(self.args.max_steps):
            misc = dict()
            if t == 0 and self.args.hard_attn and self.args.commnet:
                info['comm_action'] = np.zeros(self.args.nagents, dtype=int)
            
            # 将pruning_mask添加到info中
            if hasattr(self, 'current_pruning_mask') and self.current_pruning_mask is not None:
                info['pruning_mask'] = self.current_pruning_mask
            
            # recurrence over time
            if self.args.recurrent:
                if self.args.rnn_type == 'LSTM' and t == 0:
                    prev_hid = self.policy_net.init_hidden(batch_size=state.shape[0])

                x = [state, prev_hid]
                
                if self.args.gacomm:
                    action_out, value, prev_hid, comm_density = self.policy_net(x, info)
                else:
                    action_out, value, prev_hid = self.policy_net(x, info)

                if (t + 1) % self.args.detach_gap == 0:
                    if self.args.rnn_type == 'LSTM':
                        prev_hid = (prev_hid[0].detach(), prev_hid[1].detach())
                    else:
                        prev_hid = prev_hid.detach()
            else:
                x = state
                if self.args.gacomm:
                    action_out, value, comm_density = self.policy_net(x, info)
                else:
                    action_out, value = self.policy_net(x, info)             

            if self.args.gacomm:
                stat['density1'] = stat.get('density1', 0) + comm_density[0]
                stat['density2'] = stat.get('density2', 0) + comm_density[1]
                
            # 收集通信统计 - 只处理一轮通信，包括新增的因果影响指标
            if 'comm_stats' in info and info['comm_stats'] is not None:
                episode_comm_stats['step_count'] += 1
                
                # 处理第一轮通信的统计数据
                if 'round1' in info['comm_stats']:
                    # 累加型指标
                    for subkey in ['total_elements', 'total_comm_num']:
                        if subkey in info['comm_stats']['round1']:
                            episode_comm_stats['round1'][subkey] += info['comm_stats']['round1'][subkey]
                    
                    # 平均型指标 - 先累加后面再计算平均值（包括新增的因果影响指标）
                    for subkey in ['active_channels', 'entropy', 'max_entropy', 'effective_bandwidth', 
                                   'message_similarity', 'causal_influence', 'hidden_state_change', 
                                   'message_impact_ratio']:
                        if subkey in info['comm_stats']['round1']:
                            episode_comm_stats['round1'][subkey] += info['comm_stats']['round1'][subkey]
            
            # 收集反事实样本供 epoch 级别 CI 计算（如果存在）
            # 策略：在多进程模式下只传递统计量（均值、M2），而不是原始 tensor
            if 'counterfactual_samples' in info and info['counterfactual_samples'] is not None:
                # 将原始 tensor 转换为统计量
                if 'ci_statistics' not in episode_comm_stats:
                    episode_comm_stats['ci_statistics'] = None
                    
                sample = info['counterfactual_samples']
                if sample is not None and 'with' in sample and 'without' in sample:
                    samples_with = sample['with']  # [batch, n, n, hid_size]
                    samples_without = sample['without']
                    comm_mask = sample.get('comm_mask', None)
                    
                    # 在 worker 进程中更新统计量（Welford 算法）
                    episode_comm_stats['ci_statistics'] = self._update_ci_statistics(
                        episode_comm_stats['ci_statistics'],
                        samples_with, 
                        samples_without,
                        comm_mask
                    )
            
            action = select_action(self.args, action_out)
            action, actual = translate_action(self.args, self.env, action)
            next_state, reward, done, info = self.env.step(actual)

            # store comm_action in info for next step
            if self.args.hard_attn and self.args.commnet:
                info['comm_action'] = action[-1] if not self.args.comm_action_one else np.ones(self.args.nagents, dtype=int)

                stat['comm_action'] = stat.get('comm_action', 0) + info['comm_action'][:self.args.nfriendly]
                if hasattr(self.args, 'enemy_comm') and self.args.enemy_comm:
                    stat['enemy_comm']  = stat.get('enemy_comm', 0)  + info['comm_action'][self.args.nfriendly:]


            if 'alive_mask' in info:
                misc['alive_mask'] = info['alive_mask'].reshape(reward.shape)
            else:
                misc['alive_mask'] = np.ones_like(reward)

            # env should handle this make sure that reward for dead agents is not counted
            # reward = reward * misc['alive_mask']

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
                
        # 计算通信统计的平均值
        if episode_comm_stats['step_count'] > 0:
            # 只对平均型指标计算平均值（包括新增的因果影响指标）
            for subkey in ['active_channels', 'entropy', 'max_entropy', 'effective_bandwidth', 
                           'message_similarity', 'causal_influence', 'hidden_state_change', 
                           'message_impact_ratio']:
                if subkey in episode_comm_stats['round1']:
                    episode_comm_stats['round1'][subkey] /= episode_comm_stats['step_count']
                    
        stat['num_steps'] = t + 1
        stat['steps_taken'] = stat['num_steps']

        if hasattr(self.env, 'reward_terminal'):
            reward = self.env.reward_terminal()
            # We are not multiplying in case of reward terminal with alive agent
            # If terminal reward is masked environment should do
            # reward = reward * misc['alive_mask']

            episode[-1] = episode[-1]._replace(reward = episode[-1].reward + reward)
            stat['reward'] = stat.get('reward', 0) + reward[:self.args.nfriendly]
            if hasattr(self.args, 'enemy_comm') and self.args.enemy_comm:
                stat['enemy_reward'] = stat.get('enemy_reward', 0) + reward[self.args.nfriendly:]


        if hasattr(self.env, 'get_stat'):
            merge_stat(self.env.get_stat(), stat)
            
        # 统一返回通信统计结果
        return (episode, stat, episode_comm_stats)

    def compute_grad(self, batch):
        stat = dict()
        num_actions = self.args.num_actions
        dim_actions = self.args.dim_actions

        n = self.args.nagents
        batch_size = len(batch.state)

        rewards = torch.Tensor(batch.reward)
        episode_masks = torch.Tensor(batch.episode_mask)
        episode_mini_masks = torch.Tensor(batch.episode_mini_mask)
        actions = torch.Tensor(batch.action)
        actions = actions.transpose(1, 2).view(-1, n, dim_actions)

        # old_actions = torch.Tensor(np.concatenate(batch.action, 0))
        # old_actions = old_actions.view(-1, n, dim_actions)
        # print(old_actions == actions)

        # can't do batch forward.
        values = torch.cat(batch.value, dim=0)
        action_out = list(zip(*batch.action_out))
        action_out = [torch.cat(a, dim=0) for a in action_out]

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

        if self.args.continuous:
            action_means, action_log_stds, action_stds = action_out
            log_prob = normal_log_density(actions, action_means, action_log_stds, action_stds)
        else:
            # discrete actions
            log_p_a = [action_out[i].view(-1, num_actions[i]) for i in range(dim_actions)]
            actions = actions.contiguous().view(-1, dim_actions)

            if self.args.advantages_per_action:
                log_prob = multinomials_log_densities(actions, log_p_a)
            else:
                log_prob = multinomials_log_density(actions, log_p_a)

        if self.args.advantages_per_action:
            action_loss = -advantages.view(-1).unsqueeze(-1) * log_prob
            action_loss *= alive_masks.unsqueeze(-1)
        else:
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

        if not self.args.continuous:
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
        
        # 初始化batch级别通信统计 - 修改为只有一轮通信，新增因果影响指标
        batch_comm_stats = {
            'round1': {
                'total_elements': 0,         # 传输的消息数(元素数)
                'active_channels': 0,        # 活跃通信通道数（边数）
                'entropy': 0.0,              # 通信信息熵
                'max_entropy': 0.0,          # 最大熵
                'effective_bandwidth': 0.0,  # 有效带宽利用率
                'message_similarity': 0.0,   # 消息相似度
                'total_comm_num': 0,         # 通信次数
                'causal_influence': 0.0,     # 因果影响（新增）
                'hidden_state_change': 0.0,  # 隐藏状态变化（新增）
                'message_impact_ratio': 0.0, # 消息影响效率比率（新增）
                'pairwise_influence_matrix': [],  # 成对影响矩阵（新增）
            },
            'episode_count': 0               # 记录有效episode数
        }
            
        while len(batch) < self.args.batch_size:
            if self.args.batch_size - len(batch) <= self.args.max_steps:
                self.last_step = True
                
            # 统一执行get_episode，获取通信统计
            episode, episode_stat, episode_comm_stats = self.get_episode(epoch)
                
            # 累加通信统计（包括新增的因果影响指标）
            if episode_comm_stats['step_count'] > 0:  # 只累加有效的episode
                batch_comm_stats['episode_count'] += 1
                
                # 处理第一轮通信的统计数据
                # 累加型指标
                for subkey in ['total_elements', 'total_comm_num']:
                    if subkey in episode_comm_stats['round1']:
                        batch_comm_stats['round1'][subkey] += episode_comm_stats['round1'][subkey]
                
                # 平均型指标 - 先累加后面再计算平均值（包括新增的因果影响指标）
                for subkey in ['active_channels', 'entropy', 'max_entropy', 'effective_bandwidth', 
                               'message_similarity', 'causal_influence', 'hidden_state_change', 
                               'message_impact_ratio']:
                    if subkey in episode_comm_stats['round1']:
                        batch_comm_stats['round1'][subkey] += episode_comm_stats['round1'][subkey]
                
                # 收集 CI 统计量（而不是原始 tensor，适用于多进程）
                if 'ci_statistics' in episode_comm_stats and episode_comm_stats['ci_statistics'] is not None:
                    if 'ci_statistics' not in batch_comm_stats:
                        batch_comm_stats['ci_statistics'] = None
                    # 合并统计量
                    batch_comm_stats['ci_statistics'] = self._merge_ci_statistics(
                        batch_comm_stats['ci_statistics'],
                        episode_comm_stats['ci_statistics']
                    )
                
            merge_stat(episode_stat, self.stats)
            self.stats['num_episodes'] += 1
            batch += episode

        # 计算batch平均值
        if batch_comm_stats['episode_count'] > 0:
            # 只对平均型指标计算平均值（包括新增的因果影响指标）
            for subkey in ['active_channels', 'entropy', 'max_entropy', 'effective_bandwidth', 
                           'message_similarity', 'causal_influence', 'hidden_state_change', 
                           'message_impact_ratio']:
                if subkey in batch_comm_stats['round1']:
                    batch_comm_stats['round1'][subkey] /= batch_comm_stats['episode_count']
            
            # 累加型指标保持原样，如total_elements和total_comm_num等
            # 不需要除以episode_count

        self.last_step = False
        self.stats['num_steps'] = len(batch)
        batch = Transition(*zip(*batch))
        
        # 统一返回通信统计结果
        return batch, self.stats, batch_comm_stats

    # only used when nprocesses=1
    def train_batch(self, epoch, pruning_mask=None):
        # 存储pruning_mask以便在get_episode中使用
        self.current_pruning_mask = pruning_mask
        # 统一执行run_batch，获取通信统计
        batch, stat, batch_comm_stats = self.run_batch(epoch)
        
        # 每100个epoch验证一次学习率
        if epoch % 100 == 0:
            self.verify_fixed_lr()
            
        self.optimizer.zero_grad()

        s = self.compute_grad(batch)
        merge_stat(s, stat)
        for p in self.params:
            if p._grad is not None:
                p._grad.data /= stat['num_steps']
        
        # 添加梯度裁剪以防止梯度爆炸
        # 这对于MAGIC等复杂模型非常重要
        max_grad_norm = 10.0
        grad_norm = torch.nn.utils.clip_grad_norm_(self.params, max_grad_norm)
        
        # 检测梯度异常
        if torch.isnan(grad_norm) or torch.isinf(grad_norm):
            print(f"警告: 检测到异常梯度! grad_norm={grad_norm}")
            # 跳过这次更新
            return stat, batch_comm_stats
        
        # 检查参数是否包含NaN
        has_nan = False
        for p in self.params:
            if torch.isnan(p.data).any() or torch.isinf(p.data).any():
                print(f"警告: 参数包含NaN或Inf!")
                has_nan = True
                break
        
        if has_nan:
            print("跳过参数更新以避免崩溃")
            return stat, batch_comm_stats
        
        self.optimizer.step()
        
        # 更新后再次检查参数
        for p in self.params:
            if torch.isnan(p.data).any() or torch.isinf(p.data).any():
                print(f"严重警告: 优化器更新后参数包含NaN或Inf!")
                # 将参数重置为0（紧急措施）
                p.data.zero_()
                print("已将异常参数重置为0")

        # 统一返回通信统计结果
        return stat, batch_comm_stats

    def verify_fixed_lr(self):
        """验证学习率是否保持固定"""
        current_lr = self.optimizer.param_groups[0]['lr']
        if abs(current_lr - self.fixed_lr) > 1e-9:
            print(f"[WARNING] 学习率发生变化！期望: {self.fixed_lr}, 当前: {current_lr}")
            # 强制恢复固定学习率
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.fixed_lr
            print(f"[Trainer] 已强制恢复学习率为: {self.fixed_lr}")
        return current_lr
    
    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state):
        self.optimizer.load_state_dict(state)
        # 加载后强制恢复固定学习率
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.fixed_lr
        print(f"[Trainer] 从checkpoint恢复后，学习率已设置为: {self.fixed_lr}")
