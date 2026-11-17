import time
from utils import *
import torch
import torch.multiprocessing as mp

class MultiProcessWorker(mp.Process):
    # TODO: Make environment init threadsafe
    def __init__(self, id, trainer_maker, comm, seed, *args, **kwargs):
        self.id = id
        self.seed = seed
        super(MultiProcessWorker, self).__init__()
        self.trainer = trainer_maker()
        self.comm = comm

    def run(self):
        torch.manual_seed(self.seed + self.id + 1)
        np.random.seed(self.seed + self.id + 1)

        while True:
            task = self.comm.recv()
            if type(task) == list:
                task, epoch = task

            if task == 'quit':
                return
            elif task == 'run_batch':
                # 获取三个返回值
                batch, stat, batch_comm_stats = self.trainer.run_batch(epoch)
                self.trainer.optimizer.zero_grad()
                s = self.trainer.compute_grad(batch, stat, batch_comm_stats)
                merge_stat(s, stat)
                # 将stat和batch_comm_stats分开发送回主进程
                self.comm.send((stat, batch_comm_stats))
            elif task == 'send_grads':
                grads = []
                for p in self.trainer.params:
                    if p._grad is not None:
                        grads.append(p._grad.data)

                self.comm.send(grads)


class MultiProcessTrainer(object):
    def __init__(self, args, trainer_maker):
        self.comms = []
        self.trainer = trainer_maker()
        # itself will do the same job as workers
        self.nworkers = args.nprocesses - 1
        for i in range(self.nworkers):
            comm, comm_remote = mp.Pipe()
            self.comms.append(comm)
            worker = MultiProcessWorker(i, trainer_maker, comm_remote, seed=args.seed)
            worker.start()
        self.grads = None
        self.worker_grads = None
        self.is_random = args.random

    def quit(self):
        for comm in self.comms:
            comm.send('quit')

    def obtain_grad_pointers(self):
        # only need perform this once
        if self.grads is None:
            self.grads = []
            for p in self.trainer.params:
                if p._grad is not None:
                    self.grads.append(p._grad.data)

        if self.worker_grads is None:
            self.worker_grads = []
            for comm in self.comms:
                comm.send('send_grads')
                self.worker_grads.append(comm.recv())

    def train_batch(self, epoch):
        # run workers in parallel
        for comm in self.comms:
            comm.send(['run_batch', epoch])

        # run its own trainer
        batch, stat, main_batch_comm_stats = self.trainer.run_batch(epoch)
        self.trainer.optimizer.zero_grad()
        s = self.trainer.compute_grad(batch,stat,main_batch_comm_stats)
        merge_stat(s, stat)

        # 保存主进程的通信统计数据
        if main_batch_comm_stats is not None:
            # 创建一个副本以便安全操作
            main_comm_stats = main_batch_comm_stats
        else:
            main_comm_stats = None

        # check if workers are finished
        worker_stats = []  # 存储普通统计数据
        worker_comm_stats = []  # 专门存储通信统计数据
        for comm in self.comms:
            # 接收工作进程发回的元组
            worker_data = comm.recv()
            
            # 如果是元组，则包含stat和batch_comm_stats
            if isinstance(worker_data, tuple) and len(worker_data) == 2:
                s, worker_batch_comm_stats = worker_data
                if worker_batch_comm_stats is not None:
                    worker_comm_stats.append(worker_batch_comm_stats)
            else:
                # 兼容旧版本
                s = worker_data
                
            # 合并普通统计数据
            merge_stat(s, stat)

        # 对通信统计数据进行处理
        if main_comm_stats is not None and worker_comm_stats:
            main_ep = main_comm_stats.get('episode_count', 0)
            # 创建一个新的合并统计字典，基于主进程的结构初始化
            merged_comm_stats = {
                'round1': {
                    # 累加型指标
                    'total_elements': main_comm_stats['round1'].get('total_elements', 0),
                    'total_comm_num': main_comm_stats['round1'].get('total_comm_num', 0),
                    # 平均型指标 - 按episode_count加权平均
                    'active_channels': main_comm_stats['round1'].get('active_channels', 0.0) * main_ep,
                    'entropy': main_comm_stats['round1'].get('entropy', 0.0) * main_ep,
                    'max_entropy': main_comm_stats['round1'].get('max_entropy', 0.0) * main_ep,
                    'effective_bandwidth': main_comm_stats['round1'].get('effective_bandwidth', 0.0) * main_ep,
                    'message_similarity': main_comm_stats['round1'].get('message_similarity', 0.0) * main_ep,
                },
                'round2': {
                    # 累加型指标
                    'total_elements': main_comm_stats['round2'].get('total_elements', 0),
                    'total_comm_num': main_comm_stats['round2'].get('total_comm_num', 0),
                    # 平均型指标 - 按episode_count加权平均
                    'active_channels': main_comm_stats['round2'].get('active_channels', 0.0) * main_ep,
                    'entropy': main_comm_stats['round2'].get('entropy', 0.0) * main_ep,
                    'max_entropy': main_comm_stats['round2'].get('max_entropy', 0.0) * main_ep,
                    'effective_bandwidth': main_comm_stats['round2'].get('effective_bandwidth', 0.0) * main_ep,
                    'message_similarity': main_comm_stats['round2'].get('message_similarity', 0.0) * main_ep,
                },
                'round3': {
                    # 累加型指标
                    'total_elements': main_comm_stats['round3'].get('total_elements', 0),
                    'total_comm_num': main_comm_stats['round3'].get('total_comm_num', 0),
                    # 平均型指标 - 按episode_count加权平均
                    'active_channels': main_comm_stats['round3'].get('active_channels', 0.0) * main_ep,
                    'entropy': main_comm_stats['round3'].get('entropy', 0.0) * main_ep,
                    'max_entropy': main_comm_stats['round3'].get('max_entropy', 0.0) * main_ep,
                    'effective_bandwidth': main_comm_stats['round3'].get('effective_bandwidth', 0.0) * main_ep,
                    'message_similarity': main_comm_stats['round3'].get('message_similarity', 0.0) * main_ep,
                },
                # 累加型指标
                'total_elements': main_comm_stats.get('total_elements', 0),
                'total_comm_num': main_comm_stats.get('total_comm_num', 0),
                # 平均型指标 - 按episode_count加权平均
                'active_channels': main_comm_stats.get('active_channels', 0.0) * main_ep,
                'total_entropy': main_comm_stats.get('total_entropy', 0.0) * main_ep,
                'total_max_entropy': main_comm_stats.get('total_max_entropy', 0.0) * main_ep,
                'total_effective_bandwidth': main_comm_stats.get('total_effective_bandwidth', 0.0) * main_ep,
                'total_message_similarity': main_comm_stats.get('total_message_similarity', 0.0) * main_ep,
                # 总episode计数
                'episode_count': main_ep
            }
            
            # 累加工作进程的统计数据
            for worker_comm_stat in worker_comm_stats:
                if worker_comm_stat.get('episode_count', 0) > 0:
                    worker_episode_count = worker_comm_stat.get('episode_count', 0)
                    
                    # 处理每轮通信的统计数据
                    for key in ['round1', 'round2', 'round3']:
                        if key in worker_comm_stat:
                            # 累加型指标
                            for subkey in ['total_elements', 'total_comm_num']:
                                if subkey in worker_comm_stat[key]:
                                    merged_comm_stats[key][subkey] += worker_comm_stat[key][subkey]
                            
                            # 平均型指标 - 按episode_count加权
                            for subkey in ['active_channels', 'entropy', 'max_entropy', 'effective_bandwidth', 'message_similarity']:
                                if subkey in worker_comm_stat[key]:
                                    merged_comm_stats[key][subkey] += worker_comm_stat[key][subkey] * worker_episode_count
                    
                    # 处理总体累加型指标
                    merged_comm_stats['total_elements'] += worker_comm_stat.get('total_elements', 0)
                    merged_comm_stats['total_comm_num'] += worker_comm_stat.get('total_comm_num', 0)
                    
                    # 处理总体平均型指标 - 按episode_count加权
                    for subkey in ['active_channels', 'total_entropy', 'total_max_entropy', 'total_effective_bandwidth', 'total_message_similarity']:
                        if subkey in worker_comm_stat:
                            merged_comm_stats[subkey] += worker_comm_stat[subkey] * worker_episode_count
                    
                    # 累加episode计数
                    merged_comm_stats['episode_count'] += worker_episode_count
            
            # 计算平均值 - 只对平均型指标计算平均值
            if merged_comm_stats['episode_count'] > 0:
                for key in ['round1', 'round2', 'round3']:
                    # 对平均型指标计算加权平均值
                    for subkey in ['active_channels', 'entropy', 'max_entropy', 'effective_bandwidth', 'message_similarity']:
                        if merged_comm_stats[key][subkey] != 0:
                            merged_comm_stats[key][subkey] /= merged_comm_stats['episode_count']
                
                # 对总体平均型指标计算加权平均值
                for subkey in ['active_channels', 'total_entropy', 'total_max_entropy', 'total_effective_bandwidth', 'total_message_similarity']:
                    if merged_comm_stats[subkey] != 0:
                        merged_comm_stats[subkey] /= merged_comm_stats['episode_count']
            
            # 保存最终合并后的结果
            self.batch_comm_stats = merged_comm_stats
        elif main_comm_stats is not None:
            # 如果没有工作进程的数据，直接使用主进程的数据
            self.batch_comm_stats = main_comm_stats
        else:
            self.batch_comm_stats = None

        # add gradients of workers
        self.obtain_grad_pointers()
        for i in range(len(self.grads)):
            for g in self.worker_grads:
                self.grads[i] += g[i]
            self.grads[i] /= stat['num_steps']

        self.trainer.optimizer.step()
        return stat, self.batch_comm_stats

    def state_dict(self):
        return self.trainer.state_dict()

    def load_state_dict(self, state):
        self.trainer.load_state_dict(state)