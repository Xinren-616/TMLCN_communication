"""
新环境适配器：将 SMAC_v2、MPE (PettingZoo)、MAMuJoCo 封装为
与现有 GymWrapper 兼容的接口，供 train_with_online_shapley.py 使用。

每个 Wrapper 需实现：
  - observation_dim  (property)
  - num_actions      (property)
  - dim_actions      (property)
  - reset(epoch)     -> obs: torch.DoubleTensor [1, n_agents, obs_dim]
  - step(action)     -> (obs, reward, done, info)
  - reward_terminal()
  - get_stat()
"""

import sys
import os
import numpy as np
import torch


def _extract_discrete_actions(action, n_agents):
    """
    将 translate_action() 产出的离散动作转换为长度为 n_agents 的整数列表。

    常见输入形态：
      - list[np.ndarray(n_agents,)]              # dim_actions = 1
      - np.ndarray(dim_actions, n_agents)
      - np.ndarray(n_agents,)
      - 标量（异常场景）
    """
    if isinstance(action, torch.Tensor):
        action = action.cpu().numpy()

    arr = np.asarray(action)
    if arr.size == 0:
        values = np.zeros(n_agents, dtype=np.int64)
    else:
        arr = np.squeeze(arr)
        if arr.ndim == 0:
            values = np.full(n_agents, arr.item(), dtype=np.int64)
        elif arr.ndim == 1:
            values = arr.reshape(-1)
        else:
            # 多 action head 时优先使用第一个 head
            values = np.asarray(arr[0]).reshape(-1)

    if values.size < n_agents:
        pad_value = values[-1] if values.size > 0 else 0
        values = np.pad(values, (0, n_agents - values.size), constant_values=pad_value)

    return [int(v) for v in values[:n_agents]]


# ─────────────────────────────────────────────────────────────
# SMAC v2 Wrapper
# ─────────────────────────────────────────────────────────────

class SMACv2Wrapper:
    """
    将 SMAC v2 (StarCraft2Env) 适配为 MAGIC 框架的多智能体接口。

    SMAC v2 接口：
      env.reset()                  -> None（obs 通过 get_obs() 获取）
      env.step(actions_list)       -> reward, terminated, info
      env.get_obs()                -> list[np.ndarray], 长度 = n_agents
      env.get_avail_actions()      -> np.ndarray [n_agents, n_actions]
      env.n_agents                 -> int
      env.n_actions                -> int
      env.get_obs_size()           -> int
    """

    def __init__(self, env, episode_limit=None):
        self.env = env
        self._n_agents = env.n_agents
        self._obs_size = env.get_obs_size()
        self._n_actions = env.n_actions
        # episode_limit 由地图参数决定，供 trainer 设置 max_steps 使用
        self.episode_limit = episode_limit
        self.stat = {}

    @property
    def observation_dim(self):
        return self._obs_size

    @property
    def num_actions(self):
        return self._n_actions

    @property
    def dim_actions(self):
        # 每个 agent 每步选一个离散动作
        return 1

    def _get_obs_tensor(self):
        obs_list = self.env.get_obs()  # list of np.ndarray, each shape (obs_size,)
        obs_arr = np.stack(obs_list, axis=0)  # (n_agents, obs_size)
        obs_arr = obs_arr.reshape(1, self._n_agents, self._obs_size)
        return torch.from_numpy(obs_arr).double()

    def reset(self, epoch=None):
        self.env.reset()
        self.stat = {"battles_won": 0, "battles_game": 0}
        return self._get_obs_tensor()

    def step(self, action):
        """
        action: translate_action 返回的 actual，格式为 list[np.ndarray(n_agents,)]
        返回: (obs, reward, done, info)
        """
        actions_int = _extract_discrete_actions(action, self._n_agents)

        # SMACv2 中 action 0 为 no-op，仅死亡 agent 可用。
        # 若模型选了非法动作，则从该 agent 的合法动作中随机选一个替代。
        avail = self.env.get_avail_actions()  # shape: (n_agents, n_actions)
        for i in range(self._n_agents):
            if avail[i][actions_int[i]] == 0:
                legal = [a for a, v in enumerate(avail[i]) if v == 1]
                actions_int[i] = int(np.random.choice(legal))

        reward, terminated, info = self.env.step(actions_int)

        obs = self._get_obs_tensor()

        # reward 需为 shape (n_agents,) 的数组，供 trainer 做切片
        if isinstance(reward, (list, np.ndarray)):
            scalar_r = float(np.mean(reward))
        else:
            scalar_r = float(reward)
        reward = np.full(self._n_agents, scalar_r, dtype=np.float32)

        done = bool(terminated)
        if done:
            self.stat["battles_game"] = self.stat.get("battles_game", 0) + 1
            if info.get("battle_won", False):
                self.stat["battles_won"] = self.stat.get("battles_won", 0) + 1

        return obs, reward, done, info

    def reward_terminal(self):
        return np.zeros(self._n_agents)

    def get_stat(self):
        stat = dict(self.stat)
        won = stat.get("battles_won", 0)
        stat["success"] = float(won > 0)
        # 重置，避免跨 episode 累积导致 success 虚高
        self.stat = {"battles_won": 0, "battles_game": 0}
        return stat

    def display(self):
        pass

    def end_display(self):
        pass


# ─────────────────────────────────────────────────────────────
# MPE (PettingZoo) Wrapper
# ─────────────────────────────────────────────────────────────

class MPEWrapper:
    """
    将 PettingZoo MPE 并行 API (parallel_env) 适配为 MAGIC 框架接口。

    Success 定义（per-episode，episode 结束时更新）：
      simple_spread:  最后一步所有 landmark 被覆盖（agent 距 landmark < COVER_DIST）且无碰撞
      simple_tag:     对手奖励之和 > 0（发生捕获）
      其他:           总奖励 > 0
    """

    _COVER_DIST = 0.20  # agent.size(0.15) + landmark.size(0.05)
    _SSL_SUCCESS_DIST = 0.20  # listener-to-target threshold for simple_speaker_listener

    def __init__(self, env, scenario='simple_spread'):
        self.env = env
        self._scenario = scenario
        self.env.reset()
        self._agents = self.env.agents[:]
        self._n_agents = len(self._agents)

        sample_agent = self._agents[0]
        obs_space = self.env.observation_space(sample_agent)
        # Use the max obs size across all agents to handle heterogeneous spaces
        # (e.g. simple_speaker_listener: speaker=3, listener=11).
        self._obs_size = max(
            int(np.prod(self.env.observation_space(a).shape))
            for a in self._agents
        )

        act_space = self.env.action_space(sample_agent)
        self._n_actions = int(act_space.n) if hasattr(act_space, 'n') else int(act_space.shape[0])

        # 尝试拿到底层 world（用于 success 计算）
        self._world = self._get_world()
        # episode 内累计统计
        self._ep_collisions = 0
        self._ep_steps = 0
        self._ep_any_covered = False  # episode 内是否出现过全覆盖且无碰撞
        self.stat = {}

    def _get_world(self):
        """
        从 parallel_env 层层剥离，拿到 MPE world 对象。

        simple_spread_v3.parallel_env() 的包装层级：
          aec_to_parallel_wrapper          (.aec_env)
            └── OrderEnforcingWrapper      (.env)
                  └── SimpleEnv            (.world)
        """
        cur = self.env
        for _ in range(6):  # 最多剥 6 层
            world = getattr(cur, 'world', None)
            if world is not None:
                return world
            # 依次尝试各种内层属性名
            nxt = None
            for attr in ('aec_env', 'env', 'unwrapped'):
                nxt = getattr(cur, attr, None)
                if nxt is not None and nxt is not cur:
                    break
            if nxt is None or nxt is cur:
                break
            cur = nxt
        return None

    def _compute_step_success(self, rew_dict):
        """
        利用 world 对象计算本 step 后场景的 success 信号。
        返回 (success: float, collisions: int)
        """
        world = self._world
        if world is None:
            total = sum(float(v) for v in rew_dict.values() if v is not None)
            return (1.0 if total > 0 else 0.0), 0

        if self._scenario == 'simple_spread':
            agents = world.agents
            landmarks = world.landmarks

            all_covered = all(
                min(
                    np.sqrt(np.sum(np.square(a.state.p_pos - lm.state.p_pos)))
                    for a in agents
                ) < self._COVER_DIST
                for lm in landmarks
            )

            collisions = 0
            for i, a1 in enumerate(agents):
                for a2 in agents[i + 1:]:
                    d = np.sqrt(np.sum(np.square(a1.state.p_pos - a2.state.p_pos)))
                    if d < (a1.size + a2.size):
                        collisions += 1

            # success = 覆盖所有 landmark 即可（碰撞单独计入 collisions，不影响 success）
            success = 1.0 if all_covered else 0.0
            return success, collisions

        else:
            if self._scenario == 'simple_speaker_listener':
                # world.agents 顺序: [speaker(movable=False), listener(movable=True)]
                # 用 movable=True 或名字中含 'listener' 来定位真正的 listener
                if world is not None:
                    listener = None
                    for a in world.agents:
                        if getattr(a, 'movable', False) or 'listener' in getattr(a, 'name', ''):
                            listener = a
                            break
                    if listener is not None:
                        goal_b = getattr(listener, 'goal_b', None)
                        if goal_b is not None:
                            dist = np.sqrt(np.sum(np.square(
                                listener.state.p_pos - goal_b.state.p_pos
                            )))
                            return (1.0 if dist < self._SSL_SUCCESS_DIST else 0.0), 0
                # fallback：用 listener 的 per-step reward 作代理
                # reward = -dist(listener, target)，success 当 dist < _SSL_SUCCESS_DIST
                for k, v in rew_dict.items():
                    if 'listener' in str(k):
                        return (1.0 if float(v) > -self._SSL_SUCCESS_DIST else 0.0), 0
            total = sum(float(v) for v in rew_dict.values() if v is not None)
            return (1.0 if total > 0 else 0.0), 0

    @property
    def observation_dim(self):
        return self._obs_size

    @property
    def num_actions(self):
        return self._n_actions

    @property
    def dim_actions(self):
        return 1

    def _obs_dict_to_tensor(self, obs_dict):
        obs_list = []
        for agent in self._agents:
            if agent in obs_dict and obs_dict[agent] is not None:
                o = np.array(obs_dict[agent]).flatten()
            else:
                o = np.zeros(self._obs_size)
            # Pad to max obs size for heterogeneous spaces (e.g. speaker_listener)
            if len(o) < self._obs_size:
                o = np.concatenate([o, np.zeros(self._obs_size - len(o))])
            obs_list.append(o)
        obs_arr = np.stack(obs_list, axis=0).reshape(1, self._n_agents, self._obs_size)
        return torch.from_numpy(obs_arr).double()

    def reset(self, epoch=None):
        result = self.env.reset()
        obs_dict = result[0] if isinstance(result, tuple) else result
        self._agents = self.env.agents[:]
        self._ep_collisions = 0
        self._ep_steps = 0
        self._ep_any_covered = False
        self.stat = {}
        return self._obs_dict_to_tensor(obs_dict)

    def step(self, action):
        agent_actions = _extract_discrete_actions(action, len(self._agents))
        actions_dict = {
            agent: agent_actions[i]
            for i, agent in enumerate(self._agents)
            if agent in self.env.agents
        }

        obs_dict, rew_dict, term_dict, trunc_dict, info_dict = self.env.step(actions_dict)

        obs = self._obs_dict_to_tensor(obs_dict)

        reward_arr = np.array(
            [float(rew_dict.get(a, 0.0)) for a in self._agents],
            dtype=np.float32,
        )

        done = all(term_dict.get(a, True) or trunc_dict.get(a, False) for a in self._agents)

        # 计算本 step success 并累积
        step_success, step_collisions = self._compute_step_success(rew_dict)
        self._ep_collisions += step_collisions
        self._ep_steps += 1
        if step_success > 0:
            self._ep_any_covered = True

        # episode 结束时写入 stat，trainer 通过 get_stat() 取出
        if done:
            # success = episode 内至少有一步全覆盖且无碰撞
            self.stat['success'] = 1.0 if self._ep_any_covered else 0.0
            self.stat['collisions'] = self._ep_collisions
            self.stat['ep_steps'] = self._ep_steps

        self._agents = self.env.agents[:]
        return obs, reward_arr, done, info_dict

    def reward_terminal(self):
        return np.zeros(1)

    def get_stat(self):
        return dict(self.stat)

    def display(self):
        pass

    def end_display(self):
        pass


# ─────────────────────────────────────────────────────────────
# MAMuJoCo Wrapper
# ─────────────────────────────────────────────────────────────

class MAMuJoCoWrapper:
    """
    将 MAMuJoCo (MujocoMulti) 适配为 MAGIC 框架接口。

    MAMuJoCo 接口：
      env.reset()                -> list of obs per agent
      env.step(actions_list)     -> reward, done, info
                                   （reward 是标量，actions 是 list of np.ndarray）
      env.get_obs()              -> list of obs
      env.get_obs_size()         -> int
      env.n_agents               -> int
      env.n_actions              -> int（连续动作维度）
      env.action_space           -> list of gym Box spaces
    """

    def __init__(self, env):
        self.env = env
        self._n_agents = env.n_agents
        self._obs_size = env.get_obs_size()
        # n_actions 为每个 agent 的连续动作维度
        self._n_actions = env.n_actions
        self._action_dims = [env.action_space[i].shape[0] for i in range(self._n_agents)]
        self._action_high = [env.action_space[i].high for i in range(self._n_agents)]
        self._action_low = [env.action_space[i].low for i in range(self._n_agents)]
        self.stat = {}

    @property
    def observation_dim(self):
        return self._obs_size

    @property
    def num_actions(self):
        return self._n_actions

    @property
    def dim_actions(self):
        # 连续动作：每个 agent 输出 n_actions 维向量
        return self._n_actions

    def _obs_to_tensor(self, obs_list):
        obs_arr = np.stack([
            np.array(o).flatten()[:self._obs_size] if len(np.array(o).flatten()) >= self._obs_size
            else np.pad(np.array(o).flatten(), (0, self._obs_size - len(np.array(o).flatten())))
            for o in obs_list
        ], axis=0).reshape(1, self._n_agents, self._obs_size)
        return torch.from_numpy(obs_arr).double()

    def reset(self, epoch=None):
        obs_list = self.env.reset()
        self.stat = {}
        return self._obs_to_tensor(obs_list)

    def step(self, action):
        """
        action: Tensor 或 list，shape (n_agents, n_actions) 或扁平化版本。
        将其映射到每个 agent 的连续动作。
        """
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()

        action = np.array(action)
        if action.ndim == 1:
            action = action.reshape(self._n_agents, self._n_actions)

        actions_list = []
        for i in range(self._n_agents):
            a = np.clip(action[i], self._action_low[i], self._action_high[i])
            actions_list.append(a)

        reward, done, info = self.env.step(actions_list)

        obs_list = self.env.get_obs()
        obs = self._obs_to_tensor(obs_list)

        # reward 需为 shape (n_agents,) 的数组，供 trainer 做切片
        scalar_r = float(reward) if np.isscalar(reward) else float(np.mean(reward))
        reward_arr = np.full(self._n_agents, scalar_r, dtype=np.float32)

        return obs, reward_arr, bool(done), info

    def reward_terminal(self):
        return np.zeros(1)

    def get_stat(self):
        return dict(self.stat)

    def display(self):
        pass

    def end_display(self):
        pass


# ─────────────────────────────────────────────────────────────
# 工厂函数
# ─────────────────────────────────────────────────────────────

def make_smacv2_env(args):
    """
    根据 args 构建 SMAC v2 环境并返回 SMACv2Wrapper。

    需要的 args 字段：
      args.smac_map      (str, 默认 "10gen_terran")
      args.nagents       (int)

    地图分两类：
      - 动态生成地图（10gen_terran/zerg/protoss）：需要 team_gen + start_positions
        nagents 必须与 10gen_* 的设计规模（10）匹配；若指定其他值会给出警告。
      - 固定地图（5m_vs_6m, 2s3z, 3m, 8m 等）：n_agents 由地图文件决定，
        nagents 参数须与地图的 n_agents 一致，否则自动修正并打印警告。

    wrapper 会把地图的 episode_limit 写入 args.max_steps（若 args 尚未显式设置），
    确保 trainer 的截断步数与地图设计的 episode 长度对齐。
    """
    new_env_root = os.path.join(
        os.path.dirname(__file__), '..', '..', '..', 'New_env', 'SMAC_v2'
    )
    new_env_root = os.path.abspath(new_env_root)
    if new_env_root not in sys.path:
        sys.path.insert(0, new_env_root)

    from smacv2.env.starcraft2.wrapper import StarCraftCapabilityEnvWrapper
    from smacv2.env.starcraft2.maps import get_map_params

    map_name = getattr(args, 'smac_map', '10gen_terran')
    n_units = getattr(args, 'nagents', 10)

    map_params = get_map_params(map_name)
    map_type = map_params.get("map_type", "")
    map_n_agents = map_params.get("n_agents", None)
    episode_limit = map_params.get("limit", 200)

    # 动态生成地图需要 team_gen 来随机组队
    gen_map_types = {"terran_gen", "zerg_gen", "protoss_gen"}
    is_gen_map = map_type in gen_map_types

    if is_gen_map:
        # 动态地图：nagents 可自由指定（决定每队单位数量）
        if map_type == "terran_gen":
            unit_types = ["marine", "marauder", "medivac"]
            weights = [0.45, 0.45, 0.1]
            exception_unit_types = ["medivac"]
        elif map_type == "zerg_gen":
            unit_types = ["zergling", "baneling", "hydralisk"]
            weights = [0.45, 0.45, 0.1]
            exception_unit_types = ["baneling"]
        else:  # protoss_gen
            unit_types = ["stalker", "zealot", "colossus"]
            weights = [0.45, 0.45, 0.1]
            exception_unit_types = ["colossus"]

        capability_config = {
            "n_units": n_units,
            "n_enemies": n_units,
            "team_gen": {
                "dist_type": "weighted_teams",
                "unit_types": unit_types,
                "weights": weights,
                "observe": True,
                "exception_unit_types": exception_unit_types,
            },
            "start_positions": {
                "dist_type": "surrounded_and_reflect",
                "p": 0.5,
                "map_x": 32,
                "map_y": 32,
            },
        }

        env = StarCraftCapabilityEnvWrapper(
            capability_config=capability_config,
            map_name=map_name,
            debug=False,
            conic_fov=False,
            use_unit_ranges=True,
            min_attack_range=2,
            obs_own_pos=True,
            fully_observable=False,
        )
    else:
        # 固定地图（5m_vs_6m, 2s3z, 3m, 8m 等）
        # 注意：smacv2 包的固定地图文件与 smacv1 共用，缺少 smacv2 的 restart trigger，
        # 因此固定地图必须使用 smacv1 的 StarCraft2Env 原始接口。
        # 若需要 smacv2 风格的动态生成单位，请改用 10gen_* 系列动态地图。
        if map_n_agents is not None and n_units != map_n_agents:
            print(
                f"[SMACv2] 警告: 地图 '{map_name}' 固定有 {map_n_agents} 个友方单位，"
                f"但 --nagents 指定为 {n_units}。已自动修正为 {map_n_agents}。"
            )
            args.nagents = map_n_agents
            n_units = map_n_agents

        from smacv2.env.starcraft2.starcraft2 import StarCraft2Env
        env = StarCraft2Env(
            map_name=map_name,
            debug=False,
            obs_own_pos=True,
            fully_observable=False,
        )

    # 将地图的 episode_limit 同步到 args.max_steps（若用户未显式覆盖默认值 40）
    default_max_steps = 40
    current_max_steps = getattr(args, 'max_steps', default_max_steps)
    if current_max_steps == default_max_steps:
        print(
            f"[SMACv2] 自动设置 max_steps={episode_limit}（来自地图 '{map_name}' 的 episode limit）"
        )
        args.max_steps = episode_limit

    return SMACv2Wrapper(env, episode_limit=episode_limit)


def make_mpe_env(args):
    """
    根据 args 构建 MPE 环境并返回 MPEWrapper。

    需要的 args 字段：
      args.mpe_scenario  (str, 默认 "simple_spread")
      args.nagents       (int, 用于设置 MPE N 参数)
    """
    new_env_root = os.path.join(
        os.path.dirname(__file__), '..', '..', '..', 'New_env', 'MPE'
    )
    new_env_root = os.path.abspath(new_env_root)
    if new_env_root not in sys.path:
        sys.path.insert(0, new_env_root)

    scenario = getattr(args, 'mpe_scenario', 'simple_spread')
    n_agents = getattr(args, 'nagents', 3)
    max_cycles = getattr(args, 'max_steps', 25)

    if scenario == 'simple_spread':
        from pettingzoo.mpe import simple_spread_v3
        env = simple_spread_v3.parallel_env(
            N=n_agents,
            local_ratio=0.5,
            max_cycles=max_cycles,
            continuous_actions=False,
        )
    elif scenario == 'simple_tag':
        from pettingzoo.mpe import simple_tag_v3
        env = simple_tag_v3.parallel_env(
            num_good=n_agents,
            num_adversaries=1,
            num_obstacles=2,
            max_cycles=max_cycles,
            continuous_actions=False,
        )
    elif scenario == 'simple_adversary':
        from pettingzoo.mpe import simple_adversary_v3
        env = simple_adversary_v3.parallel_env(
            N=n_agents,
            max_cycles=max_cycles,
            continuous_actions=False,
        )
    elif scenario == 'simple_speaker_listener':
        from pettingzoo.mpe import simple_speaker_listener_v4
        env = simple_speaker_listener_v4.parallel_env(
            max_cycles=max_cycles,
            continuous_actions=False,
        )
    else:
        raise ValueError(f"不支持的 MPE 场景: {scenario}，可选: simple_spread, simple_tag, simple_adversary, simple_speaker_listener")

    return MPEWrapper(env, scenario=scenario)


def make_mamujoco_env(args):
    """
    根据 args 构建 MAMuJoCo 环境并返回 MAMuJoCoWrapper。

    需要的 args 字段：
      args.mamujoco_scenario  (str, 默认 "Ant-v2")
      args.mamujoco_agent_conf (str, 默认 "2x4")
      args.max_steps          (int, 用于 episode_limit)
    """
    new_env_root = os.path.join(
        os.path.dirname(__file__), '..', '..', '..', 'New_env', 'MAMuJoCo'
    )
    new_env_root = os.path.abspath(new_env_root)
    if new_env_root not in sys.path:
        sys.path.insert(0, new_env_root)

    from multiagent_mujoco.mujoco_multi import MujocoMulti

    scenario = getattr(args, 'mamujoco_scenario', 'Ant-v2')
    agent_conf = getattr(args, 'mamujoco_agent_conf', '2x4')
    episode_limit = getattr(args, 'max_steps', 1000)

    env_args = {
        "scenario": scenario,
        "agent_conf": agent_conf,
        "agent_obsk": 1,
        "episode_limit": episode_limit,
    }

    # MujocoMulti 需要通过 kwargs 传递参数
    class _Args:
        pass
    inner_args = _Args()
    inner_args.episode_limit = episode_limit

    env = MujocoMulti(env_args=env_args, args=inner_args)
    return MAMuJoCoWrapper(env)
