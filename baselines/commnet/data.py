import sys
import gym
import ic3net_envs
from env_wrappers import *

def init(env_name, args, final_init=True):
    if env_name == 'predator_prey':
        env = gym.make('PredatorPrey-v0')
        if args.display:
            env.init_curses()
        env.multi_agent_init(args)
        env = GymWrapper(env)
    elif env_name == 'traffic_junction':
        env = gym.make('TrafficJunction-v0')
        if args.display:
            env.init_curses()
        env.multi_agent_init(args)
        env = GymWrapper(env)
    elif env_name == 'grf':
        env = gym.make('GRFWrapper-v0')
        env.multi_agent_init(args)
        env = GymWrapper(env)
    elif env_name == 'smac_v2':
        from new_env_wrappers import make_smacv2_env
        env = make_smacv2_env(args)
    elif env_name == 'mpe':
        from new_env_wrappers import make_mpe_env
        env = make_mpe_env(args)
    elif env_name == 'mamujoco':
        from new_env_wrappers import make_mamujoco_env
        env = make_mamujoco_env(args)
    else:
        raise RuntimeError(
            f"不支持的环境名称: '{env_name}'。"
            "可选: predator_prey, traffic_junction, grf, smac_v2, mpe, mamujoco"
        )

    return env
