import sys
import gym
import ic3net_envs
from env_wrappers import *

def _make_env(env_id):
    """Create a raw gym env without the passive env checker wrapper."""
    env = gym.make(env_id)
    # gym >= 0.22 wraps envs in EnvChecker which rejects multi-agent tuple obs.
    # Unwrap to the base env so the checker never fires.
    if hasattr(env, 'unwrapped'):
        env = env.unwrapped
    return env
def init(env_name, args, final_init=True):
    if env_name == 'predator_prey':
        env = _make_env('PredatorPrey-v0')
        if args.display:
            env.init_curses()
        env.multi_agent_init(args)
        env = GymWrapper(env)
    elif env_name == 'traffic_junction':
        env = _make_env('TrafficJunction-v0')
        if args.display:
            env.init_curses()
        env.multi_agent_init(args)
        env = GymWrapper(env)
    elif env_name == 'grf':
        env = _make_env('GRFWrapper-v0')
        env.multi_agent_init(args)
        env = GymWrapper(env)

    else:
        raise RuntimeError("wrong env name")

    return env