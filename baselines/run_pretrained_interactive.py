from os.path import exists
from pathlib import Path
import uuid
from red_gym_env import RedGymEnv
from stable_baselines3 import A2C, PPO
from stable_baselines3.common import env_checker
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback
import numpy as np
from pyboy import PyBoy

def make_env(rank, env_conf, seed=0):
    """
    Utility function for multiprocessed env.
    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the initial seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = RedGymEnv(env_conf)
        #env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init

if __name__ == '__main__':

    sess_path = Path(f'session_{str(uuid.uuid4())[:8]}')
    ep_length = 2**23

    #game_save_path = '../has_pokedex_nballs.state'
    game_save_path = "save_state_file_1.state"
    env_config = {
                'headless': False, 'save_final_state': True, 'early_stop': False,
                'action_freq': 24, 'init_state': game_save_path, 'max_steps': ep_length, 
                'print_rewards': True, 'save_video': False, 'fast_video': True, 'session_path': sess_path,
                'gb_path': '../ROM/PokemonRed.gb', 'debug': False, 'sim_frame_dist': 2_000_000.0, 'extra_buttons': True,'explore_weight': 3
            }
    
    num_cpu = 1 #64 #46  # Also sets the number of episodes per training iteration
    env = make_env(0, env_config)() #SubprocVecEnv([make_env(i, env_config) for i in range(num_cpu)])
    
    #env_checker.check_env(env)
    #file_name = 'session_4da05e87_main_good/poke_439746560_steps'
    #file_name = 'session_mytrain_240919_498aa23f/poke_163840_steps'
    #file_name = 'session_mytrain_240919_6a17abd4/poke_491520_steps'
    file_name = 'session_mytrain_stage2_240920_9d8e5ac3/poke_18350080_steps'
    
    print('\nloading checkpoint')
    model = PPO.load(file_name, env=env,ent_coef=0.02, custom_objects={'lr_schedule': 0, 'clip_range': 0})
    
    epsilon = 0.01
    #keyboard.on_press_key("M", toggle_agent)
    obs, info = env.reset()
    badge_cnt = int(info['badge'] / 100)
    env.set_epsilon(epsilon)
    while True:
        action = 7 # pass action
        try:
            with open("agent_enabled.txt", "r") as f:
                agent_enabled = f.readlines()[0].startswith("yes")
        except:
            agent_enabled = False
        if agent_enabled:
            action, _states = model.predict(obs, deterministic=False)
        obs, rewards, terminated, truncated, info = env.step(action)
        env.render()

        current_badge_cnt = int(info['badge'] / 100)
        if  current_badge_cnt > badge_cnt:
            badge_cnt = current_badge_cnt
            filename = f'save_state_file_{badge_cnt}.state'
            print(f'saving state to {filename}')
            env.save_state(filename)
        if truncated:
            break
    env.close()
