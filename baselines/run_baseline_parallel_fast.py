from os.path import exists
from pathlib import Path
import uuid
from red_gym_env import RedGymEnv
from stable_baselines3 import PPO
from stable_baselines3.common import env_checker
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from tensorboard_callback import TensorboardCallback

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
        env.reset(seed=(seed + rank))
        return env
    set_random_seed(seed)
    return _init

if __name__ == '__main__':

    use_wandb_logging = False
    ep_length = 2048 * 20
    sess_id = str(uuid.uuid4())[:8]
    sess_path = Path(f'session_mytrain_stage2_240920_{sess_id}')
    print(sess_path)

    game_save_path = "save_state_file_1.state"
    env_config = {
                'headless': True, 'save_final_state': True, 'early_stop': False,
                'action_freq': 24, 'init_state': game_save_path, 'max_steps': ep_length, 
                'print_rewards': True, 'save_video': False, 'fast_video': True, 'session_path': sess_path,
                'gb_path': '../ROM/PokemonRed.gb', 'debug': False, 'sim_frame_dist': 2_000_000.0, 
                'use_screen_explore': True, 'extra_buttons': True,
                'explore_weight': 3 # 2.5
            }
    
    print(env_config)
    
    num_cpu = 8  # Also sets the number of episodes per training iteration
    env = SubprocVecEnv([make_env(i, env_config) for i in range(num_cpu)])
    
    checkpoint_callback = CheckpointCallback(save_freq=ep_length, save_path=sess_path,
                                     name_prefix='poke')
    
    #callbacks = [checkpoint_callback, TensorboardCallback(log_dir=sess_path)]
    callbacks = [checkpoint_callback]

    if use_wandb_logging:
        import wandb
        from wandb.integration.sb3 import WandbCallback
        run = wandb.init(
            project="pokemon-train",
            id=sess_id,
            config=env_config,
            sync_tensorboard=True,  
            monitor_gym=True,  
            save_code=True,
        )
        callbacks.append(WandbCallback())

    #env_checker.check_env(env)
    # put a checkpoint here you want to start from
    #file_name = 'session_4da05e87_main_good/poke_439746560_steps'
    #file_name = 'session_mytrain_240919_498aa23f/poke_163840_steps'
    file_name = 'session_mytrain_240919_6a17abd4/poke_491520_steps'

    if exists(file_name + '.zip'):
        print('\nloading checkpoint')
        model = PPO.load(file_name, env=env,ent_coef=0.02, custom_objects={'lr_schedule': 0, 'clip_range': 0, 'bytes': bytes})
        model.n_steps = ep_length
        model.n_envs = num_cpu
        model.rollout_buffer.buffer_size = ep_length
        model.rollout_buffer.n_envs = num_cpu
        model.rollout_buffer.reset()
    else:
        model = PPO('CnnPolicy', env, verbose=1, n_steps=ep_length // 8, batch_size=128, n_epochs=3, gamma=0.998, tensorboard_log=sess_path,seed=100, device='cuda')


    # run for up to 5k episodes
    model.learn(total_timesteps=(ep_length)*num_cpu*5000, callback=CallbackList(callbacks))

    if use_wandb_logging:
        run.finish()
