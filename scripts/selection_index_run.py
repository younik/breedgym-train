from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
from breeding_gym.utils.paths import DATA_PATH
from wandb.integration.sb3 import WandbCallback
from breeding_gym.vector import VecBreedingGym, SelectionValues
from callbacks.eval_callback import MyEvalCallback
from networks.features_extractors import CNNFeaturesExtractor, NoFeaturesExtractor
from networks.selection_model import SelectionAC

from wrappers.print_action import PrintActionWrapper
from wrappers.vector.adapt_vec_env import AdaptVecEnv

def main(config):     
    train_envs = VecBreedingGym(
        n_envs=32,
        initial_population=DATA_PATH.joinpath(config.genome),
        genetic_map=DATA_PATH.joinpath(config.genetic_map),
        individual_per_gen=210,
    )
    train_envs = SelectionValues(train_envs)
    train_envs = AdaptVecEnv(train_envs)
    train_envs = VecMonitor(train_envs)
    train_envs = VecNormalize(train_envs, norm_obs=False)
        
    eval_env = VecBreedingGym(
        n_envs=8,
        initial_population=DATA_PATH.joinpath(config.genome),
        genetic_map=DATA_PATH.joinpath(config.genetic_map),
        individual_per_gen=210,
    )
    eval_env = SelectionValues(eval_env)
    eval_env = AdaptVecEnv(eval_env)
    eval_env = VecMonitor(eval_env)
    eval_env = VecNormalize(eval_env, training=False, norm_obs=False)

    eval_callback = MyEvalCallback(eval_env, best_model_save_path=config.model_path,
                                log_path=config.log_path, eval_freq=1e3, n_eval_episodes=1e2,
                                deterministic=True, render=False, callback_after_eval=WandbCallback())
    
    policy_kwargs = dict(
        features_extractor_class=CNNFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=73),
        policy_hiddens=[16], 
        value_hiddens=[],
    )
    model = PPO(SelectionAC, train_envs, verbose=1, tensorboard_log=config.log_path, policy_kwargs=policy_kwargs, n_steps=10, batch_size=8)
    model.learn(total_timesteps=int(2e6), callback=eval_callback, log_interval=1)
    model.save(f"{config.model_path}/{config.unique_name}")