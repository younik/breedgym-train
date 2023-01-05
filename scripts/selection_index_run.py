from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
from stable_baselines3.common.callbacks import EvalCallback
from breeding_gym.utils.paths import DATA_PATH
from wandb.integration.sb3 import WandbCallback
from breeding_gym.vector import VecBreedingGym, SelectionValues
from networks.block_mlp import SelectionBlockMLP
from networks.features_extractors import MaskedMarkerEffects, AppendMarkerEffects, NoFeaturesExtractor
from networks.selection_model import SelectionAC

from wrappers.print_action import PrintActionWrapper
from wrappers.vector.adapt_vec_env import AdaptVecEnv


def main(config):     
    train_envs = VecBreedingGym(
        n_envs=config.n_envs,
        initial_population=DATA_PATH.joinpath(config.genome),
        genetic_map=DATA_PATH.joinpath(config.genetic_map),
        individual_per_gen=210,
    )
    train_envs = SelectionValues(train_envs, k=21)
    train_envs = AdaptVecEnv(train_envs)
    train_envs = VecMonitor(train_envs)
    train_envs = VecNormalize(train_envs, norm_obs=False)
        
    eval_env = VecBreedingGym(
        n_envs=config.n_eval_envs,
        initial_population=DATA_PATH.joinpath(config.genome),
        genetic_map=DATA_PATH.joinpath(config.genetic_map),
        individual_per_gen=210,
    )
    eval_env = SelectionValues(eval_env, k=21)
    eval_env = AdaptVecEnv(eval_env)
    eval_env = VecMonitor(eval_env)
    eval_env = VecNormalize(eval_env, training=False, norm_obs=False)

    best_model_save_path = f"{config.model_path}/{config.unique_name}" if config.sweep_id is None else None
    callback_after_eval = WandbCallback() if not config.disable_wandb else None
    eval_callback = EvalCallback(eval_env, best_model_save_path=best_model_save_path,
                                log_path=config.log_path, eval_freq=5e3 // config.n_envs, n_eval_episodes=25,
                                deterministic=True, render=False, callback_after_eval=callback_after_eval)
    
    marker_effects = train_envs.simulator.GEBV_model.marker_effects
    policy_kwargs = dict(
        features_extractor_class=MaskedMarkerEffects,
        features_extractor_kwargs=dict(marker_effects=marker_effects.squeeze()),
        #block_per_chr=12, 
        #chr_lens=train_envs.simulator.chr_lens
    )

    model = PPO(SelectionAC, train_envs, verbose=1, tensorboard_log=config.log_path, policy_kwargs=policy_kwargs, n_steps=100, batch_size=16,
                gamma=config.gamma, learning_rate=config.learning_rate, gae_lambda=config.gae_lambda, n_epochs=int(config.n_epochs))
    
    model.learn(total_timesteps=int(config.total_timesteps), callback=eval_callback, log_interval=1)