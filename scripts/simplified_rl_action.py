import gym
from gym.wrappers.record_episode_statistics import RecordEpisodeStatistics
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, ProgressBarCallback, CallbackList
from breeding_gym.utils.paths import DATA_PATH
from breeding_gym.simulator import BreedingSimulator
from wandb.integration.sb3 import WandbCallback
import jax
from networks.features_extractors import GEBVCorrcoefExtractor, NoFeaturesExtractor
from networks.gebv_corr_attention import GEBVCorrAC
from wrappers.adapt_discrete_space import AdaptDiscreteAction
from wrappers.continuous_wrapper import ContinuousWrapper
from wrappers.one_step_ep import OneStepEpWrapper
from wrappers.print_action import PrintActionWrapper
from wrappers.move_device import DeviceWrapper
import os


def main(config):
    simulator = BreedingSimulator(
        genetic_map=DATA_PATH.joinpath(config.genetic_map),
        # backend="cpu",
    )
    germplasm = simulator.load_population(DATA_PATH.joinpath(config.genome))
    del simulator
    
    def make_env_f(print_action=False, **kwargs):
        def make_env():
            env = gym.make(config.env_name,
                                initial_population=germplasm,
                                genetic_map=DATA_PATH.joinpath(config.genetic_map),
                                individual_per_gen=200,
                                # backend="cpu",
                                **kwargs
                                )
            
            if print_action:
                env = PrintActionWrapper(env)
            env = AdaptDiscreteAction(env)
            return env
        return make_env
    
    class_ = SubprocVecEnv if config.n_envs > 1 else DummyVecEnv
    train_envs = class_([make_env_f() for i in range(config.n_envs)])
    train_envs = VecNormalize(train_envs, gamma=1)
    train_envs = VecMonitor(train_envs)

    class_ = SubprocVecEnv if config.n_eval_envs > 1 else DummyVecEnv
    eval_env = class_([make_env_f(print_action=True) for i in range(config.n_eval_envs)])
    eval_env = VecMonitor(eval_env)
    eval_env = VecNormalize(eval_env, training=False, gamma=1, norm_reward=False)

    # policy_kwargs = dict(
    #     features_extractor_class=GEBVCorrcoefExtractor,
    #     features_extractor_kwargs=dict(features_dim=64),
    #     net_arch=[dict(vf=[], pi=[])]
    # )
    
    
    eval_callback = EvalCallback(eval_env, best_model_save_path=config.model_path,
                                log_path=config.log_path, eval_freq=1e5 // config.n_envs, n_eval_episodes=128,
                                deterministic=True, render=False,
                                callback_after_eval=WandbCallback() if not config.disable_wandb else None)

    callback_list = CallbackList([eval_callback, ProgressBarCallback()])

    policy_kwargs = dict(
        features_extractor_class=NoFeaturesExtractor,
        key_dim=8
    )
    model = PPO(GEBVCorrAC, train_envs, verbose=1, tensorboard_log=config.log_path, policy_kwargs=policy_kwargs,
                gamma=config.gamma, learning_rate=config.learning_rate, gae_lambda=config.gae_lambda, 
                batch_size=int(config.batch_size), n_epochs=int(config.n_epochs), n_steps=int(config.buffer_size // config.n_envs))
    
    model.learn(total_timesteps=int(1e6), callback=callback_list, log_interval=1)
    model.save(f"{config.model_path}/{config.unique_name}")