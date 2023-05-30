from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
from stable_baselines3.common.callbacks import CallbackList, EvalCallback
from breeding_gym.utils.paths import DATA_PATH
from wandb.integration.sb3 import WandbCallback
from breeding_gym.vector import VecBreedingGym, WheatBreedingGym
from callbacks.adapt_gen_number import AdaptGenNumber
from networks.features_extractors import CNNFeaturesExtractor, NoFeaturesExtractor
from networks.selection_model import SelectionAC
from networks.pair_score_model import PairScoreAC

from wrappers.vector.adapt_vec_env import AdaptVecEnv
from wrappers.vector.encode_obs import EncodeObs
from wrappers.vector.gen_number_obs import ObserveGenNumber
from wrappers.vector.histogramize import HistogramizeObs
from wrappers.vector.mask_snps import MaskSNPs
from wrappers.vector.separate_chr import SampleChr


def main(config):    
    individual_per_gen = 200
    conv1_kwargs={
        "out_channels": int(vars(config).get("out_channels1", 64)),
        "kernel_size": int(vars(config).get("kernel_size1", 256)),
        "stride": int(vars(config).get("stride1", 32))
    }
    conv2_kwargs={
        "out_channels": int(vars(config).get("out_channels2", 16)),
        "kernel_size": int(vars(config).get("kernel_size2", 8)),
        "stride": int(vars(config).get("stride2", 2))
    }
    load_path="/cluster/home/oyounis/train/outs/SupervisedTraining/models/supervised_1k_13067696"

    train_envs = VecBreedingGym(
        n_envs=config.n_envs,
        initial_population=DATA_PATH.joinpath(config.genome),
        genetic_map=DATA_PATH.joinpath(config.genetic_map),
        individual_per_gen=individual_per_gen,
        trait_names=["Yield"],
        reward_shaping=True,
        num_generations=int(vars(config).get("num_generations", 10))
    )
    train_envs = WheatBreedingGym(train_envs)
    train_envs = MaskSNPs(train_envs)
    train_envs = ObserveGenNumber(train_envs)
    train_envs = AdaptVecEnv(train_envs)
    train_envs = VecMonitor(train_envs)
    train_envs = VecNormalize(train_envs, norm_obs=False)

    eval_env = VecBreedingGym(
        n_envs=config.n_eval_envs,
        initial_population=DATA_PATH.joinpath(config.genome),
        genetic_map=DATA_PATH.joinpath(config.genetic_map),
        individual_per_gen=individual_per_gen,
        trait_names=["Yield"],
        reward_shaping=True,
        num_generations=int(vars(config).get("num_generations", 10))
    )
    eval_env = WheatBreedingGym(eval_env)
    eval_env = MaskSNPs(eval_env)
    eval_env = ObserveGenNumber(eval_env)
    eval_env = AdaptVecEnv(eval_env)
    eval_env = VecMonitor(eval_env)
    eval_env = VecNormalize(eval_env, training=False, norm_obs=False)

    best_model_save_path = f"{config.model_path}/{config.unique_name}" if config.sweep_id is None else None
    callbacks = []
    if not config.disable_wandb:
        callbacks.append(WandbCallback())
    
    callbacks_after_eval = CallbackList(callbacks)
    eval_callback = EvalCallback(eval_env, best_model_save_path=best_model_save_path,
                                log_path=config.log_path, eval_freq=1e4 // config.n_envs, n_eval_episodes=25,
                                deterministic=True, render=False, callback_after_eval=callbacks_after_eval)

    policy_kwargs = dict(
        features_extractor_class=CNNFeaturesExtractor,
        features_extractor_kwargs=dict(
            conv1_kwargs=conv1_kwargs,
            conv2_kwargs=conv2_kwargs,
            # features_dim=64,
        ),
        actor_hiddens=64,
        value_hiddens=32,
    )

    model = PPO(PairScoreAC, train_envs, verbose=1, tensorboard_log=config.log_path, policy_kwargs=policy_kwargs, n_steps=2048, batch_size=64,
                gamma=config.gamma, learning_rate=config.learning_rate, gae_lambda=config.gae_lambda, n_epochs=int(config.n_epochs), ent_coef=vars(config).get("ent_coeff", 0))

    model.learn(total_timesteps=int(config.total_timesteps), callback=eval_callback)