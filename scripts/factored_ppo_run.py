from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
from stable_baselines3.common.callbacks import CallbackList, EvalCallback
from breeding_gym.utils.paths import DATA_PATH
from wandb.integration.sb3 import WandbCallback
from breeding_gym.vector import VecBreedingGym, SelectionValues
from callbacks.adapt_gen_number import AdaptGenNumber
from factored_algorithm.factored_ppo import FactoredPPO
from factored_algorithm.factored_vec_monitor import FactoredVecMonitor
from factored_algorithm.multi_reward_vec_normalize import MultiRewardVecNormalize
from networks.features_extractors import CNNFeaturesExtractor, NoFeaturesExtractor
from factored_algorithm.factored_selection_model import FactoredSelectionAC
from wrappers.vector.action_matrix import FactorMatrixWrapper

from wrappers.vector.adapt_vec_env import AdaptVecEnv
from wrappers.vector.encode_obs import EncodeObs
from wrappers.vector.gen_number_obs import ObserveGenNumber
from wrappers.vector.histogramize import HistogramizeObs
from wrappers.vector.multi_rewards import MultiRewards
from wrappers.vector.separate_chr import SampleChr, SeparateChr
from wrappers.vector.mask_snps import MaskSNPs


def main(config):
    individual_per_gen = 200
    k_best = 20
    n_crosses = 10
    load_path="/cluster/home/oyounis/train/outs/SeparateChr/models/train_extractor_5e5_13105162"
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

    train_envs = VecBreedingGym(
        n_envs=config.n_envs,
        initial_population=DATA_PATH.joinpath(config.genome),
        genetic_map=DATA_PATH.joinpath(config.genetic_map),
        individual_per_gen=individual_per_gen,
        reward_shaping=bool(vars(config).get("reward_shaping", False)),
        num_generations=4 #int(vars(config).get("num_generations", 10))
    )
    train_envs = FactorMatrixWrapper(train_envs)
    train_envs = SelectionValues(train_envs, k=k_best, n_crosses=n_crosses)
    train_envs = SampleChr(train_envs, fix_chr=2)
    train_envs = MultiRewards(train_envs)
    train_envs = MaskSNPs(train_envs)
    train_envs = HistogramizeObs(train_envs, num_bins=int(config.num_bins))
    # train_envs = SeparateChr(train_envs)
    # train_envs = EncodeObs(train_envs, load_path, conv1_kwargs, conv2_kwargs)
    train_envs = ObserveGenNumber(train_envs)
    train_envs = AdaptVecEnv(train_envs)
    train_envs = FactoredVecMonitor(train_envs)
    train_envs = MultiRewardVecNormalize(train_envs, norm_obs=False, clip_reward=1e8)

    eval_env = VecBreedingGym(
        n_envs=config.n_eval_envs,
        initial_population=DATA_PATH.joinpath(config.genome),
        genetic_map=DATA_PATH.joinpath(config.genetic_map),
        individual_per_gen=individual_per_gen,
        reward_shaping=bool(vars(config).get("reward_shaping", False)),
        num_generations=int(vars(config).get("num_generations", 10))
    )
    eval_env = SelectionValues(eval_env, k=k_best, n_crosses=n_crosses)
    eval_env = SampleChr(eval_env, fix_chr=2)
    eval_env = MaskSNPs(eval_env)
    eval_env = HistogramizeObs(eval_env, num_bins=int(config.num_bins))
    # eval_env = SeparateChr(eval_env)
    eval_env = ObserveGenNumber(eval_env)
    eval_env = AdaptVecEnv(eval_env)
    eval_env = VecMonitor(eval_env)
    eval_env = VecNormalize(eval_env, training=False, norm_obs=False)

    best_model_save_path = None # f"{config.model_path}/{config.unique_name}" if config.sweep_id is None else None
    callbacks = [
        AdaptGenNumber(
            [4e3, 2e4, 1e5, 5e5, 1e6, 3e6],  #, 5e6],
            [5, 6, 7, 8, 9, 10]
        )
    ]
    if not config.disable_wandb:
        callbacks.append(WandbCallback())
    
    callbacks_after_eval = CallbackList(callbacks)
    eval_callback = EvalCallback(eval_env, best_model_save_path=best_model_save_path,
                                log_path=config.log_path, eval_freq=2e4 // config.n_envs, n_eval_episodes=25,
                                deterministic=True, render=False, callback_after_eval=callbacks_after_eval)

    policy_kwargs = dict(
        features_extractor_class=CNNFeaturesExtractor,
        features_extractor_kwargs=dict(
            conv1_kwargs=conv1_kwargs,
            conv2_kwargs=conv2_kwargs,
            features_dim=64
        ),
        policy_hiddens=32, # TODO: 16 e sopra 32
        value_hiddens=64,
        gen_features_dim=int(vars(config).get("gen_features_dim", 1))
    )

    model = FactoredPPO(FactoredSelectionAC, train_envs, verbose=1, tensorboard_log=config.log_path, policy_kwargs=policy_kwargs, n_steps=512, batch_size=64,
                gamma=config.gamma, learning_rate=config.learning_rate, gae_lambda=config.gae_lambda, n_epochs=int(config.n_epochs))#, ent_coef=config.ent_coeff)

    model.learn(total_timesteps=int(config.total_timesteps), callback=eval_callback)

    # torch.save(model.policy.features_extractor.shared_network.state_dict(), f"{config.model_path}/{config.unique_name}")
