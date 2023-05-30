from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
from stable_baselines3.common.callbacks import CallbackList, EvalCallback
from breeding_gym.utils.paths import DATA_PATH
from wandb.integration.sb3 import WandbCallback
from breeding_gym.vector import VecBreedingGym, SelectionValues
from callbacks.adapt_gen_number import AdaptGenNumber
from networks.features_extractors import CNNFeaturesExtractor
from networks.selection_model import SelectionAC

from wrappers.vector.adapt_vec_env import AdaptVecEnv
from wrappers.vector.gen_number_obs import ObserveGenNumber
from wrappers.vector.separate_chr import SampleChr
from wrappers.vector.mask_snps import MaskSNPs
import numpy as np


def main(config):
    individual_per_gen = 200
    k_best = 20
    n_crosses = 10
    trials = 100

    env = VecBreedingGym(
        n_envs=config.n_envs,
        initial_population=DATA_PATH.joinpath(config.genome),
        genetic_map=DATA_PATH.joinpath(config.genetic_map),
        individual_per_gen=individual_per_gen,
        reward_shaping=bool(vars(config).get("reward_shaping", False)),
        num_generations=int(vars(config).get("num_generations", 10))
    )
    env = SelectionValues(env, k=k_best, n_crosses=n_crosses)
    # env = SampleChr(env, fix_chr=2)
    env = MaskSNPs(env)
    env = AdaptVecEnv(env)
    env = VecMonitor(env)
    #env = VecNormalize(env, training=False, norm_obs=False)

    buffer_gg = np.zeros((trials, config.n_envs))
    for trial_idx in range(trials):
        print(trial_idx, flush=True)
        obs = env.reset()

        done = False
        while not np.any(done):
            obs, rew, done, _ = env.step(obs.sum(axis=(-2, -1)).numpy())
            buffer_gg[trial_idx] += rew
        assert np.all(done)

    print(buffer_gg.std())
    print(buffer_gg.mean())
    env.close()