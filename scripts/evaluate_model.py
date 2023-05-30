from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
from breeding_gym.utils.paths import DATA_PATH
from breeding_gym.vector import VecBreedingGym, SelectionValues
from wrappers.vector.adapt_vec_env import AdaptVecEnv
from wrappers.vector.gen_number_obs import ObserveGenNumber
import numpy as np

from wrappers.vector.mask_snps import MaskSNPs


def main(config):
    individual_per_gen = 200
    k_best = 20
    n_crosses = 10
    trials = 100
    num_generations= 10

    envs = VecBreedingGym(
        n_envs=config.n_eval_envs,
        initial_population=DATA_PATH.joinpath(config.genome),
        genetic_map=DATA_PATH.joinpath(config.genetic_map),
        individual_per_gen=individual_per_gen,
        reward_shaping=bool(vars(config).get("reward_shaping", False)),
        num_generations=num_generations,
        autoreset=False
    )
    envs = SelectionValues(envs, k=k_best, n_crosses=n_crosses)
    envs = MaskSNPs(envs)
    envs = ObserveGenNumber(envs)
    envs = AdaptVecEnv(envs)
    envs = VecMonitor(envs)
    envs = VecNormalize(envs, training=False, norm_obs=False)

    # model = PPO.load('/cluster/home/oyounis/train/outs/TrainICMLMax/models/train_max_16755467/best_model')
    model = PPO.load('/cluster/home/oyounis/train/outs/TrainICMLMax/models/train_max_16902718/best_model')
    # model = PPO.load('/cluster/home/oyounis/train/outs/TrainICML/models/final_training_16729805/best_model')

    buffer_gg = np.zeros((trials, config.n_eval_envs, num_generations))
    for trial_idx in range(trials):
        print(trial_idx, flush=True)
        obs = envs.reset()
        gebvs = obs['obs'].sum(axis=(2, 3))

        done = False
        for gen in range(num_generations):
            action = model.predict(obs, deterministic=True)[0]
            obs, _, done, _ = envs.step(action)
            gebvs = obs['obs'].sum(axis=(2, 3))
            buffer_gg[trial_idx, :, gen] = gebvs.max(axis=-1)
        assert np.all(done)
        
    print(np.array2string(buffer_gg.std(axis=(0, 1)), separator=", "))
    print(np.array2string(buffer_gg.mean(axis=(0, 1)), separator=", "))
    envs.close()
    