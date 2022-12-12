import random
import gym
import numpy as np
from breeding_gym.utils.paths import DATA_PATH
from breeding_gym.wrappers import ObserveStepWrapper


def choose_action(values, eps):
    if eps < random.random():
        return np.argmax(values)
    else:
        return np.random.randint(19)


alpha = 0.2
epsilon = 0.1
total_steps = 200_000

env = gym.make("breeding_gym:KBestBreedingGym",
                     initial_population=DATA_PATH.joinpath("small_geno.txt"),
                     genetic_map=DATA_PATH.joinpath("small_genetic_map.txt"),
                     individual_per_gen=200,
                     )
env = ObserveStepWrapper(env)

values_table = np.zeros((11, 19), dtype=np.float32)

for ts in range(total_steps):
    prev_obs, _ = env.reset()
    ter, tru = False, False
    a = choose_action(values_table[prev_obs], epsilon)
    while not ter and not tru:    
        obs, r, ter, tru, _ = env.step(a + 2)
        next_a = choose_action(values_table[obs], epsilon)

        td_error = r - (values_table[prev_obs] - values_table[obs])
        values_table[prev_obs] += alpha * td_error
        
        a, prev_obs = next_a, obs
    
    if ts % 1000 == 0:
        print(ts // 1000, np.argmax(values_table[:10], axis=1) + 2, flush=True)