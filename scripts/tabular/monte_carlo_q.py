import random
import gym
from breeding_gym.utils.paths import DATA_PATH
from breeding_gym.wrappers import ObserveStepWrapper
import numpy as np


def choose_action(values, eps):
    if eps < random.random():
        return np.argmax(values) + 2
    else:
        return np.random.randint(2, 21)

env = gym.make("breeding_gym:KBestBreedingGym",
                     initial_population=DATA_PATH.joinpath("small_geno.txt"),
                     genetic_map=DATA_PATH.joinpath("small_genetic_map.txt"),
                     individual_per_gen=200,
                     )
env = ObserveStepWrapper(env)

values_table = np.zeros((10, 19), dtype=np.float32)
n_visits = np.zeros((10, 19), dtype=np.int32)

epsilon = 0.1

total_steps = 100_000

print(f"RUN with eps={epsilon}", flush=True)

for ts in range(total_steps):
    ep_choices = np.zeros(10, dtype=np.int8)
    arange_10 = np.arange(10)
    
    env.reset()
    for i in arange_10:
        a = choose_action(values_table[i] / n_visits[i], epsilon)
        ep_choices[i] = a            
        _, r, _, _, _ = env.step(a)
        # assumes no-reward shaping
        
    
    n_visits[arange_10, ep_choices - 2] += 1
    values_table[arange_10, ep_choices - 2] += r
        
    if ts % 1000 == 0:
        print(ts, np.argmax(values_table, axis=1) + 2, flush=True)
    # epsilon *= lambda_
    
    