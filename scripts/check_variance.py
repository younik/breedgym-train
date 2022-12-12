from os import truncate
import gym
import breeding_gym  # noqa: F401
from breeding_gym.wrappers import ObserveStepWrapper
from breeding_gym.utils.paths import DATA_PATH
import numpy as np

env = gym.make("KBestBreedingGym",
                     initial_population=DATA_PATH.joinpath("small_geno.txt"),
                     genetic_map=DATA_PATH.joinpath("small_genetic_map.txt"),
                     individual_per_gen=200,
                     )

trial = 100
action = range(2, 21)

table = np.empty((len(action), trial))

for a in action:
    for t in range(trial):
        env.reset()
        truncated = False
        while not truncated:
            _, r, _, truncated, _ = env.step(a)
            
        table[a-2, t] = r
            


print(np.mean(table, axis=1))
print(np.std(table, axis=1))