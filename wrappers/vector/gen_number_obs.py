from gym.vector import VectorEnvWrapper
from gym import spaces
import numpy as np


class ObserveGenNumber(VectorEnvWrapper):

    def __init__(self, env):
        super().__init__(env)
        self.num_obs = self.env.observation_space.shape[0]
        self.observation_space = spaces.Dict({
            "obs": self.observation_space,
            "gen_number": spaces.Box(
                -np.ones((self.num_obs, 1)),
                np.ones((self.num_obs, 1))
            )
        })

        self.single_observation_space = spaces.Dict({
            "obs": self.single_observation_space,
            "gen_number": spaces.Box(-1, 1)
        })
        self.gen_number = None

    def reset(self, **kwargs):
        self.gen_number = -np.ones((self.num_obs, 1))
        obs, info = self.env.reset(**kwargs)
        return {"obs": obs, "gen_number": self.gen_number}, info

    def step_wait(self, *args, **kwargs):
        obs, rew, ter, tru, infos = super().step_wait(*args, **kwargs)
        if np.any(np.logical_or(ter, tru)):
            assert np.all(np.logical_or(ter, tru))
            self.gen_number = -np.ones((self.num_obs, 1))
        else:
            self.gen_number += 2 / (self.env.num_generations - 1)
        new_obs = {"obs": obs, "gen_number": self.gen_number}
        return new_obs, rew, ter, tru, infos
