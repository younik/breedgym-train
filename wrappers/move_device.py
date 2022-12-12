import gym
import jax
import numpy as np
import torch


class DeviceWrapper(gym.Wrapper):
    def __init__(self, env, device):
        super().__init__(env)
        self.device = device
    
    def reset(self, *args, **kwargs):
        obs = super().reset(*args, **kwargs)
        return self._move_obs(obs)
    
    def step(self, action):
        obs, rew, ter, tru, info = super().step(action)
        return self._move_obs(obs), rew, ter, tru, info
    
    def _move_obs(self, obs):
        if isinstance(obs, dict):
            return {
                k: self._move_obs(obs[k])
                for k in obs.keys()
            }
        elif isinstance(obs, list):
            return [self._move_obs(o) for o in obs]
        elif isinstance(obs, tuple):
            return (self._move_obs(o) for o in obs)
        elif isinstance(obs, (jax.numpy.ndarray, np.ndarray)):
            # see https://github.com/pytorch/pytorch/issues/32868
            return torch.as_tensor(obs, device="cuda")
        else:
            raise TypeError("Observation type is", type(obs))