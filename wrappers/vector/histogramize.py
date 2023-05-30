from gym.vector import VectorEnvWrapper
import jax
import jax.numpy as jnp
import numpy as np
from gym import spaces
from functools import partial


class HistogramizeObs(VectorEnvWrapper):
    
    def __init__(self, env, num_bins=None):
        super().__init__(env)
        if num_bins is None:
            num_bins = self.observation_space.shape[-2]
        self.num_bins = num_bins

        obs_space = self.single_observation_space
        shape = obs_space.shape[:-2] + (len(self.simulator.chr_lens) * self.num_bins, obs_space.shape[-1])
        ratio = 5 * obs_space.shape[-2] // self.num_bins  # euristic
        ratio = max(1, ratio)
        self.single_observation_space = spaces.Box(
            obs_space.low.item(0) * ratio,
            obs_space.high.item(0) * ratio,
            shape=shape,
            dtype=obs_space.dtype
        )
        self.observation_space = spaces.Box(
            obs_space.low.item(0) * ratio,
            obs_space.high.item(0) * ratio,
            shape=(self.num_envs,) + shape,
            dtype=obs_space.dtype
        )

    def reset(self):
        obs, info = self.env.reset()
        return self._transform_obs(obs), info

    def step_wait(self):
        obs, rew, ter, tru, infos = super().step_wait()
        return self._transform_obs(obs), rew, ter, tru, infos

    def _transform_obs(self, obs):
        vhist = jax.vmap(jnp.histogram, in_axes=(None, None, None, 1), out_axes=1)
        vhist = jax.vmap(vhist, in_axes=(None, None, None, 0))
        vhist = jax.vmap(vhist, in_axes=(None, None, None, 0))
        
        # TODO can be slow, putting in cpu, will go to gpu again?
        out = np.zeros((*obs.shape[:2], len(self.simulator.chr_lens) * self.num_bins, obs.shape[-1]), dtype=np.float32)
        
        start_chr, start_out = 0, 0
        for chr_len in self.simulator.chr_lens:
            end_chr, end_out = start_chr + chr_len, start_out + self.num_bins
            chr_ = obs[:, :, start_chr:end_chr]
            hist_chr, _ = vhist(self.simulator.cM[start_chr:end_chr], self.num_bins, (0, 150), chr_)
            out[:, :, start_out:end_out] = hist_chr
            start_chr, start_out = end_chr, end_out
            
        return out
