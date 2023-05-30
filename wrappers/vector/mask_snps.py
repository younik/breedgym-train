from gym.vector import VectorEnvWrapper
from gym import spaces
import numpy as np

class MaskSNPs(VectorEnvWrapper):
    
    def __init__(self, env):
        super().__init__(env)
        
        mrks = self.simulator.GEBV_model.marker_effects
        self.mrk_std = mrks.std()
        
        low = min(0, mrks.min() / self.mrk_std)
        high = max(0, mrks.max() / self.mrk_std)
        self.single_observation_space = spaces.Box(
            low=low.item(),
            high=high.item(),
            shape=(*self.single_observation_space.shape[:-1], self.single_observation_space.shape[-1] + 0),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=low.item(),
            high=high.item(),
            shape=(*self.observation_space.shape[:-1], self.observation_space.shape[-1] + 0),
            dtype=np.float32
        )

    def reset(self):
        obs, info = self.env.reset()
        return self._transform_obs(obs), info

    def step_wait(self):
        obs, rew, ter, tru, infos = super().step_wait()
        return self._transform_obs(obs), rew, ter, tru, infos

    def _transform_obs(self, obs):
        masked_obs = obs * self.simulator.GEBV_model.marker_effects[None, None, :]

        rec_vec = np.broadcast_to(
            self.simulator.recombination_vec[None, None, :],
            obs.shape[:-1]
        )

        return np.concatenate(
            (
                masked_obs,
                # masked_obs.max(axis=-1, keepdims=True),
                # masked_obs.min(axis=-1, keepdims=True),
                # rec_vec[..., None]
            ),
            axis=-1
        )
        