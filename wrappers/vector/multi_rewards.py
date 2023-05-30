from gym.vector import VectorEnvWrapper
import numpy as np
from wrappers.vector.mask_snps import MaskSNPs


class MultiRewards(VectorEnvWrapper):
    
    def __init__(self, env):
        super().__init__(env)
        
    def step_wait(self):
        obs, _, ter, tru, infos = super().step_wait()
        assert obs.shape[-1] == 2
        rew = self.simulator.GEBV_model(self.populations).squeeze()
        return obs, np.asarray(rew), ter, tru, infos