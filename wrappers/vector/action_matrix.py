from gym.vector import VectorEnvWrapper
import numpy as np

class FactorMatrixWrapper(VectorEnvWrapper):
    
    def __init__(self, env):
        super().__init__(env)
    
    def step_wait(self):
        obs, rews, ter, tru, infos = super().step_wait()
        action_matrix = np.zeros((self.num_envs, self.individual_per_gen, self.individual_per_gen), dtype=np.float32)
        arange_envs = np.arange(self.num_envs)
        arange_individual = np.arange(self.individual_per_gen)
        action_matrix[arange_envs[:, None], arange_individual, self.env._actions[:, :, 0]] = 1
        action_matrix[arange_envs[:, None], arange_individual, self.env._actions[:, :, 1]] = 1
        infos["action_matrix"] = action_matrix
        return obs, rews, ter, tru, infos