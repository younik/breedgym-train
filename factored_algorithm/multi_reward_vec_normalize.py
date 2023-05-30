from stable_baselines3.common.vec_env import VecNormalize
import numpy as np


class MultiRewardVecNormalize(VecNormalize):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.returns = np.zeros((self.num_envs, self.individual_per_gen))
    
    def reset(self):
        out = super().reset()
        self.returns = np.zeros((self.num_envs, self.individual_per_gen))
        return out
    
    def set_venv(self, venv) -> None:
        super().set_venv(venv)
        self.returns = np.zeros((self.num_envs, self.individual_per_gen))
    
    def _update_reward(self, reward: np.ndarray) -> None:
        """Update reward normalization statistics."""
        self.returns = self.returns * self.gamma + reward
        self.ret_rms.update(self.returns.mean(axis=-1))