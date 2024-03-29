from stable_baselines3.common.vec_env.base_vec_env import VecEnv
import numpy as np
import jax.numpy as jnp
import cupy
import torch


class AdaptVecEnv(VecEnv):
    
    def __init__(self, gym_vec_env):
        self.gym_vec_env = gym_vec_env
        
        super().__init__(
            self.gym_vec_env.num_envs,
            self.gym_vec_env.single_observation_space,
            self.gym_vec_env.single_action_space
        )
        
    def reset(self):
        obs, _ = self.gym_vec_env.reset()
        return self._adapt_obs(obs)

    def step_async(self, actions):
        return self.gym_vec_env.step_async(actions)

    def step_wait(self):
        obs, rew, ter, tru, infos = self.gym_vec_env.step_wait()
        adapt_infos = [{k: v[i] for k, v in infos.items()} 
                      for i in range(self.num_envs)]

        new_obs = self._adapt_obs(obs)
        return new_obs, rew, jnp.logical_or(ter, tru), adapt_infos

    def _adapt_obs(self, obs):
        if isinstance(obs, dict):
            return {key: self._adapt_obs(value) for key, value in obs.items()}
        
        if torch.cuda.is_available():
            obs = torch.as_tensor(cupy.asarray(obs))
        else:
            obs = np.asarray(obs)
        
        return obs

    def seed(self, seed=None):
        return self.gym_vec_env.seed(seed=seed)

    def close(self):
        return self.gym_vec_env.close()

    def __getattr__(self, name):
        return getattr(self.gym_vec_env, name)

    def get_attr(self, attr_name: str, indices=None):
        len_indices = len(self._get_indices(indices))
        return [getattr(self.gym_vec_env, attr_name)] * len_indices

    def set_attr(self, attr_name: str, value, indices=None):
        self.gym_vec_env.set_attr(attr_name, value)

    def env_method(self, method_name: str, *args, indices=None, **kwargs):
        return getattr(self.gym_vec_env, method_name)(*args, **kwargs)

    def env_is_wrapped(self, wrapper_class, indices=None):
        len_indices = len(self._get_indices(indices))
        return [isinstance(self.gym_vec_env, wrapper_class)] * len_indices

    def _get_indices(self, indices):
        if indices is None:
            return np.arange(self.n_envs)
        elif isinstance(indices, int):
            return [indices]
        else:
            return indices