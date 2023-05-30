from stable_baselines3.common.buffers import DictRolloutBuffer
from stable_baselines3.common.type_aliases import DictRolloutBufferSamples
import numpy as np
from gym import spaces


class FactoredDictRolloutBuffer(DictRolloutBuffer):

    def __init__(self, *args, individual_per_gen, **kwargs):
        self.individual_per_gen = individual_per_gen
        super().__init__(*args, **kwargs)

    def reset(self):
        super().reset()
        self.action_matrix = np.zeros((self.buffer_size, self.n_envs, self.individual_per_gen, self.individual_per_gen), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, self.n_envs, self.individual_per_gen), dtype=np.float32)
        self.values = np.zeros((self.buffer_size, self.n_envs, self.individual_per_gen), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size, self.n_envs, self.individual_per_gen), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size, self.n_envs, self.individual_per_gen), dtype=np.float32)

    def add(self, obs, action, action_matrix, reward, episode_start, value, log_prob):
        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, 1)

        for key in self.observations.keys():
            obs_ = np.array(obs[key]).copy()
            # Reshape needed when using multiple envs with discrete observations
            # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
            if isinstance(self.observation_space.spaces[key], spaces.Discrete):
                obs_ = obs_.reshape((self.n_envs,) + self.obs_shape[key])
            self.observations[key][self.pos] = obs_

        action = action.reshape((self.n_envs, self.action_dim))

        self.actions[self.pos] = np.array(action).copy()
        self.action_matrix[self.pos] = np.asarray(action_matrix).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.episode_starts[self.pos] = np.array(episode_start).copy()
        self.values[self.pos] = value.clone().cpu().numpy().squeeze()
        self.log_probs[self.pos] = log_prob.clone().cpu().numpy()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def compute_returns_and_advantage(self, last_values, dones: np.ndarray):
        # Convert to numpy
        last_values = last_values.clone().cpu().numpy().squeeze()
        last_values = last_values - last_values.mean(axis=-1, keepdims=True)
        norm_values = self.values - self.values.mean(axis=-1, keepdims=True)
        
        last_gae_lam = np.zeros_like(last_values)
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = norm_values[step + 1]
            
            factor_matrix = next_non_terminal[:, None, None] * self.action_matrix[step]
            delta = self.rewards[step] + self.gamma * np.mean(factor_matrix * next_values[:, :, None], axis=1) - norm_values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * np.mean(factor_matrix * last_gae_lam[:, :, None], axis=1)
            self.advantages[step] = last_gae_lam
        # TD(lambda) estimator, see Github PR #375 or "Telescoping in TD(lambda)"
        # in David Silver Lecture 4: https://www.youtube.com/watch?v=PnHCvfgC_ZA
        self.returns = self.advantages + self.values


    def _get_samples(self, batch_inds: np.ndarray, env=None):

        return DictRolloutBufferSamples(
            observations={key: self.to_torch(obs[batch_inds]) for (key, obs) in self.observations.items()},
            actions=self.to_torch(self.actions[batch_inds]),
            old_values=self.to_torch(self.values[batch_inds].flatten()),
            old_log_prob=self.to_torch(self.log_probs[batch_inds]),
            advantages=self.to_torch(self.advantages[batch_inds]),
            returns=self.to_torch(self.returns[batch_inds].flatten()),
        )