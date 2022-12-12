import gym
from gym import spaces
import numpy as np

class ContinuousWrapper(gym.Wrapper):
    
    def __init__(self, env):
        super().__init__(env)
        
        self.action_space = spaces.Box(-1, 1, shape=(1,))
        self.observation_space = spaces.Box(-1, 1, shape=(1,))
        self.step_id = None
        
        self.n = self.env.action_space.n - 1
        self.start = self.env.action_space.start
    
    def _step_to_obs(self):
        obs = self.step_id / 5 - 1
        return np.full((self.env.n_envs, 1), obs)
        
    def reset(self):
        self.step_id = np.zeros(1)
        self.env.reset()
        return self._step_to_obs()

    def step_async(self, action):
        self.step_id += 1
        
        kbest_action = (action + 1) / 2
        kbest_action = kbest_action * self.n + self.start
        kbest_action = np.round(kbest_action)
        return self.env.step_async(kbest_action)
        
    def step_wait(self):
        _, r, dones, info = self.env.step_wait()
        return self._step_to_obs(), r, dones, info