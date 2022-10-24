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
        return self.step_id / 5 - 1
        
    def reset(self, **kwargs):
        self.step_id = np.zeros(1)
        _, info = super().reset(**kwargs)
        
        return self._step_to_obs(), info
        
    def step(self, action):
        self.step_id += 1
        
        kbest_action = (action[0] + 1) / 2
        kbest_action = kbest_action * self.n + self.start
        kbest_action = round(kbest_action)
        _, r, ter, tru, info = super().step(kbest_action)
        
        if r != 0:
            r = (r - 0.11) * 100
        
        return self._step_to_obs(), r, ter, tru, info