import gym
from gym import spaces
import numpy as np


class OneStepEpWrapper(gym.Wrapper):
    
    def __init__(self, env):
       super().__init__(env)
       
       self.observation_space = spaces.Discrete(n=1, start=0)
       self.action_space = spaces.MultiDiscrete([self.env.action_space.n] * 10)
       
    def reset(self, **kwargs):
        _, info = super().reset(**kwargs)
        
        return 0, info
    
    def step(self, action):
        action = action + self.env.action_space.start
        
        print(action)
        for a in action:
            _, r, ter, tru, info = super().step(a)
            
        r = (r - 0.12) * 100
        return 0, r, ter, tru, info
       
       