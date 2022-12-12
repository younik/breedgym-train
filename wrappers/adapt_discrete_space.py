import gym
from gym import spaces    


class AdaptDiscreteAction(gym.Wrapper):
    
    def __init__(self, env):
        assert isinstance(env.action_space, spaces.Discrete)
        super().__init__(env)
        
        self.offset = env.action_space.start
        if self.offset != 0:
            self.action_space = spaces.Discrete(env.action_space.n)
            
            
    def step(self, action):
        return self.env.step(action + self.offset)

