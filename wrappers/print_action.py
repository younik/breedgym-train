import gym


class PrintActionWrapper(gym.Wrapper):
    
    def reset(self, **kwargs):
        print("", flush=True)
        return super().reset(**kwargs)
    
    
    def step(self, action):
        print(action, end=", ")
        return super().step(action)