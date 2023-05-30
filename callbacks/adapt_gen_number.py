from stable_baselines3.common.callbacks import BaseCallback


class AdaptGenNumber(BaseCallback):

    def __init__(self, timesteps, num_gens, verbose: int = 0):
        super().__init__(verbose)
        assert len(timesteps) == len(num_gens)
        self.pointer = 0
        self.timesteps = timesteps
        self.num_gens = num_gens
        
    def _on_step(self) -> bool:
        if self.pointer < len(self.timesteps) and self.num_timesteps > self.timesteps[self.pointer]:
            self.model.env.set_attr("num_generations", self.num_gens[self.pointer])
            self.pointer += 1

        return True