from gym.vector import VectorEnvWrapper
from gym import spaces
import numpy as np

class SampleChr(VectorEnvWrapper):
    
    def __init__(self, env, fix_chr=None):
        super().__init__(env)
        self.fix_chr = fix_chr

        self.chr_len = self.simulator.chr_lens[0]
        # assert np.all(self.chr_len == self.simulator.chr_lens)
        self.n_chr = len(self.simulator.chr_lens)
        self.simulator.chr_lens = [self.chr_len]

        obs_space = self.observation_space
        self.single_observation_space = spaces.Box(
            low=obs_space.low.item(0),
            high=obs_space.high.item(0),
            shape=(self.individual_per_gen, self.chr_len, 2),
            dtype=obs_space.dtype
        )
        self.observation_space = spaces.Box(
            low=obs_space.low.item(0),
            high=obs_space.high.item(0),
            shape=(self.n_envs, self.individual_per_gen, self.chr_len, 2),
            dtype=obs_space.dtype
        )
        
        # germ_shape = self.germplasm.shape[0], self.n_chr, self.chr_len, 2
        # self.germplasm.shape = germ_shape
        # self.set_attr("germplasm", chr_germ)
        self.sampled_chr = None
        self.rec_vec = self.simulator.recombination_vec
        self.full_cM = self.simulator.cM
        self.mrk_effects = self.simulator.GEBV_model.marker_effects
        self.full_germplasm = self.germplasm

    def reset(self, seed=None):
        if self.fix_chr is None:
            generator = np.random.default_rng(seed=seed)
            chr_ = generator.integers(self.n_chr)
        else:
            chr_ = self.fix_chr 
        slice_ = np.s_[chr_*self.chr_len : (chr_+1)*self.chr_len]
        # TODO, make a property simulator returning a Namespace
        self.simulator.recombination_vec = self.rec_vec[slice_]
        self.simulator.GEBV_model.marker_effects = self.mrk_effects[slice_]
        self.simulator.cM = self.full_cM[slice_]
        self.set_attr("germplasm", self.full_germplasm[..., slice_, :])
        return super().reset()

    
    def step_wait(self):
        obs, rew, ter, tru, infos = super().step_wait()
        dones = np.logical_or(ter, tru)
        if np.any(dones):
            assert np.all(dones)
            obs, _ = self.reset()

        return obs, rew, ter, tru, infos


class SeparateChr(VectorEnvWrapper):
    
    def __init__(self, env):
        super().__init__(env)

        # assert np.all(self.simulator.chr_lens[0] == self.simulator.chr_lens)
        self.n_chr = len(self.simulator.chr_lens)

        obs_space = self.observation_space
        assert obs_space.shape[-2] % self.n_chr == 0
        self.chr_len = obs_space.shape[-2] // self.n_chr  # maybe preprocessed

        self.num_envs = self.n_envs * self.n_chr
        self.obs_shape = (
            self.n_envs * self.n_chr,
            self.individual_per_gen,
            self.chr_len,
            obs_space.shape[-1]
        )
        self.single_observation_space = spaces.Box(
            low=obs_space.low.item(0),
            high=obs_space.high.item(0),
            shape=self.obs_shape[1:],
            dtype=obs_space.dtype
        )
        self.observation_space = spaces.Box(
            low=obs_space.low.item(0),
            high=obs_space.high.item(0),
            shape=self.obs_shape,
            dtype=obs_space.dtype
        )

    def reset(self, **kwargs):
        obs, infos = self.env.reset(**kwargs)
        return self._transform_obs(obs), infos

    def step_async(self, actions):
        actions = actions.reshape(self.n_envs, self.n_chr, self.individual_per_gen)
        actions = actions.mean(axis=1)
        return super().step_async(actions)

    def step_wait(self):
        obs, rew, ter, tru, infos = self.env.step_wait()
        return (
            self._transform_obs(obs),
            np.repeat(rew, self.n_chr),
            np.repeat(ter, self.n_chr),
            np.repeat(tru, self.n_chr),
            infos
        )

    def _transform_obs(self, obs):
        obs = obs.reshape(*obs.shape[:2], self.n_chr, self.chr_len, obs.shape[-1])
        obs = obs.transpose(0, 2, 1, 3, 4)
        return obs.reshape(-1, *obs.shape[2:])
        