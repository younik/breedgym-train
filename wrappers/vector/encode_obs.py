from gym.vector import VectorEnvWrapper
from gym import spaces
from torch import nn
import torch


class ModuleWrapper(nn.Module):
    def __init__(self, observation_space, conv1_kwargs, conv2_kwargs, cnn_out_shape, features_dim) -> None:
        super().__init__()
        self.shared_network = nn.Sequential(
            nn.Conv1d(observation_space.shape[-1], **conv1_kwargs),
            nn.ReLU(),
            nn.Conv1d(conv1_kwargs["out_channels"], **conv2_kwargs),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(cnn_out_shape, features_dim)
        )
    
    def forward(self, x):
        return self.shared_network(x)


class EncodeObs(VectorEnvWrapper):
    
    def __init__(self, env, load_path, conv1_kwargs, conv2_kwargs):
        super().__init__(env)
        state_dict = torch.load(load_path)
        features_dim, cnn_out_shape = state_dict['5.weight'].shape
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.single_observation_space = spaces.Box(0, 10, shape=(self.individual_per_gen, features_dim * len(self.simulator.chr_lens),))
        self.observation_space = spaces.Box(0, 10, shape=(self.num_envs, self.individual_per_gen, features_dim * len(self.simulator.chr_lens)))

        shared_network = nn.Sequential(
            nn.Conv1d(self.env.observation_space.shape[-1], **conv1_kwargs),
            nn.ReLU(),
            nn.Conv1d(conv1_kwargs["out_channels"], **conv2_kwargs),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(cnn_out_shape, features_dim)
        ).to(self.device)
        shared_network.load_state_dict(state_dict)
        shared_network = shared_network.eval()
        
        self.shared_network = torch.jit.script(shared_network)    

    def _forward(self, x):
        x = torch.from_numpy(x).to(self.device)
        num_chr = len(self.simulator.chr_lens)
        x = x.reshape(self.num_envs, self.individual_per_gen, num_chr, -1, x.shape[-1])
        batch_pop = x.reshape(-1, x.shape[-2], x.shape[-1])
        batch_pop = batch_pop.permute(0, 2, 1)
        out1 = self.shared_network(batch_pop)
        chan_indices = torch.arange(x.shape[-1])
        chan_indices[0] = 1
        chan_indices[1] = 0
        out2 = self.shared_network(batch_pop[:, chan_indices])
        features = out1 + out2
        return features.reshape(*x.shape[:2], -1)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        with torch.no_grad():
            features = self._forward(obs)
        return features, info

    def step_wait(self, *args, **kwargs):
        obs, rew, ter, tru, infos = super().step_wait(*args, **kwargs)
        
        with torch.no_grad():
            features = self._forward(obs)
        return features, rew, ter, tru, infos