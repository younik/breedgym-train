from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
from torch import nn
import numpy as np

class NoFeaturesExtractor(BaseFeaturesExtractor):

    def __init__(self, observation_space):
        super().__init__(observation_space, features_dim=observation_space['obs'].shape[-1])
        self.output_shape = observation_space['obs'].shape
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return observations


class GEBVCorrcoefExtractor(BaseFeaturesExtractor):

    def __init__(self, observation_space, features_dim=32):
        super().__init__(observation_space, features_dim=features_dim)
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        len_dict = len(observation_space)
        assert features_dim % len_dict == 0
        out_features = features_dim // len_dict
        
        self.networks = [nn.Linear(obs.shape[0], out_features).to(device)
                         for obs in observation_space.values()]
        
        self.non_linearity = nn.ReLU()
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        cat_out = torch.cat([
            net(obs) for net, obs in zip(self.networks, observations.values())
        ], dim=1)
        
        return self.non_linearity(cat_out)


CONV1_DEFAULT = {"out_channels": 64, "kernel_size": 256, "stride": 32}
CONV2_DEFAULT = {"out_channels": 16, "kernel_size": 8, "stride": 2}

class CNNFeaturesExtractor(BaseFeaturesExtractor):
    
    def __init__(
        self,
        observation_space,
        conv1_kwargs=CONV1_DEFAULT,
        conv2_kwargs=CONV2_DEFAULT,
        load_path=None,
        features_dim=None
    ):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        shared_network = nn.Sequential(
            nn.Conv1d(observation_space['obs'].shape[-1], **conv1_kwargs),
            # nn.Conv1d(observation_space.shape[-1], **conv1_kwargs),
            nn.ReLU(),
            nn.Conv1d(conv1_kwargs["out_channels"], **conv2_kwargs),
            nn.ReLU(),
            nn.Flatten()
        ).to(self.device)
        
        sample = torch.zeros(observation_space['obs'].shape, device=self.device)
        # sample = torch.zeros(observation_space.shape, device=self.device)
        with torch.no_grad():
            out_sample = CNNFeaturesExtractor._forward(shared_network, sample)
        
        if features_dim is None:
            features_dim = out_sample.shape[-1]
        else:
            shared_network.append(nn.Linear(out_sample.shape[-1], features_dim))
            shared_network.append(nn.ReLU())
        
        if load_path is not None:
            state_dict = torch.load(load_path)
            shared_network.load_state_dict(state_dict)
        
        super().__init__(observation_space, features_dim=features_dim)
        self.shared_network = torch.jit.script(shared_network)    

    @staticmethod
    def _forward(net, x):
        batch_pop = x.reshape(-1, x.shape[-2], x.shape[-1])
        batch_pop = batch_pop.permute(0, 2, 1)
        out1 = net(batch_pop)
        chan_indices = torch.arange(x.shape[-1])
        chan_indices[0] = 1
        chan_indices[1] = 0
        out2 = net(batch_pop[:, chan_indices])
        features = out1 + out2
        return features.reshape(*x.shape[:-2], -1)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        features = CNNFeaturesExtractor._forward(self.shared_network, observations['obs'])
        return {'obs': features, 'gen_number': observations['gen_number']}
        # return CNNFeaturesExtractor._forward(self.shared_network, observations)
