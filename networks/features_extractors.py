from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
from torch import nn

class NoFeaturesExtractor(BaseFeaturesExtractor):

    def __init__(self, observation_space, features_dim=10_000):
        super(NoFeaturesExtractor, self).__init__(observation_space, features_dim=features_dim)
        
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
    
class CNNFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=10_000, out_dimension=128):
        super().__init__(observation_space, features_dim=features_dim)
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.out_dimension = out_dimension
        
        self.network = nn.Sequential(
            nn.Conv1d(2, 8, 256, 64),
            nn.ReLU(),
            nn.Conv1d(8, 1, 8, 2)
        ).to(device)
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = observations.reshape(-1, observations.shape[2], 2)
        x = x.permute(0, 2, 1)
        out = self.network(x)
        return out