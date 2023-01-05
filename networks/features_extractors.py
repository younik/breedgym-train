from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
from torch import nn
import numpy as np

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
    

class MaskedMarkerEffects(BaseFeaturesExtractor):
    
    def __init__(self, observation_space, marker_effects, features_dim=10_000):
        super().__init__(observation_space, features_dim=features_dim)
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        marker_effects = np.asarray(marker_effects)
        self.marker_effects = torch.from_numpy(marker_effects).to(device)
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return observations * self.marker_effects[None, None, :, None] 
    
class AppendMarkerEffects(BaseFeaturesExtractor):
    
    def __init__(self, observation_space, marker_effects, features_dim=10_000):
        super().__init__(observation_space, features_dim=features_dim)
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        marker_effects = np.asarray(marker_effects)
        self.marker_effects = torch.from_numpy(marker_effects).to(device)
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        mrk_effects = torch.broadcast_to(
            self.marker_effects[None, None, :], observations.shape[:-1]
        )
        out = torch.cat((observations, mrk_effects[..., None]), dim=-1)
        return out