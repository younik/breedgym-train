from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
from torch import nn
import numpy as np

class NoFeaturesExtractor(BaseFeaturesExtractor):

    def __init__(self, observation_space):
        super().__init__(observation_space, features_dim=observation_space.shape[-2])
        self.output_shape = observation_space.shape
        
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
    
    def __init__(self, observation_space, marker_effects, normalize=True):
        super().__init__(observation_space, features_dim=observation_space.shape[-2])
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.output_shape = observation_space.shape
        
        marker_effects = np.asarray(marker_effects)
        if normalize: 
            marker_effects = (marker_effects - marker_effects.min()) / (marker_effects.max() - marker_effects.min())
            marker_effects *= 2
            marker_effects -= 1
        self.values = torch.from_numpy(marker_effects).to(device)
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return observations * self.values[None, None, :, None] 
    
class AppendMarkerEffects(BaseFeaturesExtractor):
    
    def __init__(self, observation_space, marker_effects):
        super().__init__(observation_space, features_dim=observation_space.shape[-2])
        self.output_shape = *observation_space.shape[:-1], observation_space.shape[-1] + 1
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        marker_effects = np.asarray(marker_effects)
        self.marker_effects = torch.from_numpy(marker_effects).to(device)
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        mrk_effects = torch.broadcast_to(
            self.marker_effects[None, None, :], observations.shape[:-1]
        )
        out = torch.cat((observations, mrk_effects[..., None]), dim=-1)
        return out
    
    
class MaxMinMarkerEffects(BaseFeaturesExtractor):
    
    def __init__(self, observation_space, marker_effects):
        super().__init__(observation_space, features_dim=observation_space.shape[-2])
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.output_shape = *observation_space.shape[:-1], observation_space.shape[-1] + 2

        marker_effects = np.asarray(marker_effects)
        marker_effects = (marker_effects - marker_effects.min()) / (marker_effects.max() - marker_effects.min())
        marker_effects *= 2
        marker_effects -= 1
        self.values = torch.from_numpy(marker_effects).to(device)
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        masked_mrk = observations * self.values[None, None, :, None]
        out = torch.cat(
            (
                masked_mrk,
                masked_mrk.max(dim=-1, keepdim=True).values,
                masked_mrk.min(dim=-1, keepdim=True).values
            ),
            dim=-1
        )
        
        return out