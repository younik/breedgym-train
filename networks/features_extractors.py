from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch


class NoFeaturesExtractor(BaseFeaturesExtractor):

    def __init__(self, observation_space):
        super(NoFeaturesExtractor, self).__init__(observation_space, features_dim=1)
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return observations