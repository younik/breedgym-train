from typing import Sequence
import torch
from torch import nn
from stable_baselines3.common.policies import ActorCriticPolicy


class PairScoreModel(nn.Module):

    def __init__(
        self,
        input_shape,
        value_hiddens=[]
        ) -> None:
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if value_hiddens is None:
            self.value_hiddens = []
        elif isinstance(value_hiddens, Sequence):
            self.value_hiddens = value_hiddens
        else:
            self.value_hiddens = [int(value_hiddens)]
        
        self.latent_dim_pi = 1
        self.latent_dim_vf = 200
        
        self.shared_network = nn.Sequential(
            nn.Conv1d(input_shape[-1], 128, 256, 32),
            nn.ReLU(),
            nn.Conv1d(128, 16, 8, 2),
        ).to(self.device)
        
        self.pop_size = input_shape[0]
        sample = torch.zeros(input_shape, device=self.device)
        with torch.no_grad():
            out_sample = self.shared_forward(sample)
        
        features = [2 * out_sample.shape[-1]] + self.value_hiddens + [1]
        self.features_value_net = nn.Sequential()
        for i in range(len(features) - 1):
            self.features_value_net.append(
                nn.Linear(features[i], features[i+1])
            )
            self.features_value_net.append(nn.ReLU())

    def forward_actor(self, x):
        features = self.shared_forward(x)
        return self._forward_actor(features)

    def _forward_actor(self, features):        
        out = torch.einsum('bnf,bmf->bnm', features, features)
        return out.flatten(start_dim=1)

    def forward_critic(self, x):
        features = self.shared_forward(x)
        pair_scores = self._forward_actor(features)
        return self._forward_critic(features, pair_scores)
        
    def _forward_critic(self, features, pair_scores):
        _, top_indices = torch.topk(pair_scores, k=self.pop_size)
        top_rows = torch.div(top_indices, self.pop_size, rounding_mode='floor')
        top_cols = top_indices % self.pop_size
        top_crosses = torch.stack((top_rows, top_cols), dim=2)
        nenv_arange = torch.arange(features.shape[0])
        batched_features = features[nenv_arange[:, None, None], top_crosses]
        batched_features = batched_features.flatten(start_dim=-2)
        values = self.features_value_net(batched_features)
        return values.reshape(values.shape[:-1])

    def shared_forward(self, x):
        batch_pop = x.reshape(-1, x.shape[-2], x.shape[-1])
        batch_pop = batch_pop.permute(0, 2, 1)

        features = self.shared_network(batch_pop)
        return features.reshape(*x.shape[:-2], -1)

    def forward(self, x):
        features = self.shared_forward(x)
        pair_scores = self._forward_actor(features)
        return pair_scores, self._forward_critic(features, pair_scores)


class PairScoreAC(ActorCriticPolicy):
    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        net_arch=None,
        value_hiddens=[],
        activation_fn=nn.Tanh,
        *args,
        **kwargs,
    ):
        
        self.value_hiddens = value_hiddens

        super(PairScoreAC, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            *args,
            **kwargs,
        )

        # Disable orthogonal initialization
        self.ortho_init = False

        self.action_net = nn.Identity()

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = PairScoreModel(
            input_shape=self.features_extractor.output_shape,
            value_hiddens=self.value_hiddens,
        )
