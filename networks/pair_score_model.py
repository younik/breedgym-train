from typing import Sequence
import torch
from torch import nn
import torch.nn.functional as F
from stable_baselines3.common.policies import ActorCriticPolicy


class PairScoreModel(nn.Module):

    def __init__(
        self,
        features_dim,
        value_hiddens=[32],
        actor_hiddens=[128],
        gen_features_dim=1,
        ) -> None:
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gen_features_dim = gen_features_dim

        if isinstance(value_hiddens, Sequence):
            self.value_hiddens = value_hiddens
        else:
            self.value_hiddens = [int(value_hiddens)]
        if isinstance(actor_hiddens, Sequence):
            self.actor_hiddens = actor_hiddens
        else:
            self.actor_hiddens = [int(actor_hiddens)]

        self.latent_dim_pi = 1
        self.latent_dim_vf = 1

        self.gen_actor_net = torch.jit.script(
            nn.Sequential(
                nn.Linear(1, self.gen_features_dim),
                nn.LeakyReLU()
            ).to(self.device)
        )
        
        value_features = [features_dim + gen_features_dim] + self.value_hiddens + [1]
        self.value_net = self._make_net(value_features)
        
        actor_features = [features_dim + gen_features_dim] + self.actor_hiddens
        self.features_actor_net = nn.Identity()
        if len(actor_features) > 1:
            self.features_actor_net = self._make_net(actor_features)
            
        self.actor_keys = torch.jit.script(nn.Linear(64, 64))
        self.actor_queries = torch.jit.script(nn.Linear(64, 64))

    def _make_net(self, features):
        net = nn.Sequential(
            nn.Linear(features[0], features[1])
        )
        for i in range(1, len(features) - 1):
            net.append(nn.ReLU())
            net.append(
                nn.Linear(features[i], features[i+1])
            )

        return torch.jit.script(net.to(self.device))

    def forward_actor(self, x):
        features = self.shared_net(x)
        return self._forward_actor(features)

    def _forward_actor(self, input_):
        features = self.features_actor_net(input_)
        # keys = self.actor_keys(features)
        # queries = self.actor_queries(features)
        # out = torch.einsum('bnf,bmf->bnm', keys, queries)
        out = -F.cosine_similarity(features.unsqueeze(1), features.unsqueeze(2), dim=-1)
        out = out.flatten(start_dim=-2)
        return out

    def forward_critic(self, x):
        features = self.shared_net(x)
        return self._forward_critic(features)
        
    def _forward_critic(self, x):
        values = self.value_net(x)
        return values.squeeze(-1).mean(axis=-1)

    def shared_net(self, x):
        features = x['obs']
        gen_number = x['gen_number']
        gen_features = self.gen_actor_net(gen_number)
        gen_features = torch.broadcast_to(
            gen_features[:, None, :],
            size=(*features.shape[:-1], gen_features.shape[-1])
        )
        return torch.cat([features, gen_features], dim=-1)

    def forward(self, x):
        features = self.shared_net(x)
        return self._forward_actor(features), self._forward_critic(features)


class PairScoreAC(ActorCriticPolicy):

    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        net_arch=None,
        value_hiddens=[],
        actor_hiddens=[256],
        gen_features_dim=1,
        activation_fn=nn.Tanh,
        *args,
        **kwargs,
    ):

        self.value_hiddens = value_hiddens
        self.actor_hiddens = actor_hiddens
        self.gen_features_dim = gen_features_dim

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
        self.value_net = nn.Identity()

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = PairScoreModel(
            features_dim=self.features_extractor.features_dim,
            value_hiddens=self.value_hiddens,
            actor_hiddens=self.actor_hiddens,
            gen_features_dim=self.gen_features_dim
        )