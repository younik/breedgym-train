from typing import Tuple
import torch
from torch.distributions import Normal
from torch import nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.distributions import Distribution, DiagGaussianDistribution, sum_independent_dims

class FactoredSelectionIndex(nn.Module):

    def __init__(
        self,
        features_dim,
        gen_features_dim=1,
        policy_hiddens=[16],
        value_hiddens=[],
    ) -> None:
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if policy_hiddens is None:
            self.policy_hiddens = []
        elif isinstance(policy_hiddens, int):
            self.policy_hiddens = [policy_hiddens]
        else:
            self.policy_hiddens = policy_hiddens

        if value_hiddens is None:
            self.value_hiddens = []
        elif isinstance(value_hiddens, int):
            self.value_hiddens = [value_hiddens]
        else:
            self.value_hiddens = value_hiddens

        self.latent_dim_pi = 1
        self.latent_dim_vf = 1

        self.gen_projector_actor = torch.jit.script(
            nn.Sequential(
                nn.Linear(1, gen_features_dim),
                nn.LeakyReLU()
            )
        )
        self.gen_projector_critic = torch.jit.script(
            nn.Sequential(
                nn.Linear(1, gen_features_dim),
                nn.LeakyReLU()
            )
        )

        policy_net = self._make_net(
            self.policy_hiddens, features_dim=features_dim + gen_features_dim
        )
        policy_net.append(nn.Tanh())
        self.policy_net = torch.jit.script(policy_net)

        value_net = self._make_net(
            self.value_hiddens,  features_dim=features_dim + gen_features_dim
        )
        self.value_net = torch.jit.script(value_net)

    def _make_net(self, hiddens, features_dim) -> nn.Module:
        net = nn.Sequential()
        current_features = features_dim
        for n_features in hiddens:
            net.append(nn.Linear(current_features, n_features))
            current_features = n_features
            net.append(nn.ReLU())

        net.append(nn.Linear(current_features, 1))
        return net.to(self.device)

    def forward_actor(self, x):
        features = x['obs']
        gen_number = x['gen_number']
        gen_features = self.gen_projector_actor(gen_number)
        gen_features = torch.broadcast_to(
            gen_features[:, None, :],
            size=(*features.shape[:-1], gen_features.shape[-1])
        )
        out = self.policy_net(torch.cat([features, gen_features], dim=-1))
        return out.squeeze()

    def forward_critic(self, x):
        features = x['obs']
        gen_number = x['gen_number']
        gen_features = self.gen_projector_critic(gen_number)
        gen_features = torch.broadcast_to(
            gen_features[:, None, :],
            size=(*features.shape[:-1], gen_features.shape[-1])
        )
        out = self.value_net(torch.cat([features, gen_features], dim=-1))
        #out = torch.mean(out.squeeze(), axis=-1)
        return out

    def forward(self, x):
        return self.forward_actor(x), self.forward_critic(x)


class FactoredSelectionAC(ActorCriticPolicy):

    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        net_arch=None,
        gen_features_dim=1,
        policy_hiddens=[16],
        value_hiddens=[],
        activation_fn=nn.Tanh,
        *args,
        **kwargs,
    ):

        self.policy_hiddens = policy_hiddens
        self.value_hiddens = value_hiddens
        self.gen_features_dim = gen_features_dim

        super(FactoredSelectionAC, self).__init__(
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
        self.log_std = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.value_net = nn.Identity()
        
        self.action_dist = FactoredDiagGaussianDistribution(get_action_dim(action_space))


    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = FactoredSelectionIndex(
            features_dim=self.features_extractor.features_dim,
            gen_features_dim=self.gen_features_dim,
            policy_hiddens=self.policy_hiddens,
            value_hiddens=self.value_hiddens,
        )


class FactoredDiagGaussianDistribution(DiagGaussianDistribution):
    
    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Get the log probabilities of actions according to the distribution.
        Note that you must first call the ``proba_distribution()`` method.

        :param actions:
        :return:
        """
        log_prob = self.distribution.log_prob(actions)
        #return sum_independent_dims(log_prob)
        return log_prob
        