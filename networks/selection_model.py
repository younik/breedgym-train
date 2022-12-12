import torch
from torch import nn
from stable_baselines3.common.policies import ActorCriticPolicy


class SelectionIndex(nn.Module):

    def __init__(self, features_dim, action_shape, policy_hiddens=[16], value_hiddens=[]) -> None:
        super().__init__()
        self.features_dim = features_dim
        self.action_shape = action_shape
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.latent_dim_pi = 1
        self.latent_dim_vf = 1
        self.policy_net = self._make_net(policy_hiddens)
        self.policy_net.append(nn.Tanh())
        self.value_net = self._make_net(value_hiddens)

    def _make_net(self, hiddens):
        net = nn.Sequential()
        current_features = self.features_dim
        for n_features in hiddens:
            net.append(nn.Linear(current_features, n_features))
            current_features = n_features
            net.append(nn.ReLU())

        net.append(nn.Linear(current_features, 1))
        return net.to(self.device)

    def forward_actor(self, features):
        out = self.policy_net(features)
        return out.reshape((-1, ) + self.action_shape)

    def forward_critic(self, features):
        out = self.value_net(features)
        out = out.reshape((-1, ) + self.action_shape)
        out = torch.mean(out, axis=1)
        return out

    def forward(self, x):
        return self.forward_actor(x), self.forward_critic(x)


class SelectionAC(ActorCriticPolicy):
    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        net_arch=None,
        policy_hiddens=[16],
        value_hiddens=[],
        activation_fn=nn.Tanh,
        *args,
        **kwargs,
    ):

        self.policy_hiddens = policy_hiddens
        self.value_hiddens = value_hiddens
        super(SelectionAC, self).__init__(
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
        self.mlp_extractor = SelectionIndex(
            self.features_dim,
            action_shape=self.action_space.shape,
            policy_hiddens=self.policy_hiddens,
            value_hiddens=self.value_hiddens
        )
