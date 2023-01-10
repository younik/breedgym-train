import torch
from torch import nn
from stable_baselines3.common.policies import ActorCriticPolicy


class SelectionIndex(nn.Module):

    def __init__(self, input_shape, policy_hiddens=[16], value_hiddens=[]) -> None:
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.latent_dim_pi = 1
        self.latent_dim_vf = 1
        
        self.shared_network = nn.Sequential(
            nn.Conv1d(input_shape[-1], 8, 256, 64),
            nn.ReLU(),
            nn.Conv1d(8, 1, 8, 2)
        ).to(self.device)
        
        sample = torch.zeros(input_shape, device=self.device)
        with torch.no_grad():
            out_sample = self.shared_forward(sample)
        
        self.policy_hiddens = policy_hiddens
        self.policy_net = self._make_net(
            self.policy_hiddens, features_dim=out_sample.shape[-1]
        )
        self.policy_net.append(nn.Tanh())
        
        self.value_hiddens = value_hiddens
        self.value_net = self._make_net(
            self.value_hiddens,  features_dim=out_sample.shape[-1]
        )

    def _make_net(self, hiddens, features_dim):
        net = nn.Sequential()
        current_features = features_dim
        for n_features in hiddens:
            net.append(nn.Linear(current_features, n_features))
            current_features = n_features
            net.append(nn.ReLU())

        net.append(nn.Linear(current_features, 1))
        return net.to(self.device)

    def forward_actor(self, x):
        features = self.shared_forward(x)
        return self._forward_actor(features)

    def _forward_actor(self, features):
        out = self.policy_net(features)
        return out.squeeze()

    def forward_critic(self, x):
        features = self.shared_forward(x)
        return self._forward_critic(features)

    def _forward_critic(self, features):            
        out = self.value_net(features)
        out = torch.mean(out.squeeze(), axis=-1)
        return out

    def shared_forward(self, x):
        batch_pop = x.reshape(-1, x.shape[-2], x.shape[-1])
        batch_pop = batch_pop.permute(0, 2, 1)

        features = self.shared_network(batch_pop)
        return features.reshape(*x.shape[:-2], *features.shape[1:]).squeeze()

    def forward(self, x):
        features = self.shared_forward(x)
        return self._forward_actor(features), self._forward_critic(features)


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
            input_shape=self.features_extractor.output_shape,
            policy_hiddens=self.policy_hiddens,
            value_hiddens=self.value_hiddens
        )
