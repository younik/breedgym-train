import torch
from torch import nn
from stable_baselines3.common.policies import ActorCriticPolicy


CONV1_DEFAULT = {"out_channels": 64, "kernel_size": 256, "stride": 32}
CONV2_DEFAULT = {"out_channels": 16, "kernel_size": 8, "stride": 2}


class SelectionIndex(nn.Module):

    def __init__(
        self,
        input_shape,
        gen_features_dim=1,
        policy_hiddens=[16],
        value_hiddens=[],
        conv1_kwargs=CONV1_DEFAULT,
        conv2_kwargs=CONV2_DEFAULT,
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

        self.shared_network = nn.Sequential(
            nn.Conv1d(input_shape[-1], **conv1_kwargs),
            nn.ReLU(),
            nn.Conv1d(conv1_kwargs["out_channels"], **conv2_kwargs),
            nn.ReLU(),
        ).to(self.device)

        self.gen_projector_actor = nn.Sequential(
            nn.Linear(1, gen_features_dim),
            nn.LeakyReLU()
        )
        self.gen_projector_critic = nn.Sequential(
            nn.Linear(1, gen_features_dim),
            nn.LeakyReLU()
        )

        sample = torch.zeros(input_shape, device=self.device)
        with torch.no_grad():
            out_sample, _ = self.shared_forward(sample)

        self.policy_net = self._make_net(
            self.policy_hiddens, features_dim=out_sample.shape[-1] + gen_features_dim
        )
        self.policy_net.append(nn.Tanh())

        self.value_net = self._make_net(
            self.value_hiddens,  features_dim=out_sample.shape[-1] + gen_features_dim
        )

        # self.actor_w = torch.nn.Parameter(
        #     torch.Tensor([1, 0]).to(self.device),
        #     requires_grad=False
        # ).to(self.device)
        # self.value_w = torch.nn.Parameter(
        #     torch.Tensor([1, 0]).to(self.device),
        #     requires_grad=False
        # )

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
        features, gebvs = self.shared_forward(x['obs'])
        return self._forward_actor(features, gebvs, x["gen_number"])

    def _forward_actor(self, features, gebvs, gen_number):
        gen_features = self.gen_projector_actor(gen_number)
        gen_features = torch.broadcast_to(
            gen_features[:, None, :],
            size=(*features.shape[:-1], gen_features.shape[-1])
        )
        out = self.policy_net(torch.cat([features, gen_features], dim=-1))
        # out = torch.cat([out, gebvs[..., None]], dim=-1) @ self.actor_w.T
        return out.squeeze()

    def forward_critic(self, x):
        features, gebvs = self.shared_forward(x['obs'])
        return self._forward_critic(features, gebvs, x["gen_number"])

    def _forward_critic(self, features, gebvs, gen_number):
        gen_features = self.gen_projector_critic(gen_number)
        gen_features = torch.broadcast_to(
            gen_features[:, None, :],
            size=(*features.shape[:-1], gen_features.shape[-1])
        )
        out = self.value_net(torch.cat([features, gen_features], dim=-1))
        # out = torch.cat([out, gebvs[..., None]], dim=-1) @ self.value_w.T
        out = torch.mean(out.squeeze(), axis=-1)
        return out

    def shared_forward(self, x):
        gebvs = None  # x[..., :2].sum(axis=(-2, -1))
        batch_pop = x.reshape(-1, x.shape[-2], x.shape[-1])
        batch_pop = batch_pop.permute(0, 2, 1)

        features = self.shared_network(batch_pop)
        return features.reshape(*x.shape[:-2], -1), gebvs

    def forward(self, x):
        features, gebvs = self.shared_forward(x['obs'])
        return self._forward_actor(features, gebvs, x["gen_number"]), self._forward_critic(features, gebvs, x["gen_number"])


class SelectionAC(ActorCriticPolicy):

    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        net_arch=None,
        gen_features_dim=1,
        policy_hiddens=[16],
        value_hiddens=[],
        conv1_kwargs=CONV1_DEFAULT,
        conv2_kwargs=CONV2_DEFAULT,
        activation_fn=nn.Tanh,
        *args,
        **kwargs,
    ):

        self.policy_hiddens = policy_hiddens
        self.value_hiddens = value_hiddens
        self.conv1_kwargs = conv1_kwargs
        self.conv2_kwargs = conv2_kwargs
        self.gen_features_dim = gen_features_dim

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
            gen_features_dim=self.gen_features_dim,
            policy_hiddens=self.policy_hiddens,
            value_hiddens=self.value_hiddens,
            conv1_kwargs=self.conv1_kwargs,
            conv2_kwargs=self.conv2_kwargs,
        )
