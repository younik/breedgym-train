import torch
from torch import nn
from torch.nn.functional import multi_head_attention_forward
from stable_baselines3.common.policies import ActorCriticPolicy


class GEBVCorrAttention(nn.Module):

    def __init__(self, key_dim) -> None:
        super().__init__()
        self.key_dim = key_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.latent_dim_pi = 1
        self.latent_dim_vf = 1
        
        self.actor_attention = nn.MultiheadAttention(
            embed_dim=self.key_dim,
            num_heads=1,
            vdim=2,
            kdim=2,
            batch_first=True,
            device=self.device,
        )
        self.value_attention = nn.MultiheadAttention(
            embed_dim=self.key_dim,
            num_heads=1,
            vdim=2,
            kdim=2,
            batch_first=True,
            device=self.device,
        )
        self.actor_proj = nn.Linear(self.key_dim, 1, device=self.device)
        self.value_proj = nn.Linear(self.key_dim, 1, device=self.device)
        
        # Use a learned query (bias of W_k)
        self.query = torch.zeros((1, self.key_dim), device=self.device)

    def forward(self, x):
        batch = torch.stack(list(x.values()), dim=-1)
        
        query = self.query.expand(batch.shape[0], *self.query.shape)
        action, _ = self.actor_attention(query, batch, batch)
        value, _ = self.value_attention(query, batch, batch)
        
        return self.actor_proj(action).squeeze(), self.value_proj(value).squeeze()
        


class GEBVCorrAC(ActorCriticPolicy):
    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        key_dim=8,
        net_arch=None,
        activation_fn=nn.Tanh,
        *args,
        **kwargs,
    ):
        
        self.key_dim = key_dim
        super(GEBVCorrAC, self).__init__(
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
        self.mlp_extractor = GEBVCorrAttention(self.key_dim)
