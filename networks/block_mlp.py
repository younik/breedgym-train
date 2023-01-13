import torch
from torch import nn
import torch.nn.functional as F
from stable_baselines3.common.policies import ActorCriticPolicy

        
class BlockMLP(nn.Module):

    def __init__(self, block_per_chr, chr_lens, hidden_dim=1, custom_init=True) -> None:
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.block_per_chr = block_per_chr
        self.chr_lens = chr_lens
        self.n_chr = len(chr_lens)
        self.hidden_dim = hidden_dim
        self.custom_init = custom_init
        
        self.latent_dim_pi = 1
        self.latent_dim_vf = 1
        #TODO: refactor
        first_layer_blocks = []
        for chr_length in self.chr_lens:
            block_length = round(chr_length / self.block_per_chr)
            for _ in range(self.block_per_chr - 1):
                linear = nn.Linear(block_length, self.hidden_dim, device=self.device)
                if self.custom_init:
                    with torch.no_grad():
                        linear.weight += 1
                block = nn.Sequential(
                    linear,
                    nn.ReLU(),
                    *self._get_proj_layers()
                )
                first_layer_blocks.append(block)
                
            last_length = chr_length - block_length * (self.block_per_chr - 1)
            linear = nn.Linear(last_length, self.hidden_dim, device=self.device)
            if self.custom_init:
                with torch.no_grad():
                    linear.weight += 1
            block = nn.Sequential(
                linear,
                nn.ReLU(),
                *self._get_proj_layers()
            )
            first_layer_blocks.append(block)
        self.first_layer_blocks = nn.ModuleList(first_layer_blocks)
            
        self.second_layer_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.block_per_chr, self.hidden_dim, device=self.device),
                nn.ReLU(),
                *self._get_proj_layers()
            )
            for _ in range(self.n_chr)
        ])
        
        self.actor_head = nn.Linear(self.n_chr, 1, device=self.device)
        self.critic_head = nn.Linear(self.n_chr, 1, device=self.device)

        with torch.no_grad():
            self.actor_head.weight += 1
            self.critic_head.weight += 1

    def _get_proj_layers(self):
        if self.hidden_dim == 1:
            return []
        else:
            linear = nn.Linear(self.hidden_dim, 1, device=self.device)
            if self.custom_init:
                with torch.no_grad():
                    linear.weight += 1 / self.hidden_dim
            return [
                linear,
                nn.ReLU()
            ]

    def forward_actor(self, x):
        features = self.shared_backbone(x)
        return self._forward_actor(features)

    def forward_critic(self, x):
        features = self.shared_backbone(x)
        return self._forward_critic(features)
    
    def _forward_actor(self, features):
        out = self.actor_head(features).squeeze()
        return out

    def _forward_critic(self, features):
        out = self.critic_head(features).squeeze()
        return out.mean(axis=-1)

    def shared_backbone(self, x):
        x = x.permute(0, 1, 3, 2)
        ys = torch.empty(
            x.shape[:-1] + (self.block_per_chr * self.n_chr,),
            device=self.device
        )
        start_idx = 0
        for i, net in enumerate(self.first_layer_blocks):
            end_idx = start_idx + net[0].in_features
            ys[..., i] = net(x[..., start_idx:end_idx]).squeeze()
            start_idx = end_idx
            
        ys = torch.einsum("vnzl, vnzl -> vnl", F.softmax(ys, dim=-2), ys)
        
        outs = torch.empty(x.shape[:2] + (self.n_chr,), device=self.device)
        start_idx = 0
        for i, net in enumerate(self.second_layer_blocks):
            end_idx = start_idx + net[0].in_features
            outs[..., i] = net(ys[..., start_idx:end_idx]).squeeze()
            start_idx = end_idx

        return outs

    def forward(self, x):
        features = self.shared_backbone(x)
        return self._forward_actor(features), self._forward_critic(features)


class SelectionBlockMLP(ActorCriticPolicy):
    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        block_per_chr,
        chr_lens,
        hidden_dim=1,
        custom_init=True,
        net_arch=None,
        activation_fn=nn.Tanh,
        ortho_init=False, #TODO: remove if you don't initilize with 1
        *args,
        **kwargs,
    ):

        self.block_per_chr = block_per_chr
        self.chr_lens = chr_lens
        self.hidden_dim = hidden_dim
        super(SelectionBlockMLP, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            *args,
            ortho_init=ortho_init,
            **kwargs,
        )

        self.action_net = nn.Identity()
        self.value_net = nn.Identity()

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = BlockMLP(
            block_per_chr=self.block_per_chr,
            chr_lens=self.chr_lens,
            hidden_dim=self.hidden_dim,
        )
