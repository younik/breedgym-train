from torch import nn, optim
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import lightning.pytorch as pl
import pandas as pd
from breeding_gym.utils.paths import DATA_PATH
import wandb


class PopulationGenerator(Dataset):
    
    def __init__(self, genetic_map, seed=None) -> None:
        super().__init__()
        self._generator = np.random.default_rng(seed=seed)
        genetic_map = pd.read_table(genetic_map, sep="\t", dtype={"Yield": "float32"})
        self.markers = genetic_map["Yield"].to_numpy()
        self.markers /= self.markers.std()
        self.sample_size = len(self.markers), 2
    
    def __len__(self):
        return 1024
    
    def __getitem__(self, _):
        ind = self._generator.integers(0, 2, size=self.sample_size, dtype=np.int8)
        return ind * self.markers[:, None]


class MyNet(pl.LightningModule):
    
    def __init__(self, features_dim, conv1_kwargs, conv2_kwargs, sample_size) -> None:
        super().__init__()
        
        net = nn.Sequential(
            nn.Conv1d(2, **conv1_kwargs),
            nn.ReLU(),
            nn.Conv1d(conv1_kwargs["out_channels"], **conv2_kwargs),
            nn.ReLU(),
            nn.Flatten()
        )
        
        with torch.no_grad():
            sample = torch.zeros((1,) + sample_size)
            out = net(sample.permute(0, 2, 1))
            net.append(nn.Linear(out.shape[-1], features_dim))
        
        net.append(nn.ReLU())
        net.append(nn.Linear(features_dim, 3))
        
        self.net = net
    
    def training_step(self, batch, batch_idx):
        y = torch.empty((len(batch), 3), device=batch.device)
        
        y[:, 0] = batch.mean(axis=(-2, -1))
        y[:, 1] = batch.max(axis=-1).values.mean(axis=-1)
        y[:, 2] = batch.min(axis=-1).values.mean(axis=-1)

        y_hat = self.net(batch.permute(0, 2, 1))
        
        loss_gebv = nn.functional.mse_loss(y_hat[:, 0], y[:, 0])
        loss_max = nn.functional.mse_loss(y_hat[:, 1], y[:, 1])
        loss_min = nn.functional.mse_loss(y_hat[:, 2], y[:, 2])
        loss = loss_gebv + loss_max + loss_min

        wandb.log({"train_loss": loss.item()})
        wandb.log({"train_loss_gebv": loss_gebv.item()})
        wandb.log({"train_loss_max": loss_max.item()})
        wandb.log({"train_loss_min": loss_min.item()})

        return loss
        
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


def main(config):
    conv1_kwargs={
        "out_channels": int(vars(config).get("out_channels1", 64)),
        "kernel_size": int(vars(config).get("kernel_size1", 256)),
        "stride": int(vars(config).get("stride1", 32))
    }
    conv2_kwargs={
        "out_channels": int(vars(config).get("out_channels2", 16)),
        "kernel_size": int(vars(config).get("kernel_size2", 8)),
        "stride": int(vars(config).get("stride2", 2))
    }    

    dataset = PopulationGenerator(DATA_PATH.joinpath(config.genetic_map), seed=config.seed)
    net = MyNet(int(config.features_dim), conv1_kwargs, conv2_kwargs, dataset.sample_size)
    train_loader = DataLoader(dataset, batch_size=64)
    
    trainer = pl.Trainer(max_epochs=5)
    trainer.fit(model=net, train_dataloaders=train_loader, val_dataloaders=train_loader)
    
    state_dict = net.net.state_dict()
    state_dict.pop('7.weight')
    state_dict.pop('7.bias')
    torch.save(state_dict, f"{config.model_path}/{config.unique_name}")