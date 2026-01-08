
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

from .encoder import ResNetEncoder
from .decoder import OccupancyDecoder


class OccupancyNetwork(nn.Module):

    def __init__(
        self,
        latent_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 5
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.encoder = ResNetEncoder(latent_dim=latent_dim)
        self.decoder = OccupancyDecoder(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers
        )

    def forward(
        self,
        images: torch.Tensor,
        points: torch.Tensor
    ) -> torch.Tensor:
        
        latent = self.encoder(images)
        occupancy = self.decoder(latent, points)
        return occupancy

    def encode(self, images: torch.Tensor) -> torch.Tensor:
#              to latent vector
        return self.encoder(images)

    def decode(
        self,
        latent: torch.Tensor,
        points: torch.Tensor
    ) -> torch.Tensor:
        
        return self.decoder(latent, points)

    def generate_occupancy_grid(
        self,
        images: torch.Tensor,
        resolution: int = 64,
        batch_size: int = 100000
    ) -> np.ndarray:
        
        self.eval()
        device = next(self.parameters()).device

        with torch.no_grad():
            latent = self.encode(images)  # (1, latent_dim)

            x = torch.linspace(-1, 1, resolution)
            y = torch.linspace(-1, 1, resolution)
            z = torch.linspace(-1, 1, resolution)

            xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')
            points = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)
            points = points.to(device)

            occupancy_list = []
            num_points = points.shape[0]

            for i in range(0, num_points, batch_size):
                batch_points = points[i:i + batch_size].unsqueeze(0)  #  (1, batch, 3)
                batch_occ = self.decode(latent, batch_points)  #         (1, batch, 1)
                occupancy_list.append(batch_occ.squeeze(0).squeeze(-1).cpu())

            occupancy = torch.cat(occupancy_list, dim=0)
            grid = occupancy.reshape(resolution, resolution, resolution).numpy()

        return grid

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'state_dict': self.state_dict(),
            'latent_dim': self.latent_dim,
            'hidden_dim': self.decoder.hidden_dim,
        }, path)
        print(f"Model saved to {path}")

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location='cpu')
        self.load_state_dict(checkpoint['state_dict'])
        print(f"Model loaded from {path}")

    @classmethod
    def from_checkpoint(cls, path: str) -> 'OccupancyNetwork':
        checkpoint = torch.load(path, map_location='cpu')
        model = cls(
            latent_dim=checkpoint['latent_dim'],
            hidden_dim=checkpoint['hidden_dim']
        )
        model.load_state_dict(checkpoint['state_dict'])
        return model

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class OccupancyLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()

    def forward(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        
        return self.bce(predicted, target)
    