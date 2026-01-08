
import torch
import torch.nn as nn


class OccupancyDecoder(nn.Module):


    def __init__(self, latent_dim: int = 256, hidden_dim: int = 256, num_layers: int = 5):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # our input: latent_dim + 3 (x, y, z)
        input_dim = latent_dim + 3

        layers = []

        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU(inplace=True))

        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Linear(hidden_dim, 1))
        self.mlp = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)


    def forward(self, latent: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
       
        batch_size = latent.shape[0]
        num_points = points.shape[1]

        latent_expanded = latent.unsqueeze(1).expand(-1, num_points, -1)

        features = torch.cat([latent_expanded, points], dim=-1)
        features_flat = features.reshape(-1, self.latent_dim + 3)

        output_flat = self.mlp(features_flat)
        output_flat = torch.sigmoid(output_flat)
        output = output_flat.reshape(batch_size, num_points, 1)

        return output



    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
