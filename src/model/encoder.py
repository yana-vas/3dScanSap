
import torch
import torch.nn as nn
import torchvision.models as models


class ResNetEncoder(nn.Module):

    def __init__(self, latent_dim: int = 256):
        
        super().__init__()
        self.latent_dim = latent_dim

        weights = models.ResNet18_Weights.IMAGENET1K_V1
        self.resnet = models.resnet18(weights=weights)

        num_features = self.resnet.fc.in_features

        self.resnet.fc = nn.Linear(num_features, latent_dim)

        nn.init.xavier_uniform_(self.resnet.fc.weight)
        nn.init.zeros_(self.resnet.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resnet(x)

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def freeze_backbone(self) -> None:
        for name, param in self.resnet.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for param in self.resnet.parameters():
            param.requires_grad = True
