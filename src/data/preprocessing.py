import torch
from PIL import Image
from torchvision import transforms
from typing import Union, Tuple
import numpy as np

class ImagePreprocessor:
    def __init__(self, image_size: int = 224, mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
        std: Tuple[float, ...] = (0.229, 0.224, 0.225)
    ):
        self.image_size = image_size
        self.mean = mean
        self.std = std

        self.transform = transforms.Compose(
            [transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])
        
        self.transform_with_augmentation = transforms.Compose([
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    def __call__(self,image: Union[Image.Image, np.ndarray, str], augment = False) -> torch.Tensor:
        
        if isinstance(image, str):
            image = Image.open(image)

        # np to PIL
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        
        if image.mode != 'RGB':
            image = image.convert('RGB')

        if augment:
            return self.transform_with_augmentation(image)
        return self.transform(image)
    

    def denormalize(self, tensor: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(self.mean, device=tensor.device).view(3, 1, 1)
        std = torch.tensor(self.std, device=tensor.device).view(3, 1, 1)
        return tensor * std + mean
    