import torch
import numpy as np
import trimesh
from pathlib import Path
from PIL import Image, ImageDraw
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional


from .preprocessing import ImagePreprocessor
from .point_sampling import PointSampler

class ShapeNetDataset(Dataset):

    CATEGORIES = {
        '02691156': 'airplane',
        '02828884': 'bench',
        '02933112': 'cabinet',
        '02958343': 'car',
        '03001627': 'chair',
        '03211117': 'display',
        '03636649': 'lamp',
        '03691459': 'speaker',
        '04090263': 'rifle',
        '04256520': 'sofa',
        '04379243': 'table',
        '04401088': 'telephone',
        '04530566': 'vessel',
    }


    def __init__(self,
        root: str,
        split: str = 'train',
        categories: Optional[List[str]] = None,
        num_points: int = 2048,
        image_size: int = 224,
        augment: bool = False,
        max_samples: Optional[int] = None):

        self.root = Path(root)
        self.split = split
        self.categories = categories or list(self.CATEGORIES.keys())
        self.num_points = num_points
        self.augment = augment

        self.preprocessor = ImagePreprocessor(image_size=image_size)
        self.point_sampler = PointSampler(num_points=num_points)

        self.samples = self._find_samples(max_samples)
        print(f"Found {len(self.samples)} samples for {split}")
    

    def _find_samples(self, max_samples: Optional[int] = None) -> List[Dict]:
        samples = []

        for category in self.categories:
            category_path = self.root / category

            if not category_path.exists():
                continue

            model_dirs = list(category_path.iterdir())

            # 80% train, 10% validate, 10% test
            n = len(model_dirs)
            if self.split == 'train':
                model_dirs = model_dirs[:int(0.8 * n)]
            elif self.split == 'val':
                model_dirs = model_dirs[int(0.8 * n):int(0.9 * n)]
            else:  
                model_dirs = model_dirs[int(0.9 * n):]

            for model_dir in model_dirs:
                obj_path = model_dir / 'models' / 'model_normalized.obj'

                if obj_path.exists():
                    samples.append({
                        'category': category,
                        'model_id': model_dir.name,
                        'obj_path': str(obj_path),
                        'image_dir': str(model_dir / 'images'),
                    })

                    if max_samples and len(samples) >= max_samples:
                        return samples

        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        mesh = trimesh.load(sample['obj_path'], force='mesh')
        mesh = self._normalize_mesh(mesh)

        points, occupancy = self.point_sampler.sample(mesh)

        image = self._get_image(sample, mesh)
        image_tensor = self.preprocessor(image, augment=self.augment)

        return {
            'image': image_tensor,
            'points': torch.from_numpy(points),
            'occupancy': torch.from_numpy(occupancy).unsqueeze(-1),
            'category': sample['category'],
            'model_id': sample['model_id'], }
    
    def _normalize_mesh(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        mesh.vertices -= mesh.centroid

        scale = np.abs(mesh.vertices).max()
        if scale > 0:
            mesh.vertices /= scale
            mesh.vertices *= 0.9 

        return mesh
    
    def _get_image(self, sample: Dict, mesh: trimesh.Trimesh) -> Image.Image:
        image_dir = Path(sample['image_dir'])

        if image_dir.exists():
            images = list(image_dir.glob('*.png')) + list(image_dir.glob('*.jpg'))
            if images:
                image_path = np.random.choice(images)
                return Image.open(str(image_path)).convert('RGB')

        return self._render_mesh(mesh)
    
    def _render_mesh(self, mesh: trimesh.Trimesh) -> Image.Image:

        size = 224
        image = Image.new('RGB', (size, size), 'white')
        draw = ImageDraw.Draw(image)

        vertices_2d = mesh.vertices[:, :2]  
        vertices_2d = (vertices_2d + 1) * (size / 2)  

        for face in mesh.faces:
            for i in range(3):
                v1 = vertices_2d[face[i]]
                v2 = vertices_2d[face[(i + 1) % 3]]
                draw.line([v1[0], v1[1], v2[0], v2[1]], fill='black', width=1)

        return image
    

def get_dataloader(root: str, split: str = 'train', batch_size: int = 16, num_workers: int = 4, **kwargs) -> DataLoader:
    
    dataset = ShapeNetDataset(root=root, split=split, **kwargs)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == 'train'))
