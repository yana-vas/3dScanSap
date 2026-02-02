from torch.utils.data import Dataset
from typing import Dict, List, Optional
from pathlib import Path


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
    
    # TODO - render mesh, get imgae, normalize mesh, __getitem__
    
