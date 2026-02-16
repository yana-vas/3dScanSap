import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple

from .preprocessing import ImagePreprocessor


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


class ShapeNetDataset(Dataset):

    def __init__(
        self,
        root: str,
        split: str = 'train',
        categories: Optional[List[str]] = None,
        num_points: int = 2048,
        image_size: int = 224,
        augment: bool = False,
        max_samples: Optional[int] = None,
    ):
        self.root = Path(root)
        self.split = split
        self.categories = categories or list(CATEGORIES.keys())
        self.num_points = num_points
        self.augment = augment


        self.preprocessor = ImagePreprocessor(image_size=image_size)
        self.samples = self._find_samples(max_samples)
        print(f"[{split}] Found {len(self.samples)} samples "
              f"across {len(set(s['category'] for s in self.samples))} categories")

    def _find_samples(self, max_samples: Optional[int] = None) -> List[Dict]:
        samples: List[Dict] = []

        for cat_id in sorted(self.categories):
            cat_dir = self.root / cat_id
            if not cat_dir.exists():
                continue

            model_dirs = sorted([
                d for d in cat_dir.iterdir()
                if d.is_dir() and (d / 'points.npz').exists()
            ])

            rng = np.random.RandomState(42)
            indices = rng.permutation(len(model_dirs))

            n = len(model_dirs)
            train_end = int(0.8 * n)
            val_end = int(0.9 * n)

            if self.split == 'train':
                selected = indices[:train_end]
            elif self.split == 'val':
                selected = indices[train_end:val_end]
            else:
                selected = indices[val_end:]

            for idx in selected:
                d = model_dirs[idx]
                samples.append({
                    'category': cat_id,
                    'model_id': d.name,
                    'dir': str(d),
                })
                if max_samples and len(samples) >= max_samples:
                    return samples

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        info = self.samples[idx]
        model_dir = Path(info['dir'])

        image = self._load_image(model_dir)
        image_tensor = self.preprocessor(image, augment=self.augment)

        points, occupancy = self._load_points(model_dir)

        return {
            'image': image_tensor,
            'points': torch.from_numpy(points),
            'occupancy': torch.from_numpy(occupancy).unsqueeze(-1),
            'category': info['category'],
            'model_id': info['model_id'],
        }

    def _load_image(self, model_dir: Path) -> Image.Image:
        img_dir = model_dir / 'img_choy2016'

        if img_dir.exists():
            views = sorted(img_dir.glob('*.jpg')) + sorted(img_dir.glob('*.png'))
            if views:
                path = views[np.random.randint(len(views))]
                return Image.open(str(path)).convert('RGB')

        return Image.new('RGB', (224, 224), 'white')

    def _load_points(self, model_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
        data = np.load(str(model_dir / 'points.npz'))

        all_points = data['points'].astype(np.float32)

        raw_occ = data['occupancies']
        all_occ = np.unpackbits(raw_occ)[:all_points.shape[0]]
        all_occ = all_occ.astype(np.float32)

        n = all_points.shape[0]
        replace = n < self.num_points
        choice = np.random.choice(n, size=self.num_points, replace=replace)
        points = all_points[choice]
        occupancy = all_occ[choice]

        return points, occupancy


def get_dataloader(
    root: str,
    split: str = 'train',
    batch_size: int = 16,
    num_workers: int = 4,
    **kwargs,
) -> DataLoader:
    dataset = ShapeNetDataset(root=root, split=split, **kwargs)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == 'train'),
    )