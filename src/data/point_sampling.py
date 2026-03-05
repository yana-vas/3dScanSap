import numpy as np
import trimesh
from typing import Tuple, Optional

class PointSampler:

    def __init__(self, num_points: int=2048, surface_ratio: float = 0.5, noise_std: float = 0.05, bounds: Tuple[float, float] = (-1.0, 1.0)):

        self.num_points = num_points
        self.surface_ratio = surface_ratio
        self.noise_std = noise_std
        self.bounds = bounds

    def sample(self, mesh: trimesh.Trimesh) -> Tuple[np.ndarray, np.ndarray]:

            num_surface = int(self.num_points * self.surface_ratio)
            num_uniform = self.num_points - num_surface

            uniform_points = np.random.uniform(
                self.bounds[0], self.bounds[1],
                size=(num_uniform, 3)).astype(np.float32)


            if num_surface > 0 and len(mesh.vertices) > 0:
                surface_points, _ = trimesh.sample.sample_surface(mesh, num_surface)

                surface_points = surface_points + np.random.randn(num_surface, 3) * self.noise_std
                surface_points = surface_points.astype(np.float32)
            else:
                surface_points = np.random.uniform(
                    self.bounds[0], self.bounds[1],
                    size=(num_surface, 3)).astype(np.float32)


            points = np.concatenate([uniform_points, surface_points], axis=0)

            # occupancy (inside = 1, outside = 0)
            occupancy = mesh.contains(points).astype(np.float32)

            return points, occupancy
        
    def sample_uniform(self, num_points: Optional[int] = None) -> np.ndarray:
            n = num_points or self.num_points
            return np.random.uniform(
                self.bounds[0], self.bounds[1],
                size=(n, 3)
            ).astype(np.float32)
   