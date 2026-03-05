import numpy as np
import trimesh
from skimage import measure
from typing import Optional, Tuple


class MarchingCubesExtractor:

    def __init__(
        self,
        threshold: float = 0.5,
        spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    ):
        
        self.threshold = threshold
        self.spacing = spacing

    def extract(
        self,
        occupancy_grid: np.ndarray,
        bounds: Tuple[float, float] = (-1.0, 1.0)
    ) -> Optional[trimesh.Trimesh]:
        
        if occupancy_grid.max() < self.threshold or occupancy_grid.min() > self.threshold:
            print("Warning: No surface found in occupancy grid")
            return None

        try:
            vertices, faces, normals, _ = measure.marching_cubes(
                occupancy_grid,
                level=self.threshold,
                spacing=self.spacing
            )

            resolution = occupancy_grid.shape[0]
            vertices = vertices / (resolution - 1)  # [0, 1]
            vertices = vertices * (bounds[1] - bounds[0]) + bounds[0]  

            mesh = trimesh.Trimesh(
                vertices=vertices,
                faces=faces,
                vertex_normals=normals
            )

            mesh.fix_normals()

            return mesh

        except Exception as e:
            print(f"Marching cubes failed: {e}")
            return None


def extract_mesh(
    occupancy_grid: np.ndarray,
    threshold: float = 0.5,
    bounds: Tuple[float, float] = (-1.0, 1.0)
) -> Optional[trimesh.Trimesh]:
    extractor = MarchingCubesExtractor(threshold=threshold)
    return extractor.extract(occupancy_grid, bounds=bounds)
