import trimesh
from pathlib import Path
from typing import Optional


class MeshExporter:

    # stl (binary and ascii), obj, ply
    def __init__(self, create_dirs: bool = True):
        self.create_dirs = create_dirs

    def _prepare_path(self, path: str) -> Path:
        p = Path(path)
        if self.create_dirs:
            p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def export_stl(
        self,
        mesh: trimesh.Trimesh,
        path: str,
        binary: bool = True
    ) -> bool:
        try:
            output_path = self._prepare_path(path)

            if binary:
                mesh.export(str(output_path), file_type='stl')
            else:
                mesh.export(str(output_path), file_type='stl_ascii')

            print(f"Exported STL: {output_path}")
            print(f"  Vertices: {len(mesh.vertices)}")
            print(f"  Faces: {len(mesh.faces)}")
            print(f"  Watertight: {mesh.is_watertight}")

            return True

        except Exception as e:
            print(f"Failed to export STL: {e}")
            return False

    def export_obj(self, mesh: trimesh.Trimesh, path: str) -> bool:
        try:
            output_path = self._prepare_path(path)
            mesh.export(str(output_path), file_type='obj')
            print(f"Exported OBJ: {output_path}")
            return True
        except Exception as e:
            print(f"Failed to export OBJ: {e}")
            return False

    def export_ply(self, mesh: trimesh.Trimesh, path: str) -> bool:
        try:
            output_path = self._prepare_path(path)
            mesh.export(str(output_path), file_type='ply')
            print(f"Exported PLY: {output_path}")
            return True
        except Exception as e:
            print(f"Failed to export PLY: {e}")
            return False


def save_mesh(
    mesh: trimesh.Trimesh,
    path: str,
    file_format: Optional[str] = None
) -> bool:
    exporter = MeshExporter()

    if file_format is None:
        ext = Path(path).suffix.lower()
        file_format = ext[1:] if ext else 'stl'

    if file_format == 'stl':
        return exporter.export_stl(mesh, path)
    elif file_format == 'obj':
        return exporter.export_obj(mesh, path)
    elif file_format == 'ply':
        return exporter.export_ply(mesh, path)
    else:
        print(f"Unknown format: {file_format}")
        return False
    