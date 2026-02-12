import sys
import argparse
import torch
from pathlib import Path
from PIL import Image
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.model import OccupancyNetwork
from src.data.preprocessing import ImagePreprocessor
from src.mesh import extract_mesh, save_mesh


def parse_args():
    parser = argparse.ArgumentParser(description='Generate 3D mesh from image')
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--input', type=str, required=True,help='Input image path')
    parser.add_argument('--output', type=str, required=True, help='Output STL path')
    parser.add_argument('--resolution', type=int, default=64, help='Grid resolution (default: 64)')
    parser.add_argument('--threshold', type=float, default=0.5, help='Occupancy threshold (default: 0.5)')

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 50)
    print("3D Scanner Inference")
    print("=" * 50)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Resolution: {args.resolution}")
    print(f"Threshold: {args.threshold}")
    print("=" * 50)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print("Loading model...")
    model = OccupancyNetwork.from_checkpoint(args.checkpoint)
    model = model.to(device)
    model.eval()

    print("Processing image...")
    preprocessor = ImagePreprocessor()
    image = Image.open(args.input).convert('RGB')
    image_tensor = preprocessor(image).unsqueeze(0).to(device)

    print("Generating occupancy grid...")
    occupancy_grid = model.generate_occupancy_grid(
        image_tensor,
        resolution=args.resolution
    )

    print(f"Grid shape: {occupancy_grid.shape}")
    print(f"Occupancy range: [{occupancy_grid.min():.3f}, {occupancy_grid.max():.3f}]")

    print("Extracting mesh...")
    mesh = extract_mesh(occupancy_grid, threshold=args.threshold)

    if mesh is None:
        print("Error: Could not extract mesh!")
        print("Try adjusting the threshold (--threshold 0.3)")
        return

    print(f"Mesh vertices: {len(mesh.vertices)}")
    print(f"Mesh faces: {len(mesh.faces)}")
    print(f"Watertight: {mesh.is_watertight}")

    print("Saving STL...")
    save_mesh(mesh, args.output)

    print("=" * 50)
    print("Done!")
    print(f"Output saved to: {args.output}")


if __name__ == '__main__':
    main()