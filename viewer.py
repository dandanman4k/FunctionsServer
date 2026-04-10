from pathlib import Path
import open3d as o3d


def view_gaussians(file_path: str) -> None:
    """
    Quick test viewer for a generated Gaussian PLY.

    Supports:
    - .ply point cloud files

    Example:
        view_gaussians("result_fastgs/point_cloud/iteration_3000/point_cloud.ply")
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if path.suffix.lower() != ".ply":
        raise ValueError(
            f"Unsupported file type: {path.suffix}. "
            "This test viewer currently expects a .ply file."
        )

    pcd = o3d.io.read_point_cloud(str(path))
    if pcd.is_empty():
        raise RuntimeError(f"Loaded point cloud is empty: {path}")

    print(f"Loaded: {path}")
    print(f"Points: {len(pcd.points)}")

    o3d.visualization.draw_geometries(
        [pcd],
        window_name="Gaussian Viewer",
        width=1280,
        height=720,
        point_show_normal=False,
    )


if __name__ == "__main__":
    # Replace this with your real file path
    view_gaussians("result.ply")