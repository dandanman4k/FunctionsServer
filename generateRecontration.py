from pathlib import Path
import shutil
import pycolmap
import subprocess


def build_colmap_scene(
    image_dir: str,
    scene_dir: str,
    use_sequential_matching: bool = True,
):
    image_dir = Path(image_dir)
    scene_dir = Path(scene_dir)
    db_path = scene_dir / "database.db"
    sparse_root = scene_dir / "sparse"
    sparse0 = sparse_root

    scene_dir.mkdir(parents=True, exist_ok=True)
    sparse_root.mkdir(parents=True, exist_ok=True)

    target_images = scene_dir / "images"
    if target_images.resolve() != image_dir.resolve():
        if target_images.exists():
            shutil.rmtree(target_images)
        shutil.copytree(image_dir, target_images)
        image_dir = target_images

    feat_opts = pycolmap.FeatureExtractionOptions()
    feat_opts.use_gpu = False
    feat_opts.sift.max_num_features = 8192
    feat_opts.sift.estimate_affine_shape = True
    feat_opts.sift.domain_size_pooling = True

    pycolmap.extract_features(
        database_path=db_path,
        image_path=image_dir,
        extraction_options=feat_opts,
    )

    if use_sequential_matching:
        pycolmap.match_sequential(database_path=db_path)
    else:
        pycolmap.match_exhaustive(database_path=db_path)

    maps = pycolmap.incremental_mapping(
        database_path=db_path,
        image_path=image_dir,
        output_path=sparse_root,
    )

    if not maps:
        raise RuntimeError("COLMAP failed to reconstruct anything.")

    best_idx, best_rec = max(
        maps.items(),
        key=lambda kv: len(kv[1].images)
    )

    if sparse0.exists():
        shutil.rmtree(sparse0)
    sparse0.mkdir(parents=True, exist_ok=True)
    best_rec.write(sparse0)

    print(f"Saved reconstruction {best_idx} to {sparse0}")
    print(best_rec.summary())

    # NEW: create undistorted dataset for FastGS
    undistorted_dir = scene_dir.parent / f"{scene_dir.name}_undistorted"
    if undistorted_dir.exists():
        shutil.rmtree(undistorted_dir)
    undistorted_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "colmap",
        "image_undistorter",
        "--image_path", str(scene_dir / "images"),
        "--input_path", str(sparse0),
        "--output_path", str(undistorted_dir),
        "--output_type", "COLMAP",
    ]
    print("Running undistorter:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)

    # Normalize undistorted sparse layout to sparse/0 for FastGS
    undist_sparse = undistorted_dir / "sparse"
    undist_sparse0 = undist_sparse / "0"

    if undist_sparse.exists() and not undist_sparse0.exists():
        model_names = [
            "cameras.bin", "images.bin", "points3D.bin",
            "cameras.txt", "images.txt", "points3D.txt",
        ]
        has_direct_model = any((undist_sparse / name).exists() for name in model_names)

        if has_direct_model:
            undist_sparse0.mkdir(parents=True, exist_ok=True)
            for name in model_names:
                src = undist_sparse / name
                dst = undist_sparse0 / name
                if src.exists():
                    shutil.move(str(src), str(dst))

    return {
        "reconstruction": best_rec,
        "scene_dir": scene_dir,
        "undistorted_dir": undistorted_dir,
    }