from pathlib import Path
import shutil
import pycolmap

def build_colmap_scene(
    image_dir: str,
    scene_dir: str,
    use_sequential_matching: bool = True,
):
    image_dir = Path(image_dir)
    scene_dir = Path(scene_dir)
    db_path = scene_dir / "database.db"
    sparse_root = scene_dir / "sparse"
    sparse0 = sparse_root / "0"

    scene_dir.mkdir(parents=True, exist_ok=True)
    sparse_root.mkdir(parents=True, exist_ok=True)

    # Keep dataset self-contained for gsplat
    target_images = scene_dir / "images"
    if target_images.resolve() != image_dir.resolve():
        if target_images.exists():
            shutil.rmtree(target_images)
        shutil.copytree(image_dir, target_images)
        image_dir = target_images

    # 1) Extract features
    pycolmap.extract_features(
        database_path=db_path,
        image_path=image_dir,
        sift_options={
            "max_num_features": 8192,
            "estimate_affine_shape": True,
            "domain_size_pooling": True,
        },
    )

    # 2) Match features
    if use_sequential_matching:
        pycolmap.match_sequential(
            database_path=db_path,
            matching_options={
                "guided_matching": True,
            },
            sequential_options={
                "overlap": 10,
                "quadratic_overlap": True,
            },
        )
    else:
        pycolmap.match_exhaustive(
            database_path=db_path,
            matching_options={
                "guided_matching": True,
            },
        )

    # 3) Sparse reconstruction
    maps = pycolmap.incremental_mapping(
        database_path=db_path,
        image_path=image_dir,
        output_path=sparse_root,
        options={
            "filter_max_reproj_error": 2.0,
            "filter_min_tri_angle": 4.0,
            "ba_refine_focal_length": True,
            "ba_refine_principal_point": False,
            "ba_refine_extra_params": True,
        },
    )

    if not maps:
        raise RuntimeError("COLMAP failed to reconstruct anything.")

    # Pick the largest reconstruction
    best_idx, best_rec = max(
        maps.items(),
        key=lambda kv: len(kv[1].images)
    )

    # Ensure exactly scene/sparse/0 exists for gsplat
    if sparse0.exists():
        shutil.rmtree(sparse0)
    sparse0.mkdir(parents=True, exist_ok=True)
    best_rec.write(sparse0)

    print(f"Saved reconstruction {best_idx} to {sparse0}")
    print(best_rec.summary())

    return best_rec