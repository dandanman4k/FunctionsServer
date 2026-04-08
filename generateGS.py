import argparse
import subprocess
import sys
from pathlib import Path
import os



def validate_colmap_scene(scene_dir: Path) -> None:
    images_dir = scene_dir / "images"
    sparse0_dir = scene_dir / "sparse" / "0"

    required = [
        images_dir,
        sparse0_dir / "cameras.bin",
        sparse0_dir / "images.bin",
        sparse0_dir / "points3D.bin",
    ]

    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Scene folder is not gsplat-ready. Missing:\n" + "\n".join(missing)
        )


def find_simple_trainer(gsplat_repo: Path) -> Path:
    trainer = gsplat_repo / "examples" / "simple_trainer.py"
    if not trainer.exists():
        raise FileNotFoundError(f"Could not find simple_trainer.py at: {trainer}")
    return trainer


def run_gsplat_training(
    gsplat_repo: str,
    scene_dir: str,
    result_dir: str,
    data_factor: int = -1,
    max_steps: int = 30000,
    save_ply: bool = True,
    ply_steps: int | None = None,
    disable_viewer: bool = True,
    eval_steps: int = 1,
) -> int:
    gsplat_repo = Path(gsplat_repo).resolve()
    scene_dir = Path(scene_dir).resolve()
    result_dir = Path(result_dir).resolve()

    validate_colmap_scene(scene_dir)
    trainer = find_simple_trainer(gsplat_repo)
    result_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(trainer),
        "default",
        "--data_dir",
        str(scene_dir),
        "--result_dir",
        str(result_dir),
        "--data_factor",
        str(data_factor),
        "--max_steps",
        str(max_steps),
        "--eval_steps",
        str(eval_steps),
    ]

    if disable_viewer:
        cmd.append("--disable_viewer")

    if save_ply:
        cmd.append("--save_ply")

    if ply_steps is not None:
        cmd.extend(["--ply_steps", str(ply_steps)])

    print("Running gsplat training command:")
    print(" ".join(cmd))

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{gsplat_repo}:{env.get('PYTHONPATH', '')}"

    process = subprocess.run(
        cmd,
        cwd=str(gsplat_repo / "examples"),
        check=False,
        env=env,
    )

    return process.returncode


def main():
    parser = argparse.ArgumentParser(description="Train a Gaussian Splat with gsplat")
    parser.add_argument("--gsplat-repo", required=True, help="Path to your gsplat repo")
    parser.add_argument("--scene-dir", required=True, help="Path to COLMAP scene root")
    parser.add_argument("--result-dir", required=True, help="Where gsplat outputs go")
    parser.add_argument("--data-factor", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=30000)
    parser.add_argument("--ply-steps", type=int, default=100)
    parser.add_argument("--eval_steps", type=int, default=-1)
    parser.add_argument("--no-save-ply", action="store_true")
    parser.add_argument("--enable-viewer", action="store_true")
    args = parser.parse_args()

    code = run_gsplat_training(
        gsplat_repo=args.gsplat_repo,
        scene_dir=args.scene_dir,
        result_dir=args.result_dir,
        data_factor=args.data_factor,
        max_steps=args.max_steps,
        save_ply=not args.no_save_ply,
        eval_steps=not args.eval_steps,
        ply_steps=args.ply_steps,
        disable_viewer=not args.enable_viewer
    )

    if code != 0:
        raise SystemExit(code)

    print("\nTraining finished.")
    print(f"Check result folder: {args.result_dir}")
    print("Look for a .ply file under a ply/ folder if save_ply was enabled.")


if __name__ == "__main__":
    main()