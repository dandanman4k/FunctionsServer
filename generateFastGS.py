from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Iterable


def _run(cmd: list[str], cwd: str | Path | None = None, env: dict | None = None) -> int:
    cmd = [str(x) for x in cmd]
    print("Running FastGS command:")
    print(" ".join(cmd))
    result = subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env, check=False)
    return result.returncode


def validate_fastgs_scene(scene_dir: str | Path) -> Path:
    """
    FastGS expects a COLMAP-style source directory passed with --source_path/-s.
    We validate the minimum structure your pipeline already creates.
    """
    scene_dir = Path(scene_dir)
    required = [
        scene_dir / "images",
        scene_dir / "sparse",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "FastGS scene is missing required paths:\n" + "\n".join(missing)
        )
    return scene_dir


def find_fastgs_artifact(result_dir: str | Path) -> Path:
    """
    FastGS uses a vanilla-3DGS-compatible representation, but the README excerpt
    does not spell out one exact output file path. So we search common locations.
    """
    result_dir = Path(result_dir)

    candidates: list[Path] = []

    # Common splat-ish formats
    for pattern in ("**/*.ksplat", "**/*.splat", "**/*.spz", "**/*.ply"):
        candidates.extend(result_dir.glob(pattern))

    # Prefer point-cloud-ish artifacts if present
    scored: list[tuple[int, float, Path]] = []
    for p in candidates:
        if not p.is_file():
            continue
        name = p.name.lower()
        score = 0
        if "point_cloud" in name:
            score += 20
        if "iteration" in str(p).lower():
            score += 10
        if p.suffix.lower() in {".ksplat", ".splat", ".spz"}:
            score += 15
        if p.suffix.lower() == ".ply":
            score += 5
        scored.append((score, p.stat().st_mtime, p))

    if not scored:
        raise FileNotFoundError(f"No FastGS output artifact found under {result_dir}")

    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return scored[0][2]


def run_fastgs_training(
    fastgs_repo: str | Path,
    scene_dir: str | Path,
    result_dir: str | Path,
    *,
    profile: str = "standard",   # "standard" or "big"
    resolution: int = 1,
    iterations: int = 15000,
    save_iterations: Iterable[int] = (3000, 15000),
    test_iterations: Iterable[int] = (),
    eval_split: bool = False,
    quiet: bool = False,
    extra_args: Iterable[str] = (),
) -> int:
    """
    FastGS replacement for your old generateGS wrapper.

    Important notes:
    - FastGS documents train.py with --source_path/-s and --model_path/-m.
    - It also documents --resolution/-r, --iterations, --save_iterations,
      and --test_iterations. The repo exposes train_base.sh and train_big.sh
      as presets, but train.py is the safest wrapper target. :contentReference[oaicite:2]{index=2}
    """
    fastgs_repo = Path(fastgs_repo).resolve()
    scene_dir = validate_fastgs_scene(scene_dir)
    result_dir = Path(result_dir).resolve()

    result_dir.mkdir(parents=True, exist_ok=True)

    train_py = fastgs_repo / "train.py"
    if not train_py.exists():
        raise FileNotFoundError(f"Could not find FastGS train.py at {train_py}")

    python_exe = shutil.which("python") or shutil.which("python3")
    if not python_exe:
        raise RuntimeError("Could not find a Python interpreter on PATH.")

    cmd = [
        python_exe,
        str(train_py),
        "-s", str(scene_dir),
        "-m", str(result_dir),
        "-r", str(resolution),
        "--iterations", str(iterations),
    ]

    if eval_split:
        cmd.append("--eval")

    if quiet:
        cmd.append("--quiet")

    save_iterations = list(save_iterations)
    if save_iterations:
        cmd.append("--save_iterations")
        cmd.extend(str(x) for x in save_iterations)

    test_iterations = list(test_iterations)
    if test_iterations:
        cmd.append("--test_iterations")
        cmd.extend(str(x) for x in test_iterations)

    # FastGS-specific quality/speed nudge.
    # The README exposes "standard" and "big" presets via shell scripts;
    # here we express that as a small train.py override set. :contentReference[oaicite:3]{index=3}
    if profile == "big":
        cmd.extend([
            "--loss_thresh", "0.01",
            "--grad_abs_thresh", "0.0008",
            "--grad_thresh", "0.0002",
        ])
    elif profile != "standard":
        raise ValueError("profile must be 'standard' or 'big'")

    cmd.extend(str(x) for x in extra_args)

    env = os.environ.copy()
    return _run(cmd, cwd=fastgs_repo, env=env)


if __name__ == "__main__":
    # Minimal manual smoke test
    repo = "/FastGS"
    scene = "/data/jobs/example/scene"
    out = "/data/jobs/example/result_fastgs"

    code = run_fastgs_training(
        repo,
        scene,
        out,
        profile="standard",
        resolution=1,
        iterations=3000,
        save_iterations=(1000, 3000),
        test_iterations=(),
        eval_split=False,
        quiet=False,
    )
    print("FastGS exit code:", code)

    if code == 0:
        try:
            artifact = find_fastgs_artifact(out)
            print("Found artifact:", artifact)
        except Exception as e:
            print("Training succeeded but artifact search failed:", e)