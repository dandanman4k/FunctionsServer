from pathlib import Path
import subprocess

def extract_frames(video_path: str, output_dir: str, fps: float = 1.0):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    out_pattern = output_dir / "frame_%04d.jpg"

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(video_path),
        "-vf", f"fps={fps}",
        str(out_pattern),
    ]

    print("Running:", " ".join(cmd))

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )

    print("FFMPEG STDOUT:\n", result.stdout)
    print("FFMPEG STDERR:\n", result.stderr)

    if result.returncode != 0:
        raise RuntimeError(
            f"FFmpeg failed with exit code {result.returncode}\n"
            f"STDERR:\n{result.stderr}"
        )

    saved = len(list(output_dir.glob("frame_*.jpg")))
    print(f"Saved {saved} frames to {output_dir}")

    if saved == 0:
        raise RuntimeError("FFmpeg extracted 0 frames from the video.")