import shutil
from datetime import datetime, timezone
import uuid
import json
from pathlib import Path
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from frameExtraction import extract_frames
from generateRecontration import build_colmap_scene
from generateGS import run_gsplat_training
from fastapi import BackgroundTasks
from fastapi.responses import FileResponse

import modal

app = modal.App("gaussian-splat-api")
server = FastAPI(title="Gaussian Splat API")


volume = modal.Volume.from_name("gaussian-splat-jobs", create_if_missing=True)
VOLUME_PATH = Path("/gaussian-splat-jobs/jobs")
VOLUME_DATA = Path("/gaussian-splat-jobs")


image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install(
        "git",
        "ffmpeg",
        "build-essential",
        "python3-dev",
        "gcc",
        "g++",
        "ninja-build",
        "libglm-dev",
        "colmap",
    )
    .env({
        "PYTHONPATH": "/gsplat",
        "TORCH_EXTENSIONS_DIR": "/root/.cache/torch_extensions",
        "CUDA_HOME": "/usr/local/cuda",
        "CC": "gcc",
        "CXX": "g++",
        "CUDAHOSTCXX": "g++",
        "MAX_JOBS": "4",
        "TORCH_CUDA_ARCH_LIST": "8.6",
    })
    # Packaging basics first
    .run_commands(
        "python -m pip install --upgrade pip wheel 'setuptools<82'"
    )
    # Torch before anything that compiles against torch
    .pip_install(
        "torch",
        "torchvision",
        "torchaudio",
        index_url="https://download.pytorch.org/whl/cu128",
    )
    # Normal Python deps
    .pip_install("pycolmap")
    .pip_install_from_requirements("requirements.txt")
    # CUDA extension packages after torch + deps
    .run_commands(
        "python -m pip install --no-build-isolation 'git+https://github.com/rahul-goel/fused-ssim@328dc9836f513d00c4b5bc38fe30478b4435cbb5'",
        "python -m pip install --no-build-isolation 'git+https://github.com/harry7557558/fused-bilagrid@49f0ef06c9f81810fb9b5dd9027cf1844950cc16'",
        "python -m pip install --no-build-isolation 'ppisp @ git+https://github.com/nv-tlabs/ppisp@v1.0.0'",
    )
    # Copy repo before trying to import it
    .add_local_dir("gsplat", remote_path="/gsplat", copy=True)
    # Copy your helper modules
    .add_local_python_source(
        "frameExtraction",
        "generateRecontration",
        "generateGS",
        copy=True,
    )
    # Prebuild / verify gsplat only after repo is present
    .run_commands(
        "python -c \"import os, sys; print('PYTHONPATH=', os.environ.get('PYTHONPATH')); print('TORCH_EXTENSIONS_DIR=', os.environ.get('TORCH_EXTENSIONS_DIR')); import gsplat; print('Imported gsplat from', gsplat.__file__)\""
    )
)
# Update this to match where your gsplat repo lives on the server.
# Example:
# GSPLAT_REPO = Path(r"C:\apps\gsplat")
GSPLAT_REPO = Path("/gsplat")

# Ephemeral in-memory job store.
# The server does not persist job metadata beyond runtime, and after a client
# successfully collects a splat, the corresponding job folder and job entry
# are deleted.

def write_status(job_id: str, status: str, error: str | None = None, output_file: str | None = None):
    d = VOLUME_PATH / job_id
    d.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": status,
        "error": error,
        "output_file": output_file,
    }
    (d / "status.json").write_text(json.dumps(payload))

def read_status(job_id: str) -> dict:
    status_file = VOLUME_PATH / job_id / "status.json"
    if not status_file.exists():
        raise FileNotFoundError("Job not found")
    return json.loads(status_file.read_text())



def cleanup_job(job_id: str) -> None:
    shutil.rmtree(job_id, ignore_errors=True)


def find_output_file(result_dir: Path) -> Path:
    candidates = list(result_dir.rglob("*.splat")) + list(result_dir.rglob("*.ply"))
    if not candidates:
        raise FileNotFoundError("No generated .splat or .ply file was found.")
    return max(candidates, key=lambda p: p.stat().st_mtime)

@app.function(
    image=image,
    gpu="A10G",
    volumes={"/gaussian-splat-jobs": volume},
    timeout=60 * 60 * 2,
)
async def process_generation(job_id: str, video_path: str) -> None:
    video_path = Path(video_path)
    await volume.reload.aio()

    print("exists:", video_path.exists())
    print("size:", video_path.stat().st_size if video_path.exists() else -1)

    if not video_path.exists() or video_path.stat().st_size == 0:
        raise RuntimeError(f"Uploaded video missing or empty: {video_path}")
    job_dir = VOLUME_PATH / job_id
    frames_dir = job_dir / "frames"
    scene_dir = job_dir / "scene"
    result_dir = job_dir / "result"

    try:
        write_status(job_id, status="processing")

        frames_dir.mkdir(parents=True, exist_ok=True)
        scene_dir.mkdir(parents=True, exist_ok=True)
        result_dir.mkdir(parents=True, exist_ok=True)

        # 1) Extract images from the uploaded video
        print("exists:", Path(video_path).exists())
        print("size:", Path(video_path).stat().st_size if Path(video_path).exists() else -1)
        extract_frames(str(video_path), str(frames_dir))

        # 2) Build reconstruction with COLMAP via pycolmap
        build_colmap_scene(
            image_dir=str(frames_dir),
            scene_dir=str(scene_dir),
            use_sequential_matching=True,
        )

        # 3) Train / generate the Gaussian splat
        code = run_gsplat_training(
            gsplat_repo=str(GSPLAT_REPO),
            scene_dir=str(scene_dir),
            result_dir=str(result_dir),
            data_factor=1,
            max_steps=30000,
            save_ply=True,
            ply_steps=30000,
            disable_viewer=True
        )
        if code != 0:
            raise RuntimeError(f"gsplat training failed with exit code {code}")

        output_file = find_output_file(result_dir)
        write_status(job_id, status="completed", output_file=str(output_file))

    except Exception as exc:
        write_status(job_id, status="failed", error=str(exc))


@server.post("/generate")
async def generate(video: UploadFile = File(...)):
    await volume.reload.aio()

    if not video.filename:
        raise HTTPException(status_code=400, detail="A video file is required.")

    suffix = Path(video.filename).suffix.lower()
    allowed = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    if suffix and suffix not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported video type: {suffix}. Allowed: {sorted(allowed)}",
        )

    job_id = str(uuid.uuid4())
    job_dir = VOLUME_PATH / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    input_path = job_dir / f"input{suffix or '.mp4'}"

    await video.seek(0)
    contents = await video.read()

    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded video is empty.")

    with input_path.open("wb") as f:
        f.write(contents)

    size = input_path.stat().st_size
    print("saved upload:", input_path)
    print("saved size:", size)

    if size == 0:
        raise HTTPException(status_code=400, detail="Saved uploaded video is empty.")

    status = {
        "id": job_id,
        "status": "queued",
        "error": None,
        "output_file": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    (job_dir / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")

    await video.close()

    await volume.commit.aio()

    process_generation.spawn(job_id, str(input_path))

    return JSONResponse({
        "id": job_id,
        "status": "queued",
        "message": "Generation started. Poll GET /splat/{id} to retrieve the result.",
    })


async def cleanup_job_after_send (job_id: str):
    await volume.reload.aio()
    shutil.rmtree(VOLUME_PATH / job_id, ignore_errors=True)
    await volume.commit.aio()

@server.get("/splat/{job_id}")
async def get_splat(job_id: str, background_tasks: BackgroundTasks):
    await volume.reload.aio()

    status_path = VOLUME_PATH / job_id / "status.json"
    if not status_path.exists():
        raise HTTPException(status_code=404, detail="Job not found")

    status = json.loads(status_path.read_text(encoding="utf-8"))

    if status["status"] in {"queued", "processing"}:
        return {"id": job_id, "status": status["status"]}

    if status["status"] == "failed":
        return {"id": job_id, "status": "failed", "error": status.get("error")}

    if status["status"] != "completed":
        raise HTTPException(status_code=500, detail="Job is in an unknown state.")

    output_file = Path(status["output_file"])
    if not output_file.exists():
        raise HTTPException(status_code=404, detail="Output file missing")

    background_tasks.add_task(cleanup_job_after_send, job_id)

    return FileResponse(
        path=str(output_file),
        filename=output_file.name,
        media_type="application/octet-stream",
    )



@server.get("/")
def root():
    return {
        "name": "Gaussian Splat API",
        "endpoints": {
            "generate": "POST /generate",
            "get_splat": "GET /splat/{id}",
        },
    }



@app.function(image=image, volumes={"/gaussian-splat-jobs": volume})
@modal.asgi_app()
def fastapi_app():
    return server

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:server", host="0.0.0.0", port=8000, reload=True)
