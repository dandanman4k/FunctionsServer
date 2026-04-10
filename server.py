import shutil
from datetime import datetime, timezone
import uuid
import json
from pathlib import Path
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from frameExtraction import extract_frames
from generateRecontration import build_colmap_scene
from fastapi import BackgroundTasks
from fastapi.responses import FileResponse
from generateFastGS import run_fastgs_training, find_fastgs_artifact

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
        "CUDA_HOME": "/usr/local/cuda",
        "CC": "gcc",
        "CXX": "g++",
        "CUDAHOSTCXX": "g++",
        "MAX_JOBS": "4",
        "TORCH_CUDA_ARCH_LIST": "8.6",
        "PYTHONUNBUFFERED": "1",
    })
    .run_commands(
        "python -m pip install --upgrade pip wheel 'setuptools<82'"
    )
    .pip_install(
        "torch",
        "torchvision",
        "torchaudio",
        index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install_from_requirements("requirements_fastgs.txt")
    .add_local_dir("FastGS", remote_path="/fastgs", copy=True)
    .run_commands(
        "python -m pip install --no-build-isolation 'git+https://gitlab.inria.fr/bkerbl/simple-knn.git'",
        "cd /fastgs/submodules/fused-ssim && python -m pip install --no-build-isolation .",
        "cd /fastgs/submodules/diff-gaussian-rasterization_fastgs && python -m pip install --no-build-isolation .",
    )
    .add_local_python_source(
        "frameExtraction",
        "generateRecontration",
        "generateFastGS",
        copy=True,
    )
)
@app.function(
    image=image,
    gpu="A10G",
    volumes={VOLUME_DATA: volume},
    timeout=60 * 60 * 6,
)
async def process_generation_fastgs(job_id: str, video_path: str) -> None:
    await volume.commit.aio()

    video_path = Path(video_path)
    if not video_path.exists() or video_path.stat().st_size == 0:
        raise RuntimeError(f"Uploaded video missing or empty: {video_path}")

    job_dir = VOLUME_PATH / job_id
    frames_dir = job_dir / "frames"
    scene_dir = job_dir / "scene"
    result_dir = job_dir / "result_fastgs"

    def write_status(status: str, error: str | None = None, output_file: str | None = None):
        payload = {
            "id": job_id,
            "status": status,
            "error": error,
            "output_file": output_file,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "status.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    try:
        write_status("processing")
        await volume.commit.aio()

        frames_dir.mkdir(parents=True, exist_ok=True)
        scene_dir.mkdir(parents=True, exist_ok=True)
        result_dir.mkdir(parents=True, exist_ok=True)

        extract_frames(str(video_path), str(frames_dir), fps=8)

        recon = build_colmap_scene(
            image_dir=str(frames_dir),
            scene_dir=str(scene_dir),
            use_sequential_matching=True,
        )

        fastgs_scene_dir = recon["undistorted_dir"]

        code = run_fastgs_training(
            fastgs_repo="/fastgs",
            scene_dir=str(fastgs_scene_dir),
            result_dir=str(result_dir),
            profile="big",
            resolution=1,
            iterations=50000,
            save_iterations=(),
            test_iterations=(),
            eval_split=False,
            quiet=False,
        )
        if code != 0:
            raise RuntimeError(f"FastGS training failed with exit code {code}")

        artifact = find_fastgs_artifact(result_dir)
        write_status("completed", output_file=str(artifact))
        await volume.commit.aio()

    except Exception as exc:
        write_status("failed", error=str(exc))
        await volume.commit.aio()
        raise


@server.post("/generate")
async def generate(video: UploadFile = File(...)):
    await volume.commit.aio()

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

    saved_size = input_path.stat().st_size
    if saved_size == 0:
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

    await process_generation_fastgs.spawn.aio(job_id, str(input_path))

    return JSONResponse({
        "id": job_id,
        "status": "queued",
        "message": "FastGS generation started. Poll GET /artifact/{id} or your status endpoint.",
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
