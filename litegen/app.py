"""
LTX 2.5 Lite Gen — standalone video-generation API (one container, one GPU).

Modes:
  POST /api/v1/generate/t2v      — prompt -> video (start-frame bypassed)
  POST /api/v1/generate/i2v      — image + prompt -> video
  POST /api/v1/generate/a2v      — image + speech audio -> talking video (lip sync)
  POST /api/v1/generate/lipdub   — alias of a2v with video_file input (frontend compat)
  POST /api/v1/generate/restore  — ready video -> x2 detailing/upscale pass,
                                   original audio track preserved

Engine: LTX-2.5-22b DISTILLED (nvfp4) through a headless ComfyUI (:8188), two-stage
pipeline (denoise at half res -> latent x2 upscaler -> IC-LoRA detailing pass),
native AAC audio via the audio VAE.

Sampler quality configs (ManualSigmas strings, N values = N-1 steps):
  fast     "6+2" — stage1 6 steps, stage2 2 steps. Trimmed config, verified on
                   24GB nodes (5090 Laptop): 1080p/97f ~ 76 s.
  official "8+4" — stage1 8 steps, stage2 4 sigma transitions. Official
                   Lightricks two-stage distilled schedule.

The quality switch patches the two ManualSigmas nodes (sg5516_4984 / sg5517_4985)
and the two KSamplerSelect nodes of the workflow templates at runtime.

Queue: in-memory (background=true -> {status:"queued", task_id}), dual-path task
routes for full-backend frontend compatibility. GET /api/build exposes what is
INSIDE this build (models, upscale availability, quality configs) so the web UI
can render capability badges dynamically.

Run:  uvicorn litegen.app:app --host :: --port 8000
      (host :: = dual-stack: Salad Container Gateway connects over IPv6)

LITEGEN_MOCK=1 — no GPU/ComfyUI needed: jobs return a generated test clip
(used for UI development and Hugging Face Space demos).
"""

import base64
import datetime
import json
import os
import shutil
import subprocess
import threading
import time
import uuid
import zlib

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

APP_DIR = os.path.dirname(os.path.abspath(__file__))
VERSION = "1.0.0"
MOCK = os.environ.get("LITEGEN_MOCK", "0") == "1"

# ── Engine imports (present in the docker base image v7-ltx25) ────────────────
ENGINE_AVAILABLE = False
settings = None
try:
    if not MOCK:
        from api.core.config import settings  # type: ignore
        from api.services.ltx25_comfy_service import LTX25ComfyService  # type: ignore
        from api.services.ltx25_comfy_a2v_service import LTX25ComfyA2VService  # type: ignore
        from api.services.comfy_client import ComfyClient  # type: ignore
        ENGINE_AVAILABLE = True
except Exception as _e:  # noqa: BLE001
    print(f"[litegen] engine imports unavailable ({_e}) — "
          f"UI/metadata will work, generation needs the docker base image", flush=True)

OUTPUT_DIR = os.environ.get("OUTPUT_DIR")
if not OUTPUT_DIR:
    OUTPUT_DIR = ("/home/llm/s3rf/ai_studio_3/outputs"
                  if os.path.isdir("/home/llm/s3rf/ai_studio_3") else
                  os.path.join(APP_DIR, "outputs"))
UPLOAD_DIR = os.path.join(OUTPUT_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

COMFY_PORT = int(getattr(settings, "LTX25_COMFY_PORT", 8188)) if settings else 8188
COMFY_URL = f"http://127.0.0.1:{COMFY_PORT}"

# Fixed model filenames (downloaded by the entrypoint from models manifest)
CONST_AUDIO_VAE = "ltx-2.5-audio-vae-bf16.safetensors"
CONST_LATENT_UPSCALER = "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"
CONST_IC_LORA = "ltx-2.5-22b-ic-lora-pixel-spatial-upscaler-x2-1.0.safetensors"

MODELS_INFO = [
    {"name": "ltx-2.5-22b-distilled-transformer-nvfp4", "role": "DiT (nvfp4)", "size_gb": 18.7},
    {"name": "gemma4-12b-heretic (nvfp4) / bf16 fallback", "role": "Text encoder", "size_gb": 9.9},
    {"name": "ltx-2.5-video-vae-conv-bf16", "role": "Video VAE", "size_gb": 1.5},
    {"name": "ltx-2.5-audio-vae-bf16", "role": "Audio VAE (native sound)", "size_gb": 2.0},
    {"name": "ltx-2.5-latent-spatial-upscaler-x2", "role": "Latent upscaler x2", "size_gb": 1.9},
    {"name": "ltx-2.5-22b-ic-lora-pixel-spatial-upscaler-x2", "role": "Detailing IC-LoRA", "size_gb": 0.33},
]

QUALITY_CONFIGS = {
    "fast": {
        "id": "fast", "label": "Fast 6+2",
        "desc": "Trimmed schedule: stage1 6 steps + stage2 2 steps. Verified on 24GB.",
        "stage1_sigmas": "1.0, 0.99375, 0.975, 0.909375, 0.725, 0.421875, 0.0",
        "stage1_sampler": "euler_ancestral",
        "stage2_sigmas": "0.909375, 0.725, 0.0",
        "stage2_sampler": "euler",
    },
    "official": {
        "id": "official", "label": "Official 8+4",
        "desc": "Official two-stage distilled schedule: stage1 8 steps + stage2 refinement.",
        "stage1_sigmas": "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0",
        "stage1_sampler": "euler_ancestral_cfg_pp",
        "stage2_sigmas": "0.85, 0.7250, 0.4219, 0.0",
        "stage2_sampler": "euler_cfg_pp",
    },
}

app = FastAPI(title="LTX 2.5 Lite Gen", version=VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=os.path.join(APP_DIR, "static")), name="static")
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")


# ═══════════════════════════ Engine services ═════════════════════════════════

if ENGINE_AVAILABLE:

    class _QualityMixin:
        """Per-generation sampler-config switch (set inside the gen lock)."""

        quality: str = "fast"
        t2v: bool = False

        def _apply_quality(self, api: dict, stage: int) -> None:
            cfg = QUALITY_CONFIGS[self.quality]
            if stage == 1:
                api["sg5516_4984"]["inputs"]["sigmas"] = cfg["stage1_sigmas"]
                api["sg5516_4831"]["inputs"]["sampler_name"] = cfg["stage1_sampler"]
            else:
                api["sg5517_4985"]["inputs"]["sigmas"] = cfg["stage2_sigmas"]
                api["sg5517_4976"]["inputs"]["sampler_name"] = cfg["stage2_sampler"]

    class LTX25QualityI2V(_QualityMixin, LTX25ComfyService):
        """I2V/T2V with runtime quality config (+ start-frame bypass for T2V)."""

        def _patch_t2v(self, api: dict) -> None:
            if self.t2v:
                # "Load Image (set bypass=True if t2v)" — official graph pattern:
                # the img2video conditioning is skipped, prompt drives motion.
                api["sg5514_3159"]["inputs"]["bypass"] = True
                api["sg5517_4970"]["inputs"]["bypass"] = True

        def _build_workflow(self, *a, **kw):
            api = super()._build_workflow(*a, **kw)
            self._apply_quality(api, 1)
            self._apply_quality(api, 2)
            self._patch_t2v(api)
            return api

        def _build_workflow_stage1(self, *a, **kw):
            api = super()._build_workflow_stage1(*a, **kw)
            self._apply_quality(api, 1)
            self._patch_t2v(api)
            return api

        def _build_workflow_stage2(self, *a, **kw):
            api = super()._build_workflow_stage2(*a, **kw)
            self._apply_quality(api, 2)
            api["sg5517_4970"]["inputs"]["bypass"] = self.t2v
            return api

    class LTX25QualityA2V(_QualityMixin, LTX25ComfyA2VService):
        """A2V (frozen-audio lip sync) with runtime quality config."""

        def _build_workflow(self, *a, **kw):
            api = super()._build_workflow(*a, **kw)
            self._apply_quality(api, 1)
            self._apply_quality(api, 2)
            return api

    _i2v = LTX25QualityI2V()
    _a2v = LTX25QualityA2V()

_GEN_LOCK = threading.Lock()  # one generation at a time — both engines share the GPU


def _dummy_png(path: str, wh: int = 64, gray: int = 128) -> str:
    """1x safe start-frame placeholder for T2V (bypassed by the graph anyway)."""
    raw = b"".join(b"\x00" + bytes([gray]) * (wh * 3) for _ in range(wh))

    def chunk(t: bytes, d: bytes) -> bytes:
        c = t + d
        return len(d).to_bytes(4, "big") + c + (zlib.crc32(c) & 0xFFFFFFFF).to_bytes(4, "big")

    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", wh.to_bytes(4, "big") + wh.to_bytes(4, "big")
                   + bytes([8, 2, 0, 0, 0]))
           + chunk(b"IDAT", zlib.compress(raw))
           + chunk(b"IEND", b""))
    with open(path, "wb") as fh:
        fh.write(png)
    return path


# ═══════════════════════════ Restore (x2 detailing) ══════════════════════════

def _ffprobe_video(path: str) -> dict:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,r_frame_rate,nb_frames:format=duration",
         "-of", "json", path],
        capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {r.stderr[-300:]}")
    meta = json.loads(r.stdout or "{}")
    st = (meta.get("streams") or [{}])[0]
    num, _, den = (st.get("r_frame_rate") or "24/1").partition("/")
    try:
        fps = float(num) / float(den or 1)
    except (ValueError, ZeroDivisionError):
        fps = 24.0
    has_audio = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
         "stream=index", "-of", "csv=p=0", path],
        capture_output=True, text=True).stdout.strip() != ""
    try:
        frames = int(st.get("nb_frames") or 0)
    except ValueError:
        frames = 0
    if frames <= 0:
        dur = float(meta.get("format", {}).get("duration") or 0)
        frames = max(1, int(dur * fps))
    return {"width": int(st.get("width") or 0), "height": int(st.get("height") or 0),
            "fps": fps, "frames": frames, "has_audio": has_audio}


def _snap8k1(n: int) -> int:
    return max(9, (int(n) - 1) // 8 * 8 + 1)


def _run_restore(payload: dict) -> str:
    """Upscale/detail a READY video: lanczos to target -> IC-LoRA x2 pixel pass
    (2 distilled steps) per 8k+1 chunk -> concat -> original audio muxed back.

    The standalone flavor of stage-2 (verified by docker/ltx_upscale_test.py in
    the parent project): same models as generation stage-2, so 'upscale' is
    always part of this build.
    """
    if not (ENGINE_AVAILABLE or MOCK):
        raise RuntimeError("Engine unavailable")
    src = payload["video_path"]
    meta = _ffprobe_video(src)
    fps = meta["fps"] or 24.0

    tw = int(payload.get("width") or 0)
    th = int(payload.get("height") or 0)
    if not tw or not th:  # "original x2"
        tw, th = meta["width"] * 2, meta["height"] * 2
    tw = max(256, min(3840, round(tw / 32) * 32))
    th = max(256, min(2176, round(th / 32) * 32))

    total = meta["frames"]
    chunk_cap = 49 if tw * th > 5_000_000 else 97
    parts = _i2v._part_lengths(total, chunk_cap)  # 8k+1 windows covering the clip
    if sum(parts) < total:
        print(f"[litegen-restore] tail trim: {total - sum(parts)} frame(s) "
              f"(<9, below the 8k+1 grid)", flush=True)

    ts = int(time.time())
    work = os.path.join(OUTPUT_DIR, f"restore_{ts}")
    os.makedirs(work, exist_ok=True)
    chunk_files = []
    try:
        client = ComfyClient(url=COMFY_URL)
        start = 0
        for i, n in enumerate(parts):
            chunk_src = os.path.join(work, f"src_{i}.mp4")
            r = subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error",
                 "-ss", f"{start / fps:.6f}", "-i", src,
                 "-frames:v", str(n), "-an",
                 "-c:v", "libx264", "-crf", "16", "-preset", "veryfast",
                 "-pix_fmt", "yuv420p", chunk_src],
                capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError(f"chunk extract failed: {r.stderr[-300:]}")
            actual = _ffprobe_video(chunk_src)["frames"]
            if actual < 9:
                break
            nframes = _snap8k1(actual)
            if nframes != actual:
                r = subprocess.run(
                    ["ffmpeg", "-y", "-loglevel", "error", "-i", chunk_src,
                     "-frames:v", str(nframes), "-an",
                     "-c:v", "libx264", "-crf", "16", "-preset", "veryfast",
                     "-pix_fmt", "yuv420p", chunk_src + ".cut.mp4"],
                    capture_output=True, text=True)
                if r.returncode == 0:
                    os.replace(chunk_src + ".cut.mp4", chunk_src)
                else:
                    nframes = _snap8k1(actual - (actual - _snap8k1(actual)))
            restored = os.path.join(work, f"up_{i}.mp4")
            _restore_one_chunk(client, chunk_src, restored, tw, th, nframes, fps,
                               payload.get("seed") or 43)
            chunk_files.append(restored)
            start += n
            print(f"[litegen-restore] chunk {len(chunk_files)}/{len(parts)} done",
                  flush=True)

        if not chunk_files:
            raise RuntimeError("no restorable frames (clip shorter than 9 frames?)")
        lst = os.path.join(work, "list.txt")
        with open(lst, "w") as fh:
            for p in chunk_files:
                fh.write(f"file '{p}'\n")
        silent = os.path.join(work, "video_only.mp4")
        r = subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
             "-i", lst, "-c", "copy", silent], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"concat failed: {r.stderr[-300:]}")

        out = payload["output_path"]
        if meta["has_audio"] and payload.get("keep_audio", True):
            r = subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-i", silent, "-i", src,
                 "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "copy",
                 "-c:a", "aac", "-b:a", "192k", "-shortest", out],
                capture_output=True, text=True)
        else:
            r = subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-i", silent,
                 "-c", "copy", out], capture_output=True, text=True)
        if r.returncode != 0 or not os.path.exists(out):
            raise RuntimeError(f"final mux failed: {r.stderr[-300:]}")
        return out
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _restore_one_chunk(client, src_mp4, out_mp4, tw, th, nframes, fps, seed):
    """One ComfyUI pass of the x2 pixel-detailing workflow on one chunk."""
    _ensure = None
    try:
        from api.services.z_image_service import _ensure_comfyui as _ensure  # noqa
    except Exception:
        pass
    if _ensure:
        _ensure()
    image_name = client.upload_file(src_mp4, "image")
    unet = getattr(settings, "LTX25_COMFY_UNET",
                   "ltx-2.5-22b-distilled-transformer-nvfp4.safetensors")
    video_vae = getattr(settings, "LTX25_COMFY_VIDEO_VAE",
                        "ltx-2.5-video-vae-conv-bf16.safetensors")
    clip = getattr(settings, "LTX25_COMFY_CLIP",
                   "gemma4-12b-with-proj-ltx-2.5-bf16.safetensors")
    wf = {
        "1": {"class_type": "LoadVideo", "inputs": {"file": image_name}},
        "2": {"class_type": "GetVideoComponents", "inputs": {"video": ["1", 0]}},
        "3": {"class_type": "ImageScale", "inputs": {
            "image": ["2", 0], "upscale_method": "lanczos",
            "width": tw, "height": th, "crop": "disabled"}},
        "4": {"class_type": "LTXVPreprocess", "inputs": {"image": ["3", 0], "img_compression": 18}},
        "5": {"class_type": "EmptyLTXVLatentVideo", "inputs": {
            "width": tw, "height": th, "length": nframes, "batch_size": 1}},
        "6": {"class_type": "LTXVImgToVideoInplace", "inputs": {
            "vae": ["30", 0], "image": ["4", 0], "latent": ["5", 0],
            "strength": 1.0, "bypass": False}},
        "7": {"class_type": "LTXVEmptyLatentAudio", "inputs": {
            "audio_vae": ["31", 0], "frames_number": nframes,
            "frame_rate": int(round(fps)), "batch_size": 1}},
        "8": {"class_type": "LTXVConcatAVLatent", "inputs": {
            "video_latent": ["6", 0], "audio_latent": ["7", 0]}},
        "9": {"class_type": "LatentUpscaleModelLoader", "inputs": {"model_name": CONST_LATENT_UPSCALER}},
        "10": {"class_type": "UNETLoader", "inputs": {"unet_name": unet, "weight_dtype": "default"}},
        "11": {"class_type": "LoraLoaderModelOnly", "inputs": {
            "lora_name": CONST_IC_LORA, "strength_model": 1.0, "model": ["10", 0]}},
        "12": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["32", 0], "text": ""}},
        "13": {"class_type": "LTXVConditioning", "inputs": {
            "positive": ["12", 0], "negative": ["12", 0], "frame_rate": float(fps)}},
        "14": {"class_type": "CFGGuider", "inputs": {
            "model": ["11", 0], "positive": ["13", 0], "negative": ["13", 1], "cfg": 1}},
        "15": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "16": {"class_type": "ManualSigmas", "inputs": {"sigmas": "0.909375, 0.725, 0.0"}},
        "17": {"class_type": "RandomNoise", "inputs": {"noise_seed": int(seed)}},
        "18": {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": ["17", 0], "guider": ["14", 0], "sampler": ["15", 0],
            "sigmas": ["16", 0], "latent_image": ["8", 0]}},
        "19": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["18", 0]}},
        "20": {"class_type": "VAEDecodeTiled", "inputs": {
            "samples": ["19", 0], "vae": ["30", 0], "tile_size": 512,
            "overlap": 64, "temporal_size": 128, "temporal_overlap": 32}},
        "21": {"class_type": "CreateVideo", "inputs": {
            "images": ["20", 0], "audio": ["2", 1], "fps": ["2", 2], "bit_depth": 8}},
        "22": {"class_type": "SaveVideo", "inputs": {
            "video": ["21", 0], "filename_prefix": "litegen_up", "format": "auto"}},
        "30": {"class_type": "VAELoader", "inputs": {"vae_name": video_vae}},
        "31": {"class_type": "VAELoader", "inputs": {"vae_name": CONST_AUDIO_VAE}},
        "32": {"class_type": "CLIPLoader", "inputs": {"clip_name": clip, "type": "ltxv", "device": "default"}},
    }
    pid = client.queue_prompt(wf)["prompt_id"]
    deadline = time.time() + (getattr(settings, "LTX25_COMFY_TIMEOUT_S", 3600) if settings else 3600)
    while time.time() < deadline:
        hist = client.get_history(pid)
        if hist and pid in hist:
            entry = hist[pid]
            status = entry.get("status", {})
            if status.get("status_str") == "error":
                raise RuntimeError("ComfyUI restore failed: "
                                   + json.dumps(status.get("details", []), ensure_ascii=False)[:500])
            for node_out in entry.get("outputs", {}).values():
                for lst_name in ("gifs", "videos", "images"):
                    for item in node_out.get(lst_name, []):
                        fn = item.get("filename", "")
                        if fn.endswith(".mp4"):
                            data = client.get_video(fn, item.get("subfolder", ""),
                                                    item.get("type", "output"))
                            with open(out_mp4, "wb") as fh:
                                fh.write(data)
                            return out_mp4
            if status.get("completed"):
                raise RuntimeError("restore finished but no mp4 in outputs")
        time.sleep(4)
    raise TimeoutError(f"restore chunk timed out ({pid})")


# ═══════════════════════════ Queue (in-memory) ═══════════════════════════════

_TASKS: dict = {}
_TASKS_LOCK = threading.Lock()


def _now() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _estimate_seconds(mode: str, width: int, height: int, frames: int, quality: str) -> int:
    """Rough wall-clock hint for the UI progress bar (24GB-class node, measured:
    1080p/97f fast = 76 s; 768p/97f fast = 35 s; 2K/97f fast = 144-215 s)."""
    mp = max(0.3, (width * height) / 1_000_000)
    spf = 0.78 * (mp / 2.07) ** 1.7
    if quality == "official":
        spf *= 1.45
    if mode == "restore":
        spf *= 0.35
    return max(25, int(spf * max(frames, 9)))


def _task_public(t: dict, extended: bool = False) -> dict:
    out = {
        "id": t["id"],
        "type": t["type"],
        "status": t["status"],
        "priority": t["priority"],
        "result_url": t["result_url"],
        "error_msg": t["error_msg"],
        "created_at": t["created_at"],
        "started_at": t["started_at"],
        "completed_at": t["completed_at"],
    }
    if extended:
        meta = t.get("meta", {})
        progress = None
        if t["status"] == "processing" and t.get("started_ts"):
            elapsed = time.time() - t["started_ts"]
            est = max(1, meta.get("expected_seconds", 120))
            progress = min(97, round(elapsed / est * 100))
        elif t["status"] == "completed":
            progress = 100
        out["progress"] = progress
        out["meta"] = meta
    return out


def _save_upload(up: UploadFile, ext: str) -> str:
    ts = int(time.time() * 1000)
    path = os.path.join(UPLOAD_DIR, f"up_{ts}{ext}")
    with open(path, "wb") as fh:
        shutil.copyfileobj(up.file, fh)
    return path


def _norm_video(width, height, frame_rate, resolution):
    """Resolution normalize + snap to /64 (two-stage: stage1 = final/2 must be /32).
    Returns (width, height, fps, frames_hint)."""
    fps = float(frame_rate or 24.0)
    if width and height:
        w = max(256, min(3840, round(int(width) / 64) * 64))
        h = max(256, min(2176, round(int(height) / 64) * 64))
        return w, h, fps
    presets = {
        "1080p": (1920, 1088), "2k": (2496, 1408), "4k": (3840, 2176),
        "720p": (1280, 704), "540p": (960, 544),
    }
    w, h = presets.get((resolution or "1080p").lower(), (1920, 1088))
    return w, h, fps


def _norm_seed(seed) -> "int | None":
    if seed is None or int(seed) < 0:
        return None
    return int(seed)


def _prepare_image(image_path: str, width: int, height: int) -> str:
    """Pad+blur the start frame to the target resolution (prod-identical,
    subprocess-isolated). Falls back to the raw upload if the helper is absent."""
    script = os.path.join(
        os.path.dirname(getattr(settings, "BASE_DIR", "") or "") or "/nonexistent",
        "prepare_image.py")
    candidates = [
        script,
        "/home/llm/s3rf/ai_studio_3/prepare_image.py",
        os.path.join(os.path.dirname(APP_DIR), "prepare_image.py"),
    ]
    python_bin = getattr(settings, "PYTHON_BIN", None) or "python3"
    base_dir = getattr(settings, "BASE_DIR", None) or os.path.dirname(OUTPUT_DIR)
    prepared = os.path.join(
        OUTPUT_DIR, f"prepared_{os.path.splitext(os.path.basename(image_path))[0]}.png")
    for cand in candidates:
        if not os.path.exists(cand):
            continue
        r = subprocess.run(
            [python_bin, cand, image_path, str(width), str(height), prepared],
            cwd=base_dir, capture_output=True, text=True, timeout=120)
        if r.returncode == 0 and os.path.exists(prepared):
            return prepared
    print("[litegen] prepare_image.py not found — using raw upload", flush=True)
    return image_path


def _run_i2v(payload: dict) -> str:
    width, height, fps = _norm_video(payload.get("width"), payload.get("height"),
                                     payload.get("frame_rate"), payload.get("resolution"))
    frames = payload.get("frames")
    if not frames:
        frames = _snap8k1((payload.get("duration_seconds") or 4.0) * fps)
    prompt = (payload.get("prompt") or "").strip() or "high quality video, cinematic"
    t2v = payload.get("mode") == "t2v"
    if t2v:
        dummy = os.path.join(OUTPUT_DIR, ".dummy_start.png")
        if not os.path.exists(dummy):
            _dummy_png(dummy)
        image = dummy
    else:
        image = _prepare_image(payload["image_path"], width, height)
    out = os.path.join(OUTPUT_DIR, f"output_litegen_{payload.get('mode')}_"
                                   f"{int(time.time())}.mp4")
    with _GEN_LOCK:
        _i2v.quality = payload.get("quality") or "fast"
        _i2v.t2v = t2v
        _i2v.generate_long(
            prompt=prompt, image_path=image, output_path=out,
            frames=int(frames), width=width, height=height, fps=fps,
            seed=_norm_seed(payload.get("seed")) or 42,
            image_strength=payload.get("image_strength") or 0.7)
    return out


def _run_a2v(payload: dict) -> str:
    width, height, fps = _norm_video(payload.get("width"), payload.get("height"),
                                     payload.get("frame_rate"), payload.get("resolution"))
    prompt = (payload.get("prompt") or "").strip() or "the person speaks"
    image = _prepare_image(payload["image_path"], width, height)
    out = os.path.join(OUTPUT_DIR, f"output_litegen_a2v_{int(time.time())}.mp4")
    with _GEN_LOCK:
        _a2v.quality = payload.get("quality") or "fast"
        _a2v.generate_long(
            prompt=prompt, image_path=image, audio_path=payload["audio_path"],
            output_path=out, frames=payload.get("frames"),
            width=width, height=height, fps=fps,
            seed=_norm_seed(payload.get("seed")) or 42,
            image_strength=payload.get("image_strength") or 0.7)
    return out


def _run_lipdub(payload: dict) -> str:
    """Full-backend lipdub shape (video_file) -> first frame + audio -> A2V."""
    video_path = payload["video_path"]
    ts = int(time.time())
    first_frame = os.path.join(OUTPUT_DIR, f"lipdub_first_frame_{ts}.png")
    r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", video_path,
                        "-vframes", "1", first_frame], capture_output=True)
    if r.returncode != 0 or not os.path.exists(first_frame):
        raise RuntimeError("could not extract first frame: "
                           + r.stderr.decode("utf-8", errors="ignore")[-300:])
    audio_path = os.path.join(OUTPUT_DIR, f"lipdub_audio_{ts}.wav")
    r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", video_path,
                        "-vn", "-c:a", "pcm_s16le", "-ac", "2", "-ar", "48000",
                        audio_path], capture_output=True)
    if r.returncode != 0 or not os.path.exists(audio_path):
        raise RuntimeError("could not extract audio (no audio track?): "
                           + r.stderr.decode("utf-8", errors="ignore")[-300:])
    a2v_payload = dict(payload)
    a2v_payload.pop("video_path", None)
    a2v_payload["image_path"] = first_frame
    a2v_payload["audio_path"] = audio_path
    a2v_payload.setdefault("prompt", "the person speaks")
    return _run_a2v(a2v_payload)


def _mock_run(payload: dict) -> str:
    mode = payload.get("mode", "i2v")
    out = os.path.join(OUTPUT_DIR, f"output_litegen_{mode}_{int(time.time())}.mp4")
    w, h, _ = _norm_video(payload.get("width"), payload.get("height"),
                          payload.get("frame_rate"), payload.get("resolution"))
    w, h = max(640, min(1280, w // 2)), max(360, min(720, h // 2))
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "lavfi", "-i", f"testsrc2=size={w}x{h}:rate=24:duration=3",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", out],
        capture_output=True, timeout=120)
    if not os.path.exists(out):
        raise RuntimeError("mock ffmpeg failed")
    time.sleep(4)  # let the UI progress bar breathe
    return out


def _execute(task_type: str, payload: dict) -> str:
    if MOCK:
        return _mock_run(payload)
    if not ENGINE_AVAILABLE:
        raise RuntimeError("Engine unavailable: this deployment has no ComfyUI/LTX "
                           "stack (use the docker image or set LITEGEN_MOCK=1)")
    if task_type == "t2v":
        return _run_i2v(payload)
    if task_type == "i2v":
        return _run_i2v(payload)
    if task_type == "lipdub":
        return _run_lipdub(payload)
    if task_type == "restore":
        return _run_restore(payload)
    return _run_a2v(payload)


def _worker_loop() -> None:
    while True:
        with _TASKS_LOCK:
            pending = [t for t in _TASKS.values() if t["status"] == "pending"]
            pending.sort(key=lambda t: (-t["priority"], t["created_ts"]))
            task = pending[0] if pending else None
            if task is not None:
                task["status"] = "processing"
                task["started_at"] = _now()
                task["started_ts"] = time.time()
        if task is None:
            time.sleep(0.5)
            continue
        try:
            out = _execute(task["type"], task["payload"])
            with _TASKS_LOCK:
                task["status"] = "completed"
                task["result_url"] = f"/outputs/{os.path.basename(out)}"
                task["completed_at"] = _now()
            print(f"[queue] Task completed: {task['id']} -> {task['result_url']}", flush=True)
        except Exception as e:  # noqa: BLE001
            with _TASKS_LOCK:
                task["status"] = "failed"
                task["error_msg"] = str(e)
                task["completed_at"] = _now()
            print(f"[queue] Task FAILED: {task['id']}: {e}", flush=True)


threading.Thread(target=_worker_loop, daemon=True, name="litegen-queue-worker").start()


# ═══════════════════════════ HTTP endpoints ══════════════════════════════════

class GenerationResponse(BaseModel):
    status: str
    message: str
    output_path: str
    output_url: str
    generation_time: float


def _queue_task(task_type: str, payload: dict) -> dict:
    task_id = str(uuid.uuid4())
    meta = {
        "mode": task_type,
        "quality": payload.get("quality"),
        "width": payload.get("width"), "height": payload.get("height"),
        "frames": payload.get("frames"),
        "fps": payload.get("frame_rate") or 24.0,
    }
    w, h, _ = _norm_video(payload.get("width"), payload.get("height"),
                          payload.get("frame_rate"), payload.get("resolution"))
    frames = payload.get("frames") or _snap8k1((payload.get("duration_seconds") or 4.0) * 24)
    meta["width"], meta["height"] = w, h
    meta["frames"] = int(frames)
    meta["expected_seconds"] = _estimate_seconds(
        task_type, w, h, int(frames), payload.get("quality") or "fast")
    with _TASKS_LOCK:
        _TASKS[task_id] = {
            "id": task_id, "type": task_type, "status": "pending", "priority": 0,
            "result_url": None, "error_msg": None, "created_at": _now(),
            "created_ts": time.time(), "started_at": None, "started_ts": None,
            "completed_at": None, "payload": payload, "meta": meta,
        }
    print(f"[queue] Task queued: {task_id} ({task_type}, "
          f"{meta['width']}x{meta['height']}, {meta['frames']}f, "
          f"quality={meta['quality']})", flush=True)
    return {"status": "queued", "task_id": task_id}


def _video_forms():
    return dict(
        prompt=Form(None, description="Motion/scene prompt"),
        width=Form(None, description="Output width (snapped to /64)"),
        height=Form(None, description="Output height"),
        resolution=Form(None, description="Preset fallback: 540p|720p|1080p|2k|4k"),
        frame_rate=Form(24.0, description="fps (24 recommended)"),
        frames=Form(None, description="Exact frame count (8k+1 enforced)"),
        duration_seconds=Form(None, description="Alternative to frames"),
        seed=Form(-1, description="-1 = random"),
        image_strength=Form(None, description="Start-frame conditioning strength"),
        quality=Form("fast", description="Sampler config: fast (6+2) | official (8+4)"),
        background=Form(False, description="True -> queue task, return task_id"),
    )


@app.get("/health")
@app.get("/api/v1/health")
def health():
    with _TASKS_LOCK:
        queued = sum(1 for t in _TASKS.values() if t["status"] == "pending")
        processing = sum(1 for t in _TASKS.values() if t["status"] == "processing")
    return {
        "status": "ok",
        "version": VERSION,
        "backend": ("mock" if MOCK else ("comfy" if ENGINE_AVAILABLE else "unavailable")),
        "modes": ["t2v", "i2v", "a2v", "restore"],
        "queue": {"pending": queued, "processing": processing},
    }


@app.get("/api/stats")
def stats():
    gpu = {"available": False}
    if MOCK:
        gpu = {"available": True, "name": "Mock GPU (demo)", "free_gb": 24.0, "total_gb": 24.0}
    else:
        try:
            import requests as _rq
            r = _rq.get(f"{COMFY_URL}/system_stats", timeout=4)
            dev = (r.json().get("devices") or [{}])[0]
            gpu = {
                "available": True,
                "name": dev.get("name", "GPU"),
                "free_gb": round((dev.get("vram_free") or 0) / 1024, 1),
                "total_gb": round((dev.get("vram_total") or 0) / 1024, 1),
            }
        except Exception:
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=name,memory.free,memory.total",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=6).stdout.strip()
                name, free, total = [x.strip() for x in out.split(",")]
                gpu = {"available": True, "name": name,
                       "free_gb": round(int(free) / 1024, 1),
                       "total_gb": round(int(total) / 1024, 1)}
            except Exception:
                pass
    with _TASKS_LOCK:
        counts = {"pending": 0, "processing": 0, "completed": 0, "failed": 0}
        for t in _TASKS.values():
            counts[t["status"]] = counts.get(t["status"], 0) + 1
    return {"status": "ok", "version": VERSION, "gpu": gpu,
            "queue_count": counts["pending"] + counts["processing"],
            "jobs": counts}


@app.get("/api/build")
def build_info():
    """What is inside THIS build — the UI renders its badges from this."""
    total_gb = 0.0
    if MOCK:
        total_gb = 24.0
    else:
        try:
            import requests as _rq
            dev = (_rq.get(f"{COMFY_URL}/system_stats", timeout=4).json()
                   .get("devices") or [{}])[0]
            total_gb = round((dev.get("vram_total") or 0) / 1024, 1)
        except Exception:
            pass
    if not total_gb:
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total",
                                  "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=6).stdout
            total_gb = int(out.strip().splitlines()[0]) / 1024
        except Exception:
            total_gb = 0.0
    gpu_class = "32gb" if total_gb >= 30 else ("24gb" if total_gb >= 22 else "unknown")
    auto_quality = "official" if gpu_class == "32gb" else "fast"
    return {
        "build": "ltx25-lite-gen",
        "version": VERSION,
        "backend": "mock" if MOCK else ("comfy" if ENGINE_AVAILABLE else "unavailable"),
        "model": "LTX-2.5 22B Distilled (NVFP4)",
        "pipeline": "two-stage distilled (half-res denoise -> latent x2 -> IC-LoRA detailing)",
        "modes": {"t2v": True, "i2v": True, "a2v": True, "restore": True},
        "upscale_included": True,
        "upscale_desc": ("Latent x2 upscaler + IC-LoRA pixel detailing in every generation; "
                         "standalone video restore up to 4K (original audio kept)"),
        "audio": "native AAC via LTX audio VAE (i2v generates sound; a2v/restore keep source track)",
        "quality_configs": [
            {"id": cid, "label": c["label"], "desc": c["desc"],
             "steps": "6+2" if cid == "fast" else "8+4"}
            for cid, c in QUALITY_CONFIGS.items()
        ],
        "auto_quality": auto_quality,
        "gpu_class": gpu_class,
        "vram_total_gb": total_gb,
        "caps": {
            "fps_recommended": 24,
            "frame_grid": "8k+1",
            "res_snap": 64,
            "max_width": 3840, "max_height": 2176,
            "frames_lowres": 601,   # <= 5MP (up to 2K)
            "frames_hires": 177,    # > 5MP (4K)
            "restore_chunk_lowres": 97, "restore_chunk_hires": 49,
            "notes": ("24GB: up to 2K (marginal, split-stages on) — 4K needs 32GB; "
                      "32GB: 4K up to 177 frames, 2K up to 601"),
        },
        "models": MODELS_INFO,
    }


@app.post("/api/v1/generate/i2v")
def generate_i2v(
    image_file: UploadFile = File(..., description="Start frame"),
    prompt: str = Form(None),
    width: int = Form(None), height: int = Form(None),
    resolution: str = Form(None),
    frame_rate: float = Form(24.0),
    frames: int = Form(None),
    duration_seconds: float = Form(None),
    seed: int = Form(-1),
    image_strength: float = Form(None),
    quality: str = Form("fast"),
    background: bool = Form(False),
):
    image_path = _save_upload(image_file, os.path.splitext(image_file.filename or "f.png")[1] or ".png")
    payload = dict(mode="i2v", prompt=prompt, width=width, height=height,
                   resolution=resolution, frame_rate=frame_rate, frames=frames,
                   duration_seconds=duration_seconds, seed=seed,
                   image_strength=image_strength, quality=quality,
                   image_path=image_path)
    if background:
        return _queue_task("i2v", payload)
    start_time = time.time()
    try:
        out = _execute("i2v", payload)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(e))
    return GenerationResponse(status="success", message="i2v generated",
                              output_path=out, output_url=f"/outputs/{os.path.basename(out)}",
                              generation_time=time.time() - start_time)


@app.post("/api/v1/generate/t2v")
def generate_t2v(
    prompt: str = Form(None),
    width: int = Form(None), height: int = Form(None),
    resolution: str = Form(None),
    frame_rate: float = Form(24.0),
    frames: int = Form(None),
    duration_seconds: float = Form(4.0),
    seed: int = Form(-1),
    quality: str = Form("fast"),
    background: bool = Form(False),
):
    payload = dict(mode="t2v", prompt=prompt, width=width, height=height,
                   resolution=resolution, frame_rate=frame_rate, frames=frames,
                   duration_seconds=duration_seconds, seed=seed, quality=quality)
    if background:
        return _queue_task("t2v", payload)
    start_time = time.time()
    try:
        out = _execute("t2v", payload)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(e))
    return GenerationResponse(status="success", message="t2v generated",
                              output_path=out, output_url=f"/outputs/{os.path.basename(out)}",
                              generation_time=time.time() - start_time)


@app.post("/api/v1/generate/a2v")
def generate_a2v(
    image_file: UploadFile = File(..., description="Start frame (talking person)"),
    audio_file: UploadFile = File(..., description="Speech audio (drives lip sync)"),
    prompt: str = Form(None),
    width: int = Form(None), height: int = Form(None),
    resolution: str = Form(None),
    frame_rate: float = Form(24.0),
    frames: int = Form(None, description="None -> video length = audio length (8k+1)"),
    seed: int = Form(-1),
    image_strength: float = Form(0.7),
    quality: str = Form("fast"),
    background: bool = Form(False),
):
    ext_img = os.path.splitext(image_file.filename or "f.png")[1] or ".png"
    ext_aud = os.path.splitext(audio_file.filename or "a.wav")[1] or ".wav"
    payload = dict(mode="a2v", prompt=prompt, width=width, height=height,
                   resolution=resolution, frame_rate=frame_rate, frames=frames,
                   seed=seed, image_strength=image_strength, quality=quality,
                   image_path=_save_upload(image_file, ext_img),
                   audio_path=_save_upload(audio_file, ext_aud))
    if background:
        return _queue_task("a2v", payload)
    start_time = time.time()
    try:
        out = _execute("a2v", payload)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(e))
    return GenerationResponse(status="success", message="a2v generated",
                              output_path=out, output_url=f"/outputs/{os.path.basename(out)}",
                              generation_time=time.time() - start_time)


def _lipdub_endpoint(
    video_file: UploadFile = File(..., description="Reference video with audio track"),
    prompt: str = Form("the person speaks"),
    width: int = Form(None), height: int = Form(None),
    resolution: str = Form(None),
    frames: int = Form(None),
    duration_seconds: float = Form(None),
    seed: int = Form(-1),
    frame_rate: float = Form(24.0),
    lora_strength: float = Form(1.0),
    conditioning_strength: float = Form(1.0),
    skip_stage_2: bool = Form(False),
    use_first_frame_a2v: bool = Form(False),
    use_comfyui: bool = Form(True),
    background: bool = Form(False),
):
    """Full-backend compatible lipdub alias (video_file in, A2V out)."""
    ext = os.path.splitext(video_file.filename or "v.mp4")[1] or ".mp4"
    payload = dict(mode="a2v", prompt=prompt, width=width, height=height,
                   resolution=resolution, frame_rate=frame_rate, frames=frames,
                   duration_seconds=duration_seconds, seed=seed, quality="fast",
                   video_path=_save_upload(video_file, ext),
                   use_first_frame_a2v=use_first_frame_a2v,
                   skip_stage_2=skip_stage_2)
    if background:
        return _queue_task("lipdub", payload)
    start_time = time.time()
    try:
        out = _execute("lipdub", payload)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(e))
    return GenerationResponse(status="success", message="LipDub video generated",
                              output_path=out, output_url=f"/outputs/{os.path.basename(out)}",
                              generation_time=time.time() - start_time)


@app.post("/api/v1/generate/lipdub")
def generate_lipdub(**kw):
    return _lipdub_endpoint(**kw)


@app.post("/api/v1/generate/restore")
def generate_restore(
    video_file: UploadFile = File(..., description="Source video to restore"),
    width: int = Form(None, description="Target width (None -> source x2)"),
    height: int = Form(None, description="Target height"),
    resolution: str = Form(None, description="2k|4k|... (width/height win)"),
    keep_audio: bool = Form(True, description="Mux original audio track back"),
    seed: int = Form(43),
    background: bool = Form(False),
):
    if width and height:
        w = max(256, min(3840, round(width / 32) * 32))
        h = max(256, min(2176, round(height / 32) * 32))
    elif (resolution or "").lower() == "4k":
        w, h = 3840, 2176
    elif (resolution or "").lower() == "2k":
        w, h = 2496, 1408
    else:
        w, h = 0, 0  # source x2
    payload = dict(mode="restore", video_path=None, width=w, height=h,
                   keep_audio=keep_audio, seed=seed)
    if background:
        video_path = _save_upload(video_file, os.path.splitext(video_file.filename or "v.mp4")[1] or ".mp4")
        payload["video_path"] = video_path
        return _queue_task("restore", payload)
    payload["video_path"] = _save_upload(video_file, os.path.splitext(video_file.filename or "v.mp4")[1] or ".mp4")
    start_time = time.time()
    try:
        out = _execute("restore", payload)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(e))
    return GenerationResponse(status="success", message="video restored",
                              output_path=out, output_url=f"/outputs/{os.path.basename(out)}",
                              generation_time=time.time() - start_time)


# ── Queue routes (dual-path for full-backend frontend compatibility) ──────────
@app.get("/api/v1/generate/queue/tasks")
@app.get("/api/v1/queue/tasks")
def get_all_tasks():
    with _TASKS_LOCK:
        tasks = [_task_public(t) for t in _TASKS.values()]
    tasks.sort(key=lambda t: (t["created_at"] or ""))
    return {"tasks": tasks}


@app.get("/api/v1/tasks/{task_id}")
@app.get("/api/v1/generate/queue/tasks/{task_id}")
@app.get("/api/v1/queue/tasks/{task_id}")
def get_task(task_id: str):
    with _TASKS_LOCK:
        t = _TASKS.get(task_id)
    if t is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return _task_public(t, extended=True)


@app.delete("/api/v1/generate/queue/tasks/{task_id}")
@app.delete("/api/v1/queue/tasks/{task_id}")
def delete_task(task_id: str):
    with _TASKS_LOCK:
        t = _TASKS.get(task_id)
        if t is None:
            raise HTTPException(status_code=404, detail="Task not found")
        if t["status"] == "processing":
            raise HTTPException(status_code=409, detail="Task is processing; cannot delete")
        del _TASKS[task_id]
    return {"status": "success", "message": f"Task {task_id} deleted"}


@app.post("/api/v1/generate/queue/tasks/clear_completed")
@app.post("/api/v1/queue/tasks/clear_completed")
def clear_completed_tasks():
    with _TASKS_LOCK:
        for tid in [k for k, v in _TASKS.items() if v["status"] in ("completed", "failed")]:
            del _TASKS[tid]
    return {"status": "success"}


# ── Gallery: JSON for the UI + legacy HTML table ─────────────────────────────
def _list_outputs() -> list:
    files = []
    if os.path.isdir(OUTPUT_DIR):
        for f in os.listdir(OUTPUT_DIR):
            if not f.endswith(".mp4") or f.startswith("prepared_"):
                continue
            path = os.path.join(OUTPUT_DIR, f)
            mode = "video"
            if f.startswith("output_litegen_"):
                try:
                    mode = f.split("_")[2]
                except IndexError:
                    pass
            files.append({
                "name": f,
                "url": f"/outputs/{f}",
                "mode": mode,
                "size_mb": round(os.path.getsize(path) / (1024 * 1024), 1),
                "mtime": datetime.datetime.fromtimestamp(os.path.getmtime(path))
                         .strftime("%Y-%m-%d %H:%M:%S"),
            })
    files.sort(key=lambda x: x["mtime"], reverse=True)
    return files


@app.get("/api/v1/outputs")
def outputs_json():
    return {"files": _list_outputs()}


@app.get("/api/download/{filename}")
def download_file(filename: str):
    if "/" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="bad filename")
    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="not found")
    return FileResponse(path, filename=filename, media_type="video/mp4")


@app.delete("/api/v1/outputs/{filename}")
def delete_output(filename: str):
    if "/" in filename or ".." in filename or not filename.endswith(".mp4"):
        raise HTTPException(status_code=400, detail="bad filename")
    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="not found")
    os.remove(path)
    return {"status": "success"}


@app.get("/gallery", response_class=HTMLResponse)
@app.get("/api/v1/gallery", response_class=HTMLResponse)
def gallery():
    rows = ""
    for f in _list_outputs():
        rows += (f'<tr><td><video controls preload="metadata" '
                 f'src="/outputs/{f["name"]}"></video></td>'
                 f'<td><a href="/outputs/{f["name"]}" target="_blank">{f["name"]}</a></td>'
                 f'<td>{f["mode"]}</td><td>{f["size_mb"]:.1f} MB</td>'
                 f'<td>{f["mtime"]}</td></tr>')
    return HTMLResponse(f"""
    <html><head><title>LTX 2.5 Lite Gen — gallery</title><style>
        body {{ font-family: sans-serif; margin: 40px; background: #121212; color: #eee; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #333; }}
        a {{ color: #7c8cff; text-decoration: none; }}
        video {{ max-width: 300px; border-radius: 8px; }}
    </style></head><body>
        <h1>LTX 2.5 Lite Gen — outputs ({'MOCK' if MOCK else 'comfy'})</h1>
        <table><tr><th>Preview</th><th>File</th><th>Mode</th><th>Size</th><th>Created</th></tr>
        {rows}</table></body></html>""")


@app.get("/", response_class=HTMLResponse)
def index():
    with open(os.path.join(APP_DIR, "static", "index.html"), encoding="utf-8") as fh:
        return HTMLResponse(fh.read())


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("litegen.app:app", host="::", port=int(os.environ.get("PORT", 8000)))
