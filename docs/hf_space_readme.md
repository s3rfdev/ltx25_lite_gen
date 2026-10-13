---
title: LTX 2.5 Lite Gen
emoji: 🎬
colorFrom: indigo
colorTo: purple
sdk: static
pinned: false
license: apache-2.0
short_description: LTX-2.5 distilled video studio — T2V/I2V/A2V + ×2 restore (bring your own GPU)
---

# 🎬 LTX 2.5 Lite Gen — Hugging Face Space (frontend)

This Space is the **decoupled web client** only — it runs zero models and costs
nothing. The diffusion stack (LTX-2.5 22B Distilled NVFP4 + ComfyUI, ~60 GB of
weights) runs on **your own GPU node**.

## Two ways to use it

**1. Demo mode** — click **“Try demo mode”** in the banner. Jobs are simulated
and a sample clip is shown. No GPU needed.

**2. Live mode** — deploy the backend, then paste its URL into **⚙ Endpoint**:

```bash
# SaladCloud (recommended: rtx_5090 32GB for 4K, 24GB cards handle up to 2K)
python ctl.py salad up --gpu rtx_5090

# or any Docker host with an NVIDIA GPU
docker run --gpus all -p 8000:8000 -e HF_TOKEN=hf_xxx s3rfdev/ltx25-lite-gen:latest
```

When the node is up (first boot downloads ~60 GB of gated weights — an
`HF_TOKEN` with the accepted LTX-2.5 license is required), paste
`http://<node-ip>:8000` into the ⚙ Endpoint field and generate for real:
Text→Video, Image→Video, Audio→Video (lip sync) and the ×2 video Restore.

The backend code lives at
[github.com/s3rfdev/ltx25_lite_gen](https://github.com/s3rfdev/ltx25_lite_gen).
