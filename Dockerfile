# =============================================================================
# LTX 2.5 Lite Gen — thin overlay on top of the production LTX25 backend image.
#
# Build:
#   docker build -t s3rfdev/ltx25-lite-gen:latest .
#
# Base (public on Docker Hub, ~9 GB compressed):
#   s3rfdev/ai-studio-backend:v7-ltx25
#   — python3.14-slim + ComfyUI @99073836 (all LTX nodes are core) + ffmpeg +
#     api/ service package (LTX25ComfyService / A2V / ComfyClient).
#     Model weights are NOT in the image: 7 files ~60 GB are downloaded by the
#     entrypoint on node start from the manifest (Lightricks repos are GATED —
#     HF_TOKEN with an accepted LTX-2.5 license is mandatory).
# =============================================================================
FROM s3rfdev/ai-studio-backend:v7-ltx25

USER root

# Web studio (FastAPI app + static UI)
COPY litegen /home/llm/s3rf/ai_studio_3/litegen

# Entrypoint: model download (manifest-driven, resumable) -> ComfyUI :8188 -> API :8000
COPY entrypoint_lite.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh \
    && mkdir -p /home/llm/s3rf/ai_studio_3/outputs

WORKDIR /home/llm/s3rf/ai_studio_3
EXPOSE 8000 8188

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=5 \
    CMD curl -fs http://localhost:8000/health || exit 1

ENTRYPOINT ["/usr/bin/tini", "--", "/entrypoint.sh"]
