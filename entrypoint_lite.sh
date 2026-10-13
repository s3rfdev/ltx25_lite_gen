#!/bin/bash
# =============================================================================
# LTX 2.5 Lite Gen entrypoint: models -> ComfyUI -> studio API
#   1) download the 7 model files (~60.2 GB) per the ltx25 manifest (curl -C -,
#      idempotent by Content-Length; marker stores the manifest md5 so manifest
#      changes re-trigger an incremental re-check). HF_TOKEN is REQUIRED on
#      Salad (Lightricks repos are gated — the token's user must have accepted
#      the LTX-2.5 license).
#   2) ComfyUI on :8188 (background, wait /system_stats)
#   3) litegen.app (FastAPI) on :$API_PORT (foreground, dual-stack ::)
# Env: MODEL_DOWNLOAD=0 to skip phase 1, API_PORT (default 8000).
# =============================================================================
set -e

COMFY_DIR="/home/llm/s3rf/comfy"
API_DIR="/home/llm/s3rf/ai_studio_3"
BUNDLE_ROOT="/home/llm/s3rf"
MARKER="$BUNDLE_ROOT/.models_ltx25_done"
MANIFEST="$API_DIR/docker/ltx25/models_ltx25.manifest"   # ships inside the base image
API_PORT="${API_PORT:-8000}"
COMFY_PORT="${COMFY_PORT:-8188}"

# Marker is only valid for the current manifest composition
if [ -f "$MARKER" ] && [ -f "$MANIFEST" ]; then
    if [ "$(md5sum "$MANIFEST" | cut -d' ' -f1)" != "$(cat "$MARKER")" ]; then
        echo "=== [litegen] Manifest changed — re-check models ==="
        rm -f "$MARKER"
    fi
fi

# ── 1. Models ────────────────────────────────────────────────────────────────
if [ "${MODEL_DOWNLOAD:-1}" = "1" ] && [ ! -f "$MARKER" ]; then
    echo "=== [litegen] Downloading models (before phase) ==="
    mkdir -p "$(dirname "$MARKER")"
    total=$(grep -cve '^\s*$' -e '^\s*#' "$MANIFEST" || echo 0)
    i=0
    while IFS='|' read -r dest phase url; do
        case "$dest" in ''|\#*) continue;; esac
        i=$((i+1))
        f="$BUNDLE_ROOT/$dest"
        [ "$phase" = "before" ] || continue
        if [ -s "$f" ]; then
            remote_len=$(curl -sIL -m 20 ${HF_TOKEN:+-H "Authorization: Bearer $HF_TOKEN"} "$url" | grep -i '^content-length' | tail -1 | tr -dc '0-9')
            local_len=$(stat -c %s "$f" 2>/dev/null || echo 0)
            if [ -n "$remote_len" ] && [ "$local_len" = "$remote_len" ]; then
                echo "[$i/$total] OK (cached): $dest"
                continue
            fi
        fi
        echo "[$i/$total] GET $dest"
        mkdir -p "$(dirname "$f")"
        ok=0
        for attempt in 1 2 3 4 5; do
            if curl -fL -C - --retry 3 --retry-delay 5 -m 14400 \
                ${HF_TOKEN:+-H "Authorization: Bearer $HF_TOKEN"} \
                -o "$f.part" "$url"; then
                mv "$f.part" "$f"
                ok=1
                break
            fi
            echo "  attempt $attempt failed, retrying in 15s..."; sleep 15
        done
        if [ "$ok" != "1" ]; then
            if curl -fL -C - ${HF_TOKEN:+-H "Authorization: Bearer $HF_TOKEN"} \
                -o "$f.part" "$url"; then
                mv "$f.part" "$f"
            else
                echo "FATAL: download failed: $url"; exit 1
            fi
        fi
    done < "$MANIFEST"
    md5sum "$MANIFEST" | cut -d' ' -f1 > "$MARKER"
    echo "=== [litegen] Models done (marker: $MARKER) ==="
else
    echo "=== [litegen] Model download skipped (MODEL_DOWNLOAD=0 or marker) ==="
fi

# ── 1.5 Swap (24GB nodes: long 2K decodes hit the ~31.5GB RAM ceiling) ───────
if ! swapon --show | grep -q .; then
    SWAPFILE=/home/llm/s3rf/swapfile
    if [ ! -f "$SWAPFILE" ]; then
        fallocate -l 16G "$SWAPFILE" 2>/dev/null || dd if=/dev/zero of="$SWAPFILE" bs=1M count=16384 status=none
        chmod 600 "$SWAPFILE"
        mkswap "$SWAPFILE" >/dev/null 2>&1
    fi
    if swapon "$SWAPFILE" 2>/dev/null; then
        echo "=== [litegen] Swap enabled: $(swapon --show --noheadings) ==="
    else
        echo "=== [litegen] WARNING: swapon failed (no privileges?) ==="
    fi
fi

# ── 2. ComfyUI (background) ──────────────────────────────────────────────────
export LD_LIBRARY_PATH="$(find /home/llm/s3rf/img_gen/venv/lib/python3.14/site-packages/nvidia -type d -name lib 2>/dev/null | paste -sd ':' -):${LD_LIBRARY_PATH}"
echo "=== [litegen] Starting ComfyUI on :$COMFY_PORT ==="
cd "$COMFY_DIR"
nohup venv/bin/python main.py --listen 0.0.0.0 --port "$COMFY_PORT" \
    >> /home/llm/s3rf/comfy_ltx25.log 2>&1 &
COMFY_PID=$!

for i in $(seq 1 120); do
    if curl -fs "http://127.0.0.1:$COMFY_PORT/system_stats" >/dev/null 2>&1; then
        echo "=== [litegen] ComfyUI UP (waited ${i}x2s) ==="
        break
    fi
    if ! kill -0 "$COMFY_PID" 2>/dev/null; then
        echo "=== [litegen] FATAL: ComfyUI died ==="; tail -50 /home/llm/s3rf/comfy_ltx25.log; exit 1
    fi
    sleep 2
done

# ── 3. Studio API (foreground) ───────────────────────────────────────────────
echo "=== [litegen] Starting LTX 2.5 Lite Gen API on :$API_PORT (dual-stack ::) ==="
cd "$API_DIR"
exec env PYTHONPATH="$API_DIR" \
    /home/llm/s3rf/img_gen/venv/bin/uvicorn litegen.app:app \
    --host "::" --port "$API_PORT"
