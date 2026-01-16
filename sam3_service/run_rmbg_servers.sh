#!/usr/bin/env bash
# Start multiple RMBG HTTP servers (one process per port). Lightweight, can run more ports.
# Uses config/services.rmbg_endpoints unless RMBG_PORTS is provided.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${ROOT_DIR}/config/config.yaml"
LOG_DIR=${LOG_DIR:-"${ROOT_DIR}/sam3_service/logs"}
mkdir -p "${LOG_DIR}"

# Parse ports from env or config
if [[ -n "${RMBG_PORTS:-}" ]]; then
  PORT_LIST="${RMBG_PORTS}"
else
  PORT_LIST=$(CONFIG_PATH="${CONFIG_PATH}" python - <<'PY'
import yaml, os, sys
cfg_path = os.environ.get('CONFIG_PATH')
if not cfg_path or not os.path.exists(cfg_path):
    sys.exit("[ERROR] CONFIG_PATH missing or not found")
with open(cfg_path, 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)
ports = []
for ep in cfg.get('services', {}).get('rmbg_endpoints', []):
    ep = ep.strip()
    if ep.startswith('http://'):
        ep = ep[len('http://'):]
    if ':' in ep:
        ports.append(ep.split(':',1)[1])
print(' '.join(ports))
PY
)
fi

if [[ -z "${PORT_LIST}" ]]; then
  echo "[ERROR] No RMBG ports found. Set RMBG_PORTS or config.services.rmbg_endpoints" >&2
  exit 1
fi

# Parse GPUs from env or config; if empty, onnxruntime will pick available providers
if [[ -n "${RMBG_GPUS:-}" ]]; then
  GPU_LIST="${RMBG_GPUS}"
else
  GPU_LIST=$(CONFIG_PATH="${CONFIG_PATH}" python - <<'PY'
import yaml, os, sys
cfg_path = os.environ.get('CONFIG_PATH')
if not cfg_path or not os.path.exists(cfg_path):
    sys.exit("[ERROR] CONFIG_PATH missing or not found")
with open(cfg_path, 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)
gpus = cfg.get('services', {}).get('rmbg_gpus', []) or []
print(' '.join(str(g) for g in gpus))
PY
)
fi

read -ra PORT_ARR <<<"${PORT_LIST}"
read -ra GPU_ARR <<<"${GPU_LIST:-}"

if [[ -n "${GPU_LIST}" ]] && [[ ${#PORT_ARR[@]} -ne ${#GPU_ARR[@]} ]]; then
  echo "[ERROR] RMBG_PORTS 与 RMBG_GPUS 数量不一致" >&2
  exit 1
fi

echo "ROOT_DIR=${ROOT_DIR}"
echo "CONFIG_PATH=${CONFIG_PATH}"
echo "PORTS=${PORT_LIST}"
echo "GPUS=${GPU_LIST:-auto}" 
echo "LOG_DIR=${LOG_DIR}"

for idx in "${!PORT_ARR[@]}"; do
  port="${PORT_ARR[$idx]}"
  gpu="${GPU_ARR[$idx]:-}"
  log_file="${LOG_DIR}/rmbg_${port}.log"
  echo "Starting RMBG server on port ${port} (GPU=${gpu:-auto}) -> ${log_file}"
  if [[ -n "${gpu}" ]]; then
    CUDA_VISIBLE_DEVICES="${gpu}" nohup python -m sam3_service.rmbg_server \
      --host 0.0.0.0 \
      --port "${port}" \
      --config "${CONFIG_PATH}" \
      >"${log_file}" 2>&1 &
  else
    nohup python -m sam3_service.rmbg_server \
      --host 0.0.0.0 \
      --port "${port}" \
      --config "${CONFIG_PATH}" \
      >"${log_file}" 2>&1 &
  fi
  echo "  PID=$!"
done

echo "All RMBG servers launched. Use tail -f ${LOG_DIR}/rmbg_<port>.log to view logs."
