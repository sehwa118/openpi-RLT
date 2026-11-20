#!/usr/bin/env bash
set -euo pipefail

if [[ "$(id -u)" -eq 0 ]]; then
  echo "Refusing to run as root; use your personal account (ssh gpu)." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

LATENT_ROOT=""
NORMALIZATION=""
OUT_ROOT=""
CACHE_ROOT=""
PYTHON_BIN="python"
DEVICE="0"
SEEDS="0 1 2"
VARIANTS="mlp resmlp part_attention_full part_attention_kinematic part_attention_hierarchical"
WANDB_MODE="disabled"
REP_MODEL="trlt_cross_only"
MAX_STEPS=""
RESUME=0
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: scripts/run_attention_actor_stage_b.sh [OPTIONS]

Run Stage B frozen-representation training, evaluation, and paired summaries on a shared GPU node.

Required:
  --latent-root PATH     Parent directory containing seed<S>/{train,validation,test} LatentDataset caches
  --normalization PATH   Path to normalization.json matching the frozen latent caches
  --out-root PATH        Writable output root (use /workspace/users/<user>/..., never /workspace directly)

Optional:
  --cache-root PATH      Cache root for UV/PIP/HF/XDG/WANDB (default: <out-root>/.cache)
  --python BIN           Python executable (default: python)
  --device IDX           Single GPU index passed to --device (default: 0)
  --seeds "0 1 2"        Space-separated seeds to run (default: "0 1 2")
  --variants "..."       Space-separated Stage B variants to run
                         (default: "mlp resmlp part_attention_full part_attention_kinematic part_attention_hierarchical")
  --rep-model NAME       Stage A representation model name in LatentDataset provenance (default: trlt_cross_only)
  --wandb MODE           disabled | offline | online (default: disabled)
  --max-steps N          Optional step cap forwarded to train-probe
  --resume               Forward --resume only to variants that already have a last/metadata.json checkpoint
  --dry-run              Run trex_offline.py train-probe --dry-run and print evaluate/summarize commands
  -h, --help             Show this help message
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --latent-root)
      LATENT_ROOT="${2:-}"
      shift 2
      ;;
    --normalization)
      NORMALIZATION="${2:-}"
      shift 2
      ;;
    --out-root)
      OUT_ROOT="${2:-}"
      shift 2
      ;;
    --cache-root)
      CACHE_ROOT="${2:-}"
      shift 2
      ;;
    --python)
      PYTHON_BIN="${2:-}"
      shift 2
      ;;
    --device)
      DEVICE="${2:-}"
      shift 2
      ;;
    --seeds)
      SEEDS="${2:-}"
      shift 2
      ;;
    --variants)
      VARIANTS="${2:-}"
      shift 2
      ;;
    --rep-model)
      REP_MODEL="${2:-}"
      shift 2
      ;;
    --wandb)
      WANDB_MODE="${2:-}"
      shift 2
      ;;
    --max-steps)
      MAX_STEPS="${2:-}"
      shift 2
      ;;
    --resume)
      RESUME=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "${LATENT_ROOT}" || -z "${NORMALIZATION}" || -z "${OUT_ROOT}" ]]; then
  echo "Error: --latent-root, --normalization, and --out-root are required." >&2
  usage >&2
  exit 1
fi

if [[ -z "${REP_MODEL}" ]]; then
  echo "Error: --rep-model must be non-empty." >&2
  exit 1
fi

case "${WANDB_MODE}" in
  disabled|offline|online)
    ;;
  *)
    echo "Error: --wandb must be disabled, offline, or online (got '${WANDB_MODE}')." >&2
    exit 1
    ;;
esac

if [[ -z "${DEVICE}" || "${DEVICE}" == "cpu" || "${DEVICE}" == *","* ]]; then
  echo "Error: --device must specify one GPU index (got '${DEVICE}')." >&2
  exit 1
fi

CURRENT_USER="$(id -un)"

validate_writable_path() {
  local label="$1"
  local raw_path="$2"
  local clean="${raw_path%/}"
  clean="${clean:-/}"
  if [[ "${clean}" =~ (^|/)\.\.(/|$) ]]; then
    echo "Error: ${label} must not contain '..' path segments (got '${raw_path}')." >&2
    exit 1
  fi
  if [[ "${clean}" == "/workspace" || "${clean}" == /workspace/* ]]; then
    case "${clean}" in
      /workspace/users/"${CURRENT_USER}"/*)
        ;;
      /workspace/teams/*/*)
        if [[ "${clean}" == /workspace/teams//* ]]; then
          echo "Error: invalid ${label} under /workspace/teams (got '${raw_path}')." >&2
          exit 1
        fi
        ;;
      *)
        echo "Error: ${label} under /workspace must live inside /workspace/users/${CURRENT_USER}/<dir> or /workspace/teams/<team>/<dir> (got '${raw_path}')." >&2
        exit 1
        ;;
    esac
  elif [[ "${clean}" == "/home" || "${clean}" == /home/* ]]; then
    echo "Warning: ${label} '${raw_path}' is under /home, which is ephemeral on container recreation; prefer /workspace/users/${CURRENT_USER}/..." >&2
  fi
}

validate_writable_path "--out-root" "${OUT_ROOT}"
OUT_CLEAN="${OUT_ROOT%/}"
OUT_CLEAN="${OUT_CLEAN:-/}"

if [[ -z "${CACHE_ROOT}" ]]; then
  CACHE_ROOT="${OUT_CLEAN}/.cache"
fi
validate_writable_path "--cache-root" "${CACHE_ROOT}"
CACHE_CLEAN="${CACHE_ROOT%/}"
CACHE_CLEAN="${CACHE_CLEAN:-/}"

config_for_variant() {
  case "$1" in
    mlp)
      echo "${REPO_ROOT}/configs/trlt/attention_actor/probe_mlp.json"
      ;;
    resmlp)
      echo "${REPO_ROOT}/configs/trlt/attention_actor/probe_resmlp.json"
      ;;
    part_attention_full)
      echo "${REPO_ROOT}/configs/trlt/attention_actor/probe_part_attn_full.json"
      ;;
    part_attention_kinematic)
      echo "${REPO_ROOT}/configs/trlt/attention_actor/probe_part_attn_kinematic.json"
      ;;
    part_attention_hierarchical)
      echo "${REPO_ROOT}/configs/trlt/attention_actor/probe_part_attn_hierarchical.json"
      ;;
    *)
      echo "Error: unknown Stage B variant '$1'." >&2
      return 1
      ;;
  esac
}

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
export UV_CACHE_DIR="${CACHE_CLEAN}/uv"
export PIP_CACHE_DIR="${CACHE_CLEAN}/pip"
export HF_HOME="${CACHE_CLEAN}/hf"
export XDG_CACHE_HOME="${CACHE_CLEAN}/xdg"
export WANDB_DIR="${CACHE_CLEAN}/wandb"

if [[ "${DRY_RUN}" -eq 0 ]]; then
  mkdir -p "${CACHE_CLEAN}" "${OUT_CLEAN}/checkpoints" "${OUT_CLEAN}/logs"
fi

read -r -a SEED_LIST <<< "${SEEDS}"
read -r -a VARIANT_LIST <<< "${VARIANTS}"

HAS_MLP=0
HAS_RESMLP=0
EVAL_METRICS=()

for variant in "${VARIANT_LIST[@]}"; do
  config_for_variant "${variant}" >/dev/null
  if [[ "${variant}" == "mlp" ]]; then
    HAS_MLP=1
  elif [[ "${variant}" == "resmlp" ]]; then
    HAS_RESMLP=1
  fi
done

for seed in "${SEED_LIST[@]}"; do
  TRAIN_LATENTS="${LATENT_ROOT%/}/seed${seed}/train"
  VAL_LATENTS="${LATENT_ROOT%/}/seed${seed}/validation"
  TEST_LATENTS="${LATENT_ROOT%/}/seed${seed}/test"

  for variant in "${VARIANT_LIST[@]}"; do
    CFG="$(config_for_variant "${variant}")"
    CKPT_DIR="${OUT_CLEAN}/checkpoints/seed${seed}/probe_${variant}"
    LOG_DIR="${OUT_CLEAN}/logs/seed${seed}/probe_${variant}"
    EVAL_DIR="${OUT_CLEAN}/logs/seed${seed}/eval_${variant}"

    TRAIN_CMD=(
      "${PYTHON_BIN}" "${REPO_ROOT}/scripts/trex_offline.py" train-probe
      --config "${CFG}"
      --device "${DEVICE}"
      --seed "${seed}"
      --train-latents "${TRAIN_LATENTS}"
      --validation-latents "${VAL_LATENTS}"
      --output-dir "${CKPT_DIR}"
      --log-dir "${LOG_DIR}"
      --wandb "${WANDB_MODE}"
    )
    if [[ -n "${MAX_STEPS}" ]]; then
      TRAIN_CMD+=(--max-steps "${MAX_STEPS}")
    fi
    if [[ "${RESUME}" -eq 1 && -f "${CKPT_DIR}/last/metadata.json" ]]; then
      TRAIN_CMD+=(--resume)
    fi

    EVAL_CMD=(
      "${PYTHON_BIN}" "${REPO_ROOT}/scripts/trex_offline.py" evaluate
      --config "${CFG}"
      --device "${DEVICE}"
      --seed "${seed}"
      --latents "${TEST_LATENTS}"
      --probe-checkpoint "${CKPT_DIR}/best"
      --normalization "${NORMALIZATION}"
      --output-dir "${EVAL_DIR}"
    )

    if [[ "${DRY_RUN}" -eq 1 ]]; then
      "${TRAIN_CMD[@]}" --dry-run
      printf '[dry-run evaluate]'
      printf ' %q' "${EVAL_CMD[@]}"
      printf '\n'
    else
      "${TRAIN_CMD[@]}"
      "${EVAL_CMD[@]}"
    fi
    EVAL_METRICS+=("${EVAL_DIR}/metrics.json")
  done
done

if [[ "${HAS_MLP}" -eq 1 ]]; then
  SUM_MLP_CMD=(
    "${PYTHON_BIN}" "${REPO_ROOT}/scripts/trex_offline.py" summarize
    "${EVAL_METRICS[@]}"
    --baseline "${REP_MODEL}"
    --output "${OUT_CLEAN}/logs/summary_vs_mlp.json"
  )
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    printf '[dry-run summarize]'
    printf ' %q' "${SUM_MLP_CMD[@]}"
    printf '\n'
  else
    "${SUM_MLP_CMD[@]}"
  fi
fi

if [[ "${HAS_RESMLP}" -eq 1 ]]; then
  RESMLP_ARGS=()
  for m in "${EVAL_METRICS[@]}"; do
    if [[ "${m}" != *"/eval_mlp/metrics.json" ]]; then
      RESMLP_ARGS+=("${m}")
    fi
  done
  if [[ "${#RESMLP_ARGS[@]}" -gt 0 ]]; then
    SUM_RESMLP_CMD=(
      "${PYTHON_BIN}" "${REPO_ROOT}/scripts/trex_offline.py" summarize
      "${RESMLP_ARGS[@]}"
      --baseline "${REP_MODEL}__resmlp"
      --output "${OUT_CLEAN}/logs/summary_vs_resmlp.json"
    )
    if [[ "${DRY_RUN}" -eq 1 ]]; then
      printf '[dry-run summarize]'
      printf ' %q' "${SUM_RESMLP_CMD[@]}"
      printf '\n'
    else
      "${SUM_RESMLP_CMD[@]}"
    fi
  fi
fi
