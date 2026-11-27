#!/usr/bin/env bash
# Assembly tactile MLP-GRU Stage B comparison (docs/trlt/assembly_tactile_gru_plan.md), one stage per call.
#
#   b0   part_attention hierarchical d128 drop0.1, no separate touch path (REUSED checkpoints, --b0-root)
#   m1   + current-frame raw tip taxel xyz -> MLP          (tactile_gru/probe_tgru_m1.json, raw T=1 cache)
#   m5   + 5-frame concatenated xyz -> MLP                 (probe_tgru_m5.json,  raw T=5 cache)
#   m5s  + 5-frame xyz -> capacity-matched MLP (width 53)  (probe_tgru_m5s.json, raw T=5 cache)
#   g5   + 5-frame xyz -> shared GRU(32), last hidden      (probe_tgru_g5.json,  raw T=5 cache)
#
# Stages (run in this order; none starts another):
#   caches       CPU  train/validation raw touch caches (T=1 and/or T=5, only those needed)
#   train        GPU  Stage B training of the selected trained variants (never b0)
#   seal         CPU  seal every best checkpoint (incl. reused b0) per seed before any test access
#   test-caches  CPU  test raw touch caches (requires the seal)
#   evaluate     GPU  one test evaluation per variant and seed (requires the seal)
#   summarize    CPU  paired trajectory bootstrap vs b0 and vs m5 (primary G5-M5)
set -euo pipefail

if [[ "$(id -u)" -eq 0 ]]; then
  echo "Refusing to run as root; use your personal account." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

STAGE=""
PROTOCOL_DIR=""
DATA_ROOT=""
LATENT_ROOT=""
LATENT_SEED="0"
NORMALIZATION=""
OUT_ROOT=""
TOUCH_ROOT=""
CACHE_ROOT=""
B0_ROOT=""
PYTHON_BIN="python"
DEVICE="0"
SEEDS="0 1 2"
VARIANTS="b0 m1 m5 m5s g5"
REP_MODEL="trlt_self_only"
MIN_FREE_GB="10"
MEM_FRACTION="0.05"
CPU_THREADS="16"
WANDB_MODE="disabled"
MAX_STEPS=""
DRY_RUN=0

usage() {
  sed -n '2,16p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  cat <<'EOF'

Required:
  --stage NAME           caches | train | seal | test-caches | evaluate | summarize
  --protocol-dir PATH    Assembly protocol directory (protocol.json, splits, queries)
  --data-root PATH       Assembly LeRobot root (lerobot_all)
  --latent-root PATH     Parent of seed<S>/{train,validation,test} frozen latent caches
  --out-root PATH        Writable root under /workspace/users/<you>/... (checkpoints, logs)
Required for some stages:
  --b0-root PATH         Read-only root holding checkpoints/seed<S>/probe_part_attention_hierarchical_drop01
                         (seal/evaluate/summarize when b0 is selected)
  --normalization PATH   protocol normalization.json (evaluate)
Optional:
  --touch-root PATH      Touch cache root (default <out-root>/touch_cache)
  --latent-seed N        Frozen latent seed directory (default 0)
  --seeds "0 1 2"        Stage B seeds (default "0 1 2")
  --variants "b0 m1 m5 m5s g5"
  --device IDX           Single GPU index (default 0)
  --min-free-gb N        Refuse GPU stages when free memory on --device is below N GB (default 10)
  --mem-fraction F       XLA_PYTHON_CLIENT_MEM_FRACTION cap for GPU stages (default 0.05, about 4.8 GB of 96 GB)
  --cpu-threads N        Threads for CPU cache building (default 16)
  --python BIN           Python executable (default python)
  --cache-root PATH      UV/PIP/HF/XDG/WANDB caches (default <out-root>/.cache)
  --wandb MODE           disabled | offline | online (default disabled)
  --max-steps N          Optional Stage B step cap (smoke tests; use a separate --out-root)
  --dry-run              Print the commands of the stage without executing them
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage) STAGE="${2:-}"; shift 2 ;;
    --protocol-dir) PROTOCOL_DIR="${2:-}"; shift 2 ;;
    --data-root) DATA_ROOT="${2:-}"; shift 2 ;;
    --latent-root) LATENT_ROOT="${2:-}"; shift 2 ;;
    --latent-seed) LATENT_SEED="${2:-}"; shift 2 ;;
    --normalization) NORMALIZATION="${2:-}"; shift 2 ;;
    --out-root) OUT_ROOT="${2:-}"; shift 2 ;;
    --touch-root) TOUCH_ROOT="${2:-}"; shift 2 ;;
    --cache-root) CACHE_ROOT="${2:-}"; shift 2 ;;
    --b0-root) B0_ROOT="${2:-}"; shift 2 ;;
    --python) PYTHON_BIN="${2:-}"; shift 2 ;;
    --device) DEVICE="${2:-}"; shift 2 ;;
    --seeds) SEEDS="${2:-}"; shift 2 ;;
    --variants) VARIANTS="${2:-}"; shift 2 ;;
    --rep-model) REP_MODEL="${2:-}"; shift 2 ;;
    --min-free-gb) MIN_FREE_GB="${2:-}"; shift 2 ;;
    --mem-fraction) MEM_FRACTION="${2:-}"; shift 2 ;;
    --cpu-threads) CPU_THREADS="${2:-}"; shift 2 ;;
    --wandb) WANDB_MODE="${2:-}"; shift 2 ;;
    --max-steps) MAX_STEPS="${2:-}"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
  esac
done

case "${STAGE}" in
  caches|train|seal|test-caches|evaluate|summarize) ;;
  *) echo "Error: --stage must be one of caches, train, seal, test-caches, evaluate, summarize." >&2; exit 1 ;;
esac
for pair in "protocol-dir:${PROTOCOL_DIR}" "data-root:${DATA_ROOT}" "latent-root:${LATENT_ROOT}" "out-root:${OUT_ROOT}"; do
  if [[ -z "${pair#*:}" ]]; then
    echo "Error: --${pair%%:*} is required." >&2
    exit 1
  fi
done
if [[ ! "${DEVICE}" =~ ^[0-9]+$ ]]; then
  echo "Error: --device must be one GPU index (got '${DEVICE}')." >&2
  exit 1
fi

CURRENT_USER="$(id -un)"
validate_writable_path() {  # outputs only under the caller's own /workspace/users/<user>/...
  local label="$1" clean="${2%/}"
  if [[ "${clean}" =~ (^|/)\.\.(/|$) ]]; then
    echo "Error: ${label} must not contain '..'." >&2; exit 1
  fi
  case "${clean}" in
    /workspace/users/"${CURRENT_USER}"/*) ;;
    *) echo "Error: ${label} must be inside /workspace/users/${CURRENT_USER}/<dir>." >&2; exit 1 ;;
  esac
}
OUT_CLEAN="${OUT_ROOT%/}"
TOUCH_ROOT="${TOUCH_ROOT:-${OUT_CLEAN}/touch_cache}"
TOUCH_ROOT="${TOUCH_ROOT%/}"
CACHE_ROOT="${CACHE_ROOT:-${OUT_CLEAN}/.cache}"
CACHE_ROOT="${CACHE_ROOT%/}"
validate_writable_path "--out-root" "$(realpath -m "${OUT_CLEAN}")"
validate_writable_path "--touch-root" "$(realpath -m "${TOUCH_ROOT}")"
validate_writable_path "--cache-root" "$(realpath -m "${CACHE_ROOT}")"

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
export UV_CACHE_DIR="${CACHE_ROOT}/uv" PIP_CACHE_DIR="${CACHE_ROOT}/pip" HF_HOME="${CACHE_ROOT}/hf"
export XDG_CACHE_HOME="${CACHE_ROOT}/xdg" WANDB_DIR="${CACHE_ROOT}/wandb"

AA_DIR="${REPO_ROOT}/configs/trlt/3dtacdex_assembly/attention_actor"
TG_DIR="${REPO_ROOT}/configs/trlt/3dtacdex_assembly/tactile_gru"
config_for() {
  case "$1" in
    b0) echo "${AA_DIR}/probe_part_attn_hierarchical_drop01.json" ;;
    m1|m5|m5s|g5) echo "${TG_DIR}/probe_tgru_$1.json" ;;
    *) echo "Error: unknown variant '$1' (use b0 m1 m5 m5s g5)." >&2; return 1 ;;
  esac
}
frames_for() {  # raw touch history frames; empty for b0
  case "$1" in m1) echo 1 ;; m5|m5s|g5) echo 5 ;; *) echo "" ;; esac
}
name_for() {  # probe_variant_name (checkpoint/log directory suffix)
  case "$1" in
    b0) echo "part_attention_hierarchical_drop01" ;;
    *) echo "touch_part_attention_hierarchical_tgru_$1" ;;
  esac
}
touch_dir() {  # touch_dir <frames> <split>
  echo "${TOUCH_ROOT}/raw_t$1/$2"
}

read -r -a SEED_LIST <<< "${SEEDS}"
read -r -a VARIANT_LIST <<< "${VARIANTS}"
for variant in "${VARIANT_LIST[@]}"; do
  cfg="$(config_for "${variant}")"
  [[ -f "${cfg}" ]] || { echo "Error: missing config ${cfg}" >&2; exit 1; }
done
has_b0=0; [[ " ${VARIANTS} " == *" b0 "* ]] && has_b0=1
if [[ "${has_b0}" -eq 1 && "${STAGE}" =~ ^(seal|evaluate)$ && -z "${B0_ROOT}" ]]; then
  echo "Error: b0 is reused, not trained: pass --b0-root for ${STAGE}." >&2; exit 1
fi

run() {  # run <cmd...>: print, then execute unless --dry-run
  printf '+'; printf ' %q' "$@"; printf '\n'
  if [[ "${DRY_RUN}" -eq 0 ]]; then "$@"; fi
}

gpu_guard() {
  export XLA_PYTHON_CLIENT_PREALLOCATE=false
  export XLA_PYTHON_CLIENT_MEM_FRACTION="${MEM_FRACTION}"
  export CUDA_VISIBLE_DEVICES="${DEVICE}"
  unset JAX_PLATFORMS
  if [[ "${DRY_RUN}" -eq 1 ]]; then return 0; fi
  local free_mib
  free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${DEVICE}" | head -n1 | tr -d ' ')"
  if [[ ! "${free_mib}" =~ ^[0-9]+$ ]]; then
    echo "Error: could not read free memory for GPU ${DEVICE}." >&2; exit 1
  fi
  if (( free_mib < MIN_FREE_GB * 1024 )); then
    echo "Error: GPU ${DEVICE} has ${free_mib} MiB free (< ${MIN_FREE_GB} GB). Not starting." >&2
    exit 1
  fi
  echo "GPU ${DEVICE}: ${free_mib} MiB free; XLA mem fraction ${MEM_FRACTION}, no preallocation."
}

cpu_env() {  # Hide GPUs from CPU-only stages.
  export CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu
}

build_caches() {  # build_caches <split> [seal]
  local split="$1" seal="${2:-}" done_frames=""
  cpu_env
  for variant in "${VARIANT_LIST[@]}"; do
    local frames; frames="$(frames_for "${variant}")"
    [[ -z "${frames}" || " ${done_frames} " == *" ${frames} "* ]] && continue
    done_frames+=" ${frames}"
    local out; out="$(touch_dir "${frames}" "${split}")"
    if [[ -f "${out}/manifest.json" ]]; then
      echo "skip existing ${out}"; continue
    fi
    local cmd=("${PYTHON_BIN}" "${REPO_ROOT}/scripts/build_touch_tokens.py"
      --protocol-dir "${PROTOCOL_DIR}" --data-root "${DATA_ROOT}" --split "${split}"
      --source raw --frames "${frames}" --output-dir "${out}" --threads "${CPU_THREADS}")
    [[ -n "${seal}" ]] && cmd+=(--selection-seal "${seal}")
    run "${cmd[@]}"
  done
}

seal_path() { echo "${OUT_CLEAN}/logs/seed$1/stage_b_selections.json"; }
ckpt_dir() {  # ckpt_dir <seed> <variant>
  if [[ "$2" == "b0" ]]; then
    echo "${B0_ROOT%/}/checkpoints/seed$1/probe_$(name_for b0)"
  else
    echo "${OUT_CLEAN}/checkpoints/seed$1/probe_$(name_for "$2")"
  fi
}
LAT() { echo "${LATENT_ROOT%/}/seed${LATENT_SEED}/$1"; }

case "${STAGE}" in
  caches)
    build_caches train
    build_caches validation
    ;;
  train)
    gpu_guard
    for seed in "${SEED_LIST[@]}"; do
      for variant in "${VARIANT_LIST[@]}"; do
        [[ "${variant}" == "b0" ]] && { echo "b0: reused checkpoints, not trained"; continue; }
        CKPT="$(ckpt_dir "${seed}" "${variant}")"
        if [[ -e "${CKPT}" ]]; then
          echo "Error: ${CKPT} already exists; inspect it instead of retraining." >&2; exit 1
        fi
        frames="$(frames_for "${variant}")"
        cmd=("${PYTHON_BIN}" "${REPO_ROOT}/scripts/trex_offline.py" train-probe
          --config "$(config_for "${variant}")" --device "${DEVICE}" --seed "${seed}"
          --train-latents "$(LAT train)" --validation-latents "$(LAT validation)"
          --train-touch "$(touch_dir "${frames}" train)" --validation-touch "$(touch_dir "${frames}" validation)"
          --output-dir "${CKPT}" --log-dir "${OUT_CLEAN}/logs/seed${seed}/probe_$(name_for "${variant}")"
          --wandb "${WANDB_MODE}")
        [[ -n "${MAX_STEPS}" ]] && cmd+=(--max-steps "${MAX_STEPS}")
        run "${cmd[@]}"
      done
    done
    ;;
  seal)
    cpu_env
    for seed in "${SEED_LIST[@]}"; do
      args=()
      for variant in "${VARIANT_LIST[@]}"; do args+=(--probe "${variant}=$(ckpt_dir "${seed}" "${variant}")/best"); done
      run "${PYTHON_BIN}" "${REPO_ROOT}/scripts/prepare_tacdex_experiment.py" seal-probes \
        --protocol-dir "${PROTOCOL_DIR}" --seed "${seed}" --output "$(seal_path "${seed}")" "${args[@]}"
    done
    ;;
  test-caches)
    # Touch caches do not depend on the Stage B seed; every seed must be sealed before test access.
    for seed in "${SEED_LIST[@]}"; do
      if [[ "${DRY_RUN}" -eq 0 && ! -f "$(seal_path "${seed}")" ]]; then
        echo "Error: run --stage seal first ($(seal_path "${seed}") missing)." >&2; exit 1
      fi
    done
    build_caches test "$(seal_path "${SEED_LIST[0]}")"
    ;;
  evaluate)
    [[ -n "${NORMALIZATION}" ]] || { echo "Error: --normalization is required for evaluate." >&2; exit 1; }
    gpu_guard
    for seed in "${SEED_LIST[@]}"; do
      for variant in "${VARIANT_LIST[@]}"; do
        out="${OUT_CLEAN}/logs/seed${seed}/eval_$(name_for "${variant}")"
        if [[ -e "${out}/metrics.json" ]]; then
          echo "Error: ${out} already evaluated; never re-run test evaluation." >&2; exit 1
        fi
        cmd=("${PYTHON_BIN}" "${REPO_ROOT}/scripts/trex_offline.py" evaluate
          --config "$(config_for "${variant}")" --device "${DEVICE}" --seed "${seed}"
          --latents "$(LAT test)" --probe-checkpoint "$(ckpt_dir "${seed}" "${variant}")/best"
          --normalization "${NORMALIZATION}" --output-dir "${out}"
          --protocol-dir "${PROTOCOL_DIR}" --selection-seal "$(seal_path "${seed}")")
        frames="$(frames_for "${variant}")"
        [[ -n "${frames}" ]] && cmd+=(--touch "$(touch_dir "${frames}" test)")
        run "${cmd[@]}"
      done
    done
    ;;
  summarize)
    cpu_env
    metrics=()
    for seed in "${SEED_LIST[@]}"; do
      for variant in "${VARIANT_LIST[@]}"; do metrics+=("${OUT_CLEAN}/logs/seed${seed}/eval_$(name_for "${variant}")/metrics.json"); done
    done
    for base in b0 m5 m1; do
      [[ " ${VARIANTS} " == *" ${base} "* ]] || continue
      run "${PYTHON_BIN}" "${REPO_ROOT}/scripts/trex_offline.py" summarize "${metrics[@]}" \
        --baseline "${REP_MODEL}__$(name_for "${base}")" --output "${OUT_CLEAN}/logs/summary_vs_${base}.json"
    done
    ;;
esac
