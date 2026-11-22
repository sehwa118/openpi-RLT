#!/usr/bin/env bash
# LingBot-Touch Stage B comparison on 3DTacDex Assembly (H0..H3), one explicit stage per call.
#
#   H0 part_attention_hierarchical                (z + proprio, existing baseline config)
#   H1 + pretrained LingBot-Touch Tiny tokens     (configs/.../lingbot_touch/probe_touch_lingbot_tiny.json)
#   H2 + random-init LingBot tokens (control)     (probe_touch_lingbot_random.json)
#   H3 + raw PaXini taxel tokens (control)        (probe_touch_raw_taxel.json)
#
# Stages (run in this order; none starts another):
#   caches       CPU  train/validation touch caches for the selected touch variants
#   train        GPU  Stage B training, validation-based best checkpoint
#   seal         CPU  seal every selected best checkpoint before any test access
#   test-caches  CPU  test touch caches (requires the seal)
#   evaluate     GPU  one test evaluation per variant (requires the seal)
#   summarize    CPU  paired trajectory-bootstrap summaries vs H0 and vs H2
set -euo pipefail

if [[ "$(id -u)" -eq 0 ]]; then
  echo "Refusing to run as root; use your personal account (ssh gpu)." >&2
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
BUNDLE=""
OUT_ROOT=""
TOUCH_ROOT=""
CACHE_ROOT=""
PYTHON_BIN="python"
DEVICE="0"
SEEDS="0"
VARIANTS="h0 h1 h2 h3"
REP_MODEL="trlt_self_only"
RANDOM_INIT_SEED="0"
FRAMES="2"
MIN_FREE_GB="20"
MEM_FRACTION="0.20"
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
  --normalization PATH   protocol normalization.json (evaluate)
  --out-root PATH        Writable root under /workspace/users/<you>/... (checkpoints, logs)
Optional:
  --bundle PATH          Converted LingBot Flax bundle (needed when h1 is selected)
  --touch-root PATH      Touch cache root (default <out-root>/touch_cache)
  --latent-seed N        Frozen latent seed directory (default 0)
  --seeds "0 1 2"        Stage B seeds (default "0")
  --variants "h0 h1 h2 h3"
  --frames N             LingBot window length T (default 2)
  --device IDX           Single GPU index (default 0)
  --min-free-gb N        Refuse GPU stages when free memory on --device is below N GB (default 20)
  --mem-fraction F       XLA_PYTHON_CLIENT_MEM_FRACTION cap for GPU stages (default 0.20, about 19 GB of 96 GB)
  --cpu-threads N        Threads for CPU cache building (default 16)
  --python BIN           Python executable (default python)
  --cache-root PATH      UV/PIP/HF/XDG/WANDB caches (default <out-root>/.cache)
  --wandb MODE           disabled | offline | online (default disabled)
  --max-steps N          Optional Stage B step cap (smoke tests)
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
    --bundle) BUNDLE="${2:-}"; shift 2 ;;
    --out-root) OUT_ROOT="${2:-}"; shift 2 ;;
    --touch-root) TOUCH_ROOT="${2:-}"; shift 2 ;;
    --cache-root) CACHE_ROOT="${2:-}"; shift 2 ;;
    --python) PYTHON_BIN="${2:-}"; shift 2 ;;
    --device) DEVICE="${2:-}"; shift 2 ;;
    --seeds) SEEDS="${2:-}"; shift 2 ;;
    --variants) VARIANTS="${2:-}"; shift 2 ;;
    --rep-model) REP_MODEL="${2:-}"; shift 2 ;;
    --frames) FRAMES="${2:-}"; shift 2 ;;
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
validate_writable_path() {
  local label="$1" clean="${2%/}"
  if [[ "${clean}" =~ (^|/)\.\.(/|$) ]]; then
    echo "Error: ${label} must not contain '..'." >&2; exit 1
  fi
  if [[ "${clean}" == "/workspace" || "${clean}" == /workspace/* ]]; then
    case "${clean}" in
      /workspace/users/"${CURRENT_USER}"/*|/workspace/teams/*/*) ;;
      *) echo "Error: ${label} must be inside /workspace/users/${CURRENT_USER}/<dir> or /workspace/teams/<team>/<dir>." >&2; exit 1 ;;
    esac
  fi
}
OUT_CLEAN="${OUT_ROOT%/}"
TOUCH_ROOT="${TOUCH_ROOT:-${OUT_CLEAN}/touch_cache}"
TOUCH_ROOT="${TOUCH_ROOT%/}"
CACHE_ROOT="${CACHE_ROOT:-${OUT_CLEAN}/.cache}"
CACHE_ROOT="${CACHE_ROOT%/}"
validate_writable_path "--out-root" "${OUT_CLEAN}"
validate_writable_path "--touch-root" "${TOUCH_ROOT}"
validate_writable_path "--cache-root" "${CACHE_ROOT}"

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
export UV_CACHE_DIR="${CACHE_ROOT}/uv" PIP_CACHE_DIR="${CACHE_ROOT}/pip" HF_HOME="${CACHE_ROOT}/hf"
export XDG_CACHE_HOME="${CACHE_ROOT}/xdg" WANDB_DIR="${CACHE_ROOT}/wandb"

AA_DIR="${REPO_ROOT}/configs/trlt/3dtacdex_assembly/attention_actor"
LT_DIR="${REPO_ROOT}/configs/trlt/3dtacdex_assembly/lingbot_touch"
config_for() {
  case "$1" in
    h0) echo "${AA_DIR}/probe_part_attn_hierarchical.json" ;;
    h1) echo "${LT_DIR}/probe_touch_lingbot_tiny.json" ;;
    h2) echo "${LT_DIR}/probe_touch_lingbot_random.json" ;;
    h3) echo "${LT_DIR}/probe_touch_raw_taxel.json" ;;
    *) echo "Error: unknown variant '$1' (use h0 h1 h2 h3)." >&2; return 1 ;;
  esac
}
source_for() {
  case "$1" in h1) echo lingbot ;; h2) echo random ;; h3) echo raw ;; *) echo "" ;; esac
}
name_for() {
  case "$1" in
    h0) echo "part_attention_hierarchical" ;;
    h1) echo "touch_part_attention_hierarchical_lingbot_tiny" ;;
    h2) echo "touch_part_attention_hierarchical_lingbot_random" ;;
    h3) echo "touch_part_attention_hierarchical_raw_taxel" ;;
  esac
}
touch_dir() {  # touch_dir <source> <split>
  echo "${TOUCH_ROOT}/$1_t${FRAMES}/$2"
}

read -r -a SEED_LIST <<< "${SEEDS}"
read -r -a VARIANT_LIST <<< "${VARIANTS}"
for variant in "${VARIANT_LIST[@]}"; do
  cfg="$(config_for "${variant}")"
  [[ -f "${cfg}" ]] || { echo "Error: missing config ${cfg}" >&2; exit 1; }
  if [[ "${variant}" == "h1" && -z "${BUNDLE}" && "${STAGE}" == *caches ]]; then
    echo "Error: h1 requires --bundle for cache building." >&2; exit 1
  fi
done

run() {  # run <cmd...>: print, then execute unless --dry-run
  printf '+'; printf ' %q' "$@"; printf '\n'
  if [[ "${DRY_RUN}" -eq 0 ]]; then "$@"; fi
}

gpu_guard() {
  export XLA_PYTHON_CLIENT_PREALLOCATE=false
  export XLA_PYTHON_CLIENT_MEM_FRACTION="${MEM_FRACTION}"
  if [[ "${DRY_RUN}" -eq 1 ]]; then return 0; fi
  local free_mib
  free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${DEVICE}" | head -n1 | tr -d ' ')"
  if [[ ! "${free_mib}" =~ ^[0-9]+$ ]]; then
    echo "Error: could not read free memory for GPU ${DEVICE}." >&2; exit 1
  fi
  if (( free_mib < MIN_FREE_GB * 1024 )); then
    echo "Error: GPU ${DEVICE} has ${free_mib} MiB free (< ${MIN_FREE_GB} GB). Check nvitop; not starting." >&2
    exit 1
  fi
  echo "GPU ${DEVICE}: ${free_mib} MiB free; XLA mem fraction ${MEM_FRACTION}, no preallocation."
}

cpu_env() {  # Hide GPUs from CPU-only stages.
  export CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu
}

build_caches() {  # build_caches <split> [seal]
  local split="$1" seal="${2:-}"
  cpu_env
  local done_sources=""
  for variant in "${VARIANT_LIST[@]}"; do
    local src; src="$(source_for "${variant}")"
    [[ -z "${src}" || " ${done_sources} " == *" ${src} "* ]] && continue
    done_sources+=" ${src}"
    local out; out="$(touch_dir "${src}" "${split}")"
    if [[ -f "${out}/manifest.json" ]]; then
      echo "skip existing ${out}"; continue
    fi
    local cmd=("${PYTHON_BIN}" "${REPO_ROOT}/scripts/build_touch_tokens.py"
      --protocol-dir "${PROTOCOL_DIR}" --data-root "${DATA_ROOT}" --split "${split}"
      --source "${src}" --frames "${FRAMES}" --output-dir "${out}" --threads "${CPU_THREADS}")
    case "${src}" in
      lingbot) cmd+=(--bundle "${BUNDLE}") ;;
      random) cmd+=(--random-init-seed "${RANDOM_INIT_SEED}") ;;
    esac
    [[ -n "${seal}" ]] && cmd+=(--selection-seal "${seal}")
    run "${cmd[@]}"
  done
}

seal_path() { echo "${OUT_CLEAN}/logs/seed$1/stage_b_selections.json"; }
ckpt_dir() { echo "${OUT_CLEAN}/checkpoints/seed$1/probe_$2"; }
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
        CKPT="$(ckpt_dir "${seed}" "${variant}")"
        if [[ -f "${CKPT}/last/metadata.json" ]]; then
          echo "Error: ${CKPT} already has a checkpoint; inspect it instead of retraining." >&2; exit 1
        fi
        cmd=("${PYTHON_BIN}" "${REPO_ROOT}/scripts/trex_offline.py" train-probe
          --config "$(config_for "${variant}")" --device "${DEVICE}" --seed "${seed}"
          --train-latents "$(LAT train)" --validation-latents "$(LAT validation)"
          --output-dir "${CKPT}" --log-dir "${OUT_CLEAN}/logs/seed${seed}/probe_${variant}" --wandb "${WANDB_MODE}")
        src="$(source_for "${variant}")"
        if [[ -n "${src}" ]]; then
          cmd+=(--train-touch "$(touch_dir "${src}" train)" --validation-touch "$(touch_dir "${src}" validation)")
        fi
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
    # Touch tokens do not depend on the Stage B seed; any sealed seed authorizes test access.
    SEAL="$(seal_path "${SEED_LIST[0]}")"
    if [[ "${DRY_RUN}" -eq 0 && ! -f "${SEAL}" ]]; then
      echo "Error: run --stage seal first (${SEAL} missing)." >&2; exit 1
    fi
    build_caches test "${SEAL}"
    ;;
  evaluate)
    [[ -n "${NORMALIZATION}" ]] || { echo "Error: --normalization is required for evaluate." >&2; exit 1; }
    gpu_guard
    for seed in "${SEED_LIST[@]}"; do
      for variant in "${VARIANT_LIST[@]}"; do
        cmd=("${PYTHON_BIN}" "${REPO_ROOT}/scripts/trex_offline.py" evaluate
          --config "$(config_for "${variant}")" --device "${DEVICE}" --seed "${seed}"
          --latents "$(LAT test)" --probe-checkpoint "$(ckpt_dir "${seed}" "${variant}")/best"
          --normalization "${NORMALIZATION}" --output-dir "${OUT_CLEAN}/logs/seed${seed}/eval_${variant}"
          --protocol-dir "${PROTOCOL_DIR}" --selection-seal "$(seal_path "${seed}")")
        src="$(source_for "${variant}")"
        [[ -n "${src}" ]] && cmd+=(--touch "$(touch_dir "${src}" test)")
        run "${cmd[@]}"
      done
    done
    ;;
  summarize)
    cpu_env
    metrics=()
    for seed in "${SEED_LIST[@]}"; do
      for variant in "${VARIANT_LIST[@]}"; do metrics+=("${OUT_CLEAN}/logs/seed${seed}/eval_${variant}/metrics.json"); done
    done
    for base in h0 h2; do
      [[ " ${VARIANTS} " == *" ${base} "* ]] || continue
      run "${PYTHON_BIN}" "${REPO_ROOT}/scripts/trex_offline.py" summarize "${metrics[@]}" \
        --baseline "${REP_MODEL}__$(name_for "${base}")" --output "${OUT_CLEAN}/logs/summary_vs_${base}.json"
    done
    ;;
esac
