# Project instructions

## Required environment

- Activate the `sehwa` conda environment before working in this repository, including running Git, GitHub, Hugging Face, Python, and uv commands.
- This server is shared. The user's GitHub and Hugging Face account configuration is scoped to `sehwa`; use that environment for all account access. Do not change shared or other users' authentication settings.
- Shell activation does not persist between separate command sessions. Start each new shell session with:

  ```bash
  source /workspace/miniforge3/etc/profile.d/conda.sh
  conda activate sehwa
  ```

- The uv environment setup described in `README.md` has already been completed. Use the existing setup; do not repeat installation or recreate environments unless the task requires it.
- Run the appropriate project command after activating `sehwa`, including `uv run` when using the project's uv environment.

## GPU usage

- Do not use GPUs unless the user explicitly requests GPU usage for the task.
- Implementing code does not authorize running it on a GPU for validation. Use CPU-only checks by default; leave GPU-dependent validation unrun unless explicitly requested.
- For validation commands that may initialize an accelerator, explicitly disable GPU access (for example, `CUDA_VISIBLE_DEVICES=""`; also use `JAX_PLATFORMS=cpu` for JAX) and select CPU devices in the relevant framework.

## Storage locations

- The repository root already contains `datasets/`, `checkpoints/`, and `experiments/`. Use these directories for their respective artifacts.
- Store datasets in `datasets/`.
- Store model checkpoints in `checkpoints/`.
- Store experiment results, including logs, metrics, plots, and reports, in `experiments/`.
- Configure download, training, and evaluation output paths to follow these locations. Resolve these paths relative to the repository root, even when running commands from a subdirectory.
- Keep `datasets/`, `checkpoints/`, and `experiments/` ignored by Git. Do not force-add or commit their contents.

## Project orientation

- `README.md` describes the openpi-based RL Token pipeline and stage-1 training and serving entry points.
- `rlt_online_rl/README.md` describes the real-robot online RL runtime, launch order, and evaluation flow.
- Core model and training code lives in `src/openpi/`; the online RL runtime lives in `rlt_online_rl/`.
