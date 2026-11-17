# Validation procedure and local evidence

Run the bounded GPU and host commands in [README.md](README.md). The implementation
keeps generated experiment results outside Git, under `experiments/trlt/`.

The actual execution report is `experiments/trlt/validation/implementation_report.md`.
Case-level checks and original logs are in `final_test_coverage.json` and the adjacent
XML/log files. Run provenance is `experiments/trlt/implementation_manifest.json`.

The checks cover attention connectivity and synchronous updates, readout isolation,
masked losses, causal windows, split/normalization integrity, decoder equality,
finite gradients and updates, frozen representations, cache invalidation, and
checkpoint/optimizer restoration. GPU validation requires explicit device selection
and records actual parameter/input placement. Highest matmul precision is enforced.

Saved parameters and optimizer buffers must restore exactly. Independently compiled
GPU training continuations are checked with strict numerical tolerance; freeze
hash checks remain exact. The local report records observed numerical differences
and any failed assertion that was corrected after diagnosis.

Dataset evidence remains with the pinned source:

- `datasets/trex_squeeze/download_manifest.json`
- `datasets/trex_squeeze/validation_full.json`
- `datasets/trex_squeeze/packet_index_validation.json`
- `datasets/trex_squeeze/splits.json` and `normalization.json`
- `datasets/trex_squeeze/query_manifests/`

Boundary frames are decoded, while a grouped MP4/H264 packet-PTS scan checks selected
interval frame counts and timing. This does not certify physical object/session
identity or the absence of near-duplicate demonstrations.

Full training and final source test action evaluation require an explicit launch
after implementation review. Legacy CLI and parameter-tree checks are distinct
from loading an actual legacy RLT checkpoint into a live policy server; consult
the execution report for what was run and what remains unrun.

Training progress/W&B implementation (2026-11-14): CPU-only synthetic one-update
training with actual W&B offline recording and tracker lifecycle checks passed
(2 tests). Evidence: `experiments/trlt/tracking_validation/report.md`, `pytest.log`,
`pytest.xml` and CLI `dry_run.json`. Online authentication/upload, GPU execution,
and long source-data runs were not performed for this change.


DataLoader/JIT throughput implementation (2026-11-14): validation artifacts are
in `experiments/trlt/loading_validation/`. The report records CPU tests for worker
isolation, exact sample/mask/count equivalence, deterministic resume and partial
batches, concurrent cold-cache construction, nonfinite gradients, plus a bounded
comparison of eight actual **train** samples. Test samples are not used. GPU
throughput/utilization and long training are not validated by these host checks.

Final recorded result: 33 CPU regression tests passed, plus one explicit worker
CPU-configuration isolation check; all eight real train samples matched exactly.
See `experiments/trlt/loading_validation/summary.json` and the report for commands,
actual synthetic optimizer steps, warnings and unrun GPU checks.

W&B grouping update (2026-11-14): 4 CPU-only tracking tests passed, including
one actual synthetic optimizer update with offline W&B. Stage A/B grouping and
exclusion of aggregation counts/mask fractions were verified. Evidence and exact
command: `experiments/trlt/tracking_groups_validation/report.md` and adjacent
`pytest.log`/`pytest.xml`. GPU and online W&B were not run.

Stage A step budget (2026-11-14): the four tracked configs now request15,000
optimizer updates, no epoch limit and no early stopping. Stage B defaults remain
unchanged. CPU-only stopping-policy regression results, CLI plans and exact
commands are recorded in `experiments/trlt/stage_a_15000_validation/report.md`.
This setting change does not launch training, migrate prior run configs or alter
an already running process. Long GPU training and final test evaluation remain
separate execution decisions.

Full user-authorized experiment completed on2026-11-15 at04:17:42UTC, with all28
stages successful: four Stage A runs of15000updates, four frozen action probes,
test feature extraction, four final test evaluations and summary/plot generation.
Only seed0 was requested. Final report, exact configs/commands, checkpoint
selection evidence and per-model outputs:
`experiments/trlt/runs/squeeze_15k_seed0_wandb_20261114T081820Z/final_report.md`.
The selected checkpoint hashes were sealed before test extraction. Final test
episode IDs and valid-element counts match across models. Paired episode-bootstrap
intervals include zero; no across-seed stability claim is supported.

Hand-local geometry extension (2026-11-15): CPU checks completed for mesh-derived
pad-center FK, causal geometry caches, MLP and same-hand wrist/tactile 3D RoPE.
Evidence: `experiments/trlt/geometry_validation/report.md`, `verification.json`,
`geometry_pytest.xml` (16 passed) and `regression_pytest.xml` (59 passed,
10 optimizer-heavy cases deselected). Eight actual train queries match exactly
between synchronous/worker loading; original-source reference FK agrees within
7.32e-9 m. Two reduced synthetic CPU pipelines each completed two representation
updates, latent export and one probe update (optimizer JIT disabled for these
bounded smokes). Train/validation geometry caches cover121/26episodes; test
geometry was not prepared. GPU/full optimizer JIT, benchmark training and final
evaluation were not run. See [geometry.md](geometry.md) for the exact contract,
commands and limits; the execution report distinguishes interrupted initial
compile attempts from completed checks.

Geometry experiment replacement launch (2026-11-15 at07:42:06UTC): user-authorized actual GPU
preflights/training/evaluation are queued in
`experiments/trlt/runs/squeeze_geometry_15k_seed0_20261115T074104Z/`.
The preparation's `cpu_preflight.json` verifies all147train/validationgeometry
episodes and all1573existing test visual anchors without decoding/evaluating test.
Eight CLI dry runs passed. GPU preflight reports are written only after actual
successful optimizer updates; consult per-run status/events and
`*_gpu_preflight.json` before claiming success. Historical CPU-only implementation
validation above remains distinct from this newly authorized experiment.

Encoder sensor-ID ablation (2026-11-16):16CPU tensor/gradient/identity tests passed;
six model initializations/latents exactly match pre-change source when the flag
is enabled, and eight historical saved config identities are preserved. Evidence:
`experiments/trlt/no_sensor_id_validation/report.md`, pytest XML/log and
compatibility.json. The user-authorized20-stage GPU queue for alternating
MLP/RoPE3D with encoder ID disabled and Stage A10000updates is isolated under
`experiments/trlt/runs/squeeze_no_sensor_id_alt_10k_seed0_20261116T062348Z/`.
Stage B defaults remain unchanged; decoder sensor_query is retained. Consult
actual per-run GPU preflight reports/events before reporting GPU success.

Subsequent GPU evidence for the no-sensor-ID experiment: both MLP and RoPE
full-shape preflights passed one optimizer update and four validation queries
on GPU0, batch32/micro4. See the run's `*_gpu_preflight.json` and
`experiments/trlt/no_sensor_id_validation/gpu_preflight_summary.json`. The
bounded checks used train/validation only and do not establish benchmark results.

No-encoder-sensor-ID completion (verified2026-11-17): the authorized run
`squeeze_no_sensor_id_alt_10k_seed0_20261116T062348Z` finished all20stages
at2026-11-16T13:39:08Z. Actual last representation checkpoints show10000updates
for both candidates; both final evaluations contain26episodes and2715096valid
action elements. `final_report.md`, `historical_comparisons.json`,
`comparison_validation.json`, `final_selections.json` and per-model
`seed0/evaluation/metrics.json` are the evidence under that run directory.
Primary MSE:MLP0.066686607, RoPE0.076711648. Historical ID-on comparisons also
change Stage A budget15000→10000; both paired95% intervals include zero.
See [the combined results summary](architecture_and_results.md). No new training,
extraction or evaluation ran during this documentation update.
