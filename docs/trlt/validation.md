# Validation procedure and local evidence

Run the bounded GPU and host commands in [README.md](README.md). The implementation
keeps generated experiment results outside Git, under `experiments/trlt/`.

The actual execution report is `experiments/trlt/validation/implementation_report.md`.
Case-level checks and original logs are in `final_test_coverage.json` and the adjacent
XML/log files. Run provenance is `experiments/trlt/implementation_manifest.json`.

The separate [3DTacDex Assembly preparation](3dtacdex_assembly.md) has its own
CPU source/conversion evidence under
`experiments/trlt/3dtacdex_assembly/preparation/`. Inspect `pipeline_status.json`
and actual checksum/audit/validation outputs for completion; it does not change
the existing T-Rex experiment or authorize an Assembly training run.
Its all-source preparation completed on2026-11-17:8CPUtests passed;
22069numeric/source/instructionrows and44138RGBframes matched exactly;
248actual reader boundary samples passed. All124segments, including89`nouse`
segments, are retained with command masks. Actual evidence:
`experiments/trlt/3dtacdex_assembly/preparation/report.md` and
`lerobot_all_validation.json` in that directory. No Assembly benchmark was run.

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

Assembly four-model implementation (2026-11-17): see
[the experiment protocol](3dtacdex_experiments.md) and actual CPU evidence in
`experiments/trlt/3dtacdex_assembly/implementation/report.md`. Checks cover grouped
source trajectories, masked train-only statistics, native tactile models, 22D
element masks, test selection seals and real train/validation worker equivalence.
Full-backbone extraction, GPU optimizer validation, benchmark training and final
test evaluation were not run for this implementation.

Assembly revision (2026-11-18): fusion modality embeddings removed in Assembly
configs, Stage A8000updates/effective batch64/micro4, train/validation/test and
validation checkpoint selection retained by explicit user choice. CPU evidence:
`experiments/trlt/3dtacdex_assembly/revisions/20261118_no_modality_8k_b64/report.md`.
GPU/batch64 benchmark validation and training remain unrun.

Assembly full-microbatch/W&B revision (2026-11-18): Stage A batch64/micro64,
accumulation1, W&B train logging every10optimizer steps with validation events
retained. User authorized bounded GPU validation; real train/validation subset
feature/optimizer checks and actual W&B offline files are recorded under
`experiments/trlt/3dtacdex_assembly/revisions/20261118_full_b64_wandb10/`.
Read `report.md`/`summary.json` for actual completions. No final test or8000-step
benchmark is part of those checks; the earlier micro4 evidence is historical.

The full-microbatch checks completed: four full-size models each passed1GPU
update at64/64/1 +4validation queries; alternating frozen export/probe12updates
also passed. Actual offline records and a single minimal online verification run
were checked through the remote API.28latest CPU testcase outcomes and28CLI dry
runs passed. These bounded results do not mean the8000-step plan was launched.

User-authorized full Assembly experiment launched2026-11-18T01:23:03Z under
`experiments/trlt/3dtacdex_assembly/runs/assembly_8k_fullb64_wandb10_seed0_20261118_prepared/`.
The earlier bounded validation remains separate evidence. Initial observation:
protocol reused, train feature extraction started. Use current status/progress,
stage logs and actual saved checkpoints for later execution claims. No duplicated
queue or implicit resume is authorized by the launch record.

Optimized Assembly feature restart (2026-11-18): user stopped the initial queue and
requested deleting78fresh shared cache anchors. Sequential exact-row CPU decode,
spawned prefetch and batch32vision/stable batch1Gemma extraction passed11CPUtests
and a real64anchor GPU/cache pipeline check; sixteen historical batch1 anchors
match withinrelative L2<1e-4. Direct batched-Gemma numerical mismatch was retained
and rejected. Evidence: `experiments/trlt/3dtacdex_assembly/feature_optimization_20261118/`.
New34-stage run: `assembly_8k_fullb64_batched_seed0_20261118`; consult its actual
status/logs/extraction reports for production completion, not the benchmark.


- Production re-extraction completed and verified at 2026-11-18 02:10 UTC:
  train 4,973 + validation 1,162 = 6,135 anchors; exact expected key sets,
  no test keys, matching cache identity, eight distributed shape/finite/mask
  sample checks passed. Full-stage extraction/write rates: 6.754 / 6.559 anchors/s.
  See [production audit](../../experiments/trlt/3dtacdex_assembly/feature_optimization_20261118/production_cache_validation.json)
  and split reports linked from its optimization report. Queue had advanced to
  stage 4/34 RLT GPU preflight; this is not a main-training completion claim.


Assembly final evaluation completion (verified 2026-11-19 UTC):
`assembly_8k_fullb64_batched_seed0_20261118` finished all 34 stages
at 2026-11-18T16:11:22Z. All four last Stage A checkpoints contain 8,000 updates;
validation-selected A steps are 8000/7888/7656/8000. All four evaluations share
16 source trajectories, identical per-trajectory element counts and 248,112
valid elements each. Primary metrics recomputed from trajectory means match
raw metrics/summary. Selection sealing preceded test extraction. See
[results review](../../experiments/trlt/3dtacdex_assembly/runs/assembly_8k_fullb64_batched_seed0_20261118/results_review.json) and
[protocol/results](3dtacdex_experiments.md#완료된-assembly-실험-결과-2026-11-18).
This supersedes earlier running/preflight-only observations. No new training,
extraction or evaluation was run during the review.


Assembly self-only depth follow-up (2026-11-19): base-selection reuse and explicit
model replacement preserve the four-model default sealing contract. Two CPU
regressions passed (including unchanged records, new representation/probe lineage,
idempotence and mutation rejection); Ruff passed; seven actual follow-up CLI
dry runs passed. Historical checkpoint hashes and isolated report replay passed.
Evidence: `experiments/trlt/3dtacdex_assembly/depth_2layer_validation/`
(`pytest.xml`, `cpu_preflight.json`, per-command dry-run JSON, `report_replay.log`).
Replay uses already-published historical metrics and is not a 2-layer result.
New 9-stage run `assembly_self2_8k_fullb64_seed0_20261119T002037Z` launched at
2026-11-19T00:22:58Z, GPU0; initial observation was 2-layer GPU preflight startup.
Inspect run status/logs for actual GPU/update/benchmark completion; do not infer
completion from launch. No visual cache rebuild or extra seed is requested.

The actual 2-layer GPU preflight subsequently passed one full-size optimizer
update and all 3472 validation queries on GPU0 at batch64/micro64. Main Stage A
started fresh at2026-11-19T00:24:55Z, W&B run `tjyjid6l`. See run
`gpu_preflight_observation.json`; the benchmark/final evaluation are still later
stages. No preflight result is reported as final action-prediction performance.


Assembly self-only2 completion, verified2026-11-19: run `assembly_self2_8k_fullb64_seed0_20261119T002037Z`
finished all9stages at2026-11-19T03:26:42Z. Actual Stage A8000updates; selected
A7888/B epoch12; final primary MSE0.081193840. Same16source trajectories and
248112valid elements as baselines, identical protocol/features; primary
recomputed from source-trajectory means. Only model setting difference is
fusion4→2. Selected checkpoint lineage and test-after-seal timing passed.
Evidence: [results review](../../experiments/trlt/3dtacdex_assembly/runs/assembly_self2_8k_fullb64_seed0_20261119T002037Z/results_review.json) and
[completed results](3dtacdex_experiments.md#self-attention-2층-후속-실험-완료-결과-2026-11-19).
No new GPU work/evaluation ran during this review; earlier startup observations
are historical. Preserve the post-initial-test exploratory designation.


Assembly self2 modality-on follow-up (2026-11-19): the sole experiment-setting
change is `use_modality_embedding:false→true`. CPU serialization regression1test
and7CLI dry runs passed; existing selection hashes and isolated historical
report replay passed. Evidence: `experiments/trlt/3dtacdex_assembly/modality_2layer_validation/`.
Actual run `assembly_self2_modality_8k_fullb64_seed0_20261119T034907Z` launched
at 2026-11-19T03:50:26.710345+00:00, GPU0,9stages. Consult actual logs/state for later
completion. Queue checks true's legacy default omission and actual modality
parameter[3,512]. No feature rebuild. Test follows new validation-selection seal.

Actual modality-on GPU check passed one full-size optimizer update plus all3472
validation queries on GPU0 at64/64/1; saved checkpoint modality embedding is
finite with shape[3,512]. Main Stage A started fresh at2026-11-19T03:52:34.818280+00:00.
See run `gpu_preflight_observation.json`; no full8k/final-evaluation completion
is implied by this bounded check.


Modality-on self2 final completion, verified2026-11-19: run `assembly_self2_modality_8k_fullb64_seed0_20261119T034907Z`
finished9/9stages at2026-11-19T07:06:42Z. Actual Stage A8000updates; selected A7192
and B epoch13. Primary0.080929551, vs off0.081193840 (-0.33%); paired trajectory
95% delta CI[-0.005464,+0.005207] includes zero. Best checkpoint modality embedding
[3,512] is finite/present; canonical ModelConfig comparison differs only in the
modality flag. Same16source trajectories,248112valid elements and protocol/cache
provenance; raw means, checkpoint lineage and test-after-seal timing passed.
See [results review](../../experiments/trlt/3dtacdex_assembly/runs/assembly_self2_modality_8k_fullb64_seed0_20261119T034907Z/results_review.json) and the completed Korean summary
in `3dtacdex_experiments.md`. No new GPU work/evaluation was run during review.

Assembly self2 rawclip30 follow-up (2026-11-19): CPU regression20cases, Ruff,
10actual train/validation query transformations,8synchronous/spawned-worker
query equivalence checks and7CLI dry runs passed. Default normalization identity,
clipping/sign/bounds, invalid-pad masks, unchanged action labels, reconstruction
target and bidirectional preprocessing-provenance rejection are covered.
Evidence: `experiments/trlt/3dtacdex_assembly/rawclip30_validation/`.
Historical report replay is not a new benchmark result. For authorized GPU
preflight/training/evaluation status, inspect the rawclip30 run's actual events
and checkpoints linked from `3dtacdex_experiments.md`.

Rawclip30 full-shape GPU preflight subsequently passed1optimizer update and all
3472validation queries at64/64/1 onGPU0. Saved provenance confirms raw_clip30.
Main Stage A started fresh at2026-11-19T10:53:36Z; bounded validation evidence is
`gpu_preflight_observation.json` in that run. Full benchmark/final evaluation
completion remains to be verified from actual later artifacts.

Assembly clipping-only default (2026-11-20): removed tactile mean/std input and
statistics-generation paths; default and all7Assembly configs use raw_clip30.
State/action statistics and historical artifacts remain intact. Twenty-one CPU
regressions passed, including protocol preparation without tactile statistics,
raw clipping/masks, legacy-mode rejection and bidirectional provenance checks.
Actual default-vs-raw30 real train/validation query checks, spawned-worker
identity, seven CLI dry runs and compatibility with the completed rawclip30
checkpoint are recorded under
`experiments/trlt/3dtacdex_assembly/clipping_default_validation/`.
No new GPU work/training/evaluation was run. Completed benchmark result review:
`experiments/trlt/3dtacdex_assembly/runs/assembly_self2_rawclip30_8k_fullb64_seed0_20261119T104744Z/results_review.json`.
