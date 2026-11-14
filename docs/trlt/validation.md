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
