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
