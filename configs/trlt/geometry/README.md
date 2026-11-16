# Geometry configurations

The `mlp/` and `rope3d/` folders contain independent examples for the three TRLT
fusion schedules. No geometry experiment is started by reading these files.
See [the full protocol](../../../docs/trlt/geometry.md) for preparation and use.

`sharpa_wave_v1.json` is a numeric FK/representative-point specification derived
from Sharpa's Apache-2.0 hand descriptions vendored in the official T-Rex repo:
https://github.com/ZhuoyangLiu2005/T-Rex/tree/ba4047ce3e9f3d2013d60de600e136897e90ccb8/dataset_quickstart/third_party/sharpa-urdf-usd-xml

Source hashes and the precise approximation are embedded in the spec. The source
license and notice are preserved as `SHARPA_LICENSE.txt` and `SHARPA_NOTICE.txt`.
The derivation is an adaptation for CPU FK and approximate pad-center geometry;
it is not an official sensor calibration. XML/STL source downloads remain under
the Git-ignored `datasets/` root; the runtime needs only the checked-in spec.
