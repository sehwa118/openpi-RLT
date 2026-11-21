#!/usr/bin/env python3
"""Assembly entry point using the shared explicit-stage offline runner.

Always supply one of configs/trlt/3dtacdex_assembly/*.json.
"""

import json

from trex_offline import main
from trex_offline import parser

if __name__ == "__main__":
    from openpi.trlt.config import root_path

    args = parser().parse_args()
    if not getattr(args, "config", None):
        raise SystemExit("Assembly requires --config configs/trlt/3dtacdex_assembly/<model>.json")
    if json.loads(root_path(args.config).read_text()).get("dataset_kind") != "3dtacdex_assembly":
        raise SystemExit("tacdex_offline.py requires a 3dtacdex_assembly config")
    main()
