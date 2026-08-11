#!/usr/bin/env python3
"""Compare GAEL against ISIS on the two features the sdB suites cannot reach:
element abundances and more than one stellar component.

Both need a metal-bearing grid, so these run on the Feros grids with
X-Shooter-like mock spectra (UVB + VIS arms) produced by
``gen_mock_xshooter``.  As in the other suites the gate is GAEL-vs-ISIS
agreement; the mock's known truth is reported but not gated.

    ./test_xshooter_vs_isis.py                    # check tolerances
    ./test_xshooter_vs_isis.py --calibrate        # (re)derive them
    ./test_xshooter_vs_isis.py --kinds metal --limit 3     # quick run

See gael_isis/xshooter.py for what each suite holds fixed and why.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gael_isis import runner, xshooter  # noqa: E402


def build_cases(cfg, kind: str, limit: int | None):
    return xshooter.cases(cfg, kind, limit)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    runner.add_common_args(ap)
    # add_common_args defaults to the sdB suites' kinds; this suite has its own.
    ap.set_defaults(kinds="metal,binary")
    args = ap.parse_args()
    return runner.execute(
        "xshooter", build_cases, need_db=False, args=args, has_truth=True
    )


if __name__ == "__main__":
    sys.exit(main())
