#!/usr/bin/env python3
"""Compare GAEL against ISIS on mock spectra with known input parameters.

100 single-spectrum fits and 100 joint fits of 5 spectra each.  The spectra
come from GAEL's own ``mock_data_generator`` with a fixed seed, so the data --
and therefore the cached ISIS references -- are reproducible.

Because the generating parameters are known, this suite also reports how well
each code recovers the truth; only the GAEL-vs-ISIS agreement is gated.

    ./test_mock_vs_isis.py                 # check against tests/tolerances.json
    ./test_mock_vs_isis.py --calibrate     # (re)derive those tolerances
    ./test_mock_vs_isis.py --kinds single --limit 10   # quick smoke run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gael_isis import mockdata, runner  # noqa: E402


def build_cases(cfg, kind: str, limit: int | None):
    return mockdata.cases(cfg, kind, limit)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    runner.add_common_args(ap)
    args = ap.parse_args()
    return runner.execute(
        "mock", build_cases, need_db=False, args=args, has_truth=True
    )


if __name__ == "__main__":
    sys.exit(main())
