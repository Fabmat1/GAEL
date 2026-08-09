#!/usr/bin/env python3
"""Compare GAEL against ISIS on real observed spectra from ASTRA's RVVD project.

100 single-spectrum fits and 100 joint fits of 5 spectra of the same star.
Spectra are drawn from LAMOST/LRS and SDSS/BOSS -- both constant resolving
power, which maps exactly onto GAEL's R(lambda) model.  SOAR is excluded
because those spectra have known fitting issues.

The exact sample is frozen in ``tests/manifests/real_*.json`` so that results
(and the ISIS cache) stay stable as the database grows.  Regenerate with::

    python -m gael_isis.selection --regenerate

    ./test_real_vs_isis.py                 # check against tests/tolerances.json
    ./test_real_vs_isis.py --calibrate     # (re)derive those tolerances
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gael_isis import runner, selection  # noqa: E402


def build_cases(cfg, kind: str, limit: int | None):
    export_dir = cfg.cache_dir / "real_spectra"
    return selection.cases_from_manifest(f"real_{kind}", export_dir, limit)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    runner.add_common_args(ap)
    args = ap.parse_args()
    return runner.execute("real", build_cases, need_db=True, args=args)


if __name__ == "__main__":
    sys.exit(main())
