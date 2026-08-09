"""Run the cases across all CPU cores and drive the pass/fail decision.

Both fits for a case happen in the same worker process, so a case is fully
independent of every other one.  ISIS is single-threaded and GAEL is pinned to
one thread here, which makes ``jobs`` the real degree of parallelism.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from . import SKIP_EXIT_CODE, compare, config, gael_run, isis_ref
from .jobs import Case, FitOutcome

_WORKER_CFG = None
_WORKER_TMP = None


def _init_worker(cfg, tmp):
    global _WORKER_CFG, _WORKER_TMP
    _WORKER_CFG = cfg
    _WORKER_TMP = tmp
    # Keep BLAS/OpenMP inside GAEL from oversubscribing: the pool already uses
    # every core, so each child must stay single-threaded.
    for var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "TBB_NUM_THREADS",
    ):
        os.environ[var] = "1"


def _run_case(case: Case) -> tuple[Case, FitOutcome, FitOutcome, bool]:
    cfg, tmp = _WORKER_CFG, _WORKER_TMP
    isis_out, cached = isis_ref.get_or_run(case, cfg, tmp)
    # No point spending GAEL time on a case ISIS could not fit.
    if not isis_out.ok:
        return case, isis_out, FitOutcome(ok=False, backend="gael", error="skipped"), cached
    gael_out = gael_run.run(case, cfg, tmp)
    return case, isis_out, gael_out, cached


def run_cases(cases: list[Case], cfg, progress_every: int = 10):
    results = []
    tmp_root = str(cfg.work_dir) if cfg.work_dir else None
    if tmp_root:
        Path(tmp_root).mkdir(parents=True, exist_ok=True)

    started = time.time()
    workers = max(1, min(cfg.jobs, len(cases)))
    print(f"[runner] {len(cases)} cases on {workers} worker(s)", flush=True)

    with tempfile.TemporaryDirectory(prefix="gael-isis-", dir=tmp_root) as tmp:
        with ProcessPoolExecutor(
            max_workers=workers, initializer=_init_worker, initargs=(cfg, tmp)
        ) as pool:
            futures = {pool.submit(_run_case, c): c for c in cases}
            done = 0
            for fut in as_completed(futures):
                case = futures[fut]
                try:
                    results.append(fut.result())
                except Exception as exc:  # noqa: BLE001
                    results.append(
                        (
                            case,
                            FitOutcome(
                                ok=False,
                                backend="isis",
                                error=f"worker crashed: {type(exc).__name__}: {exc}",
                            ),
                            FitOutcome(ok=False, backend="gael", error="skipped"),
                            False,
                        )
                    )
                done += 1
                if done % progress_every == 0 or done == len(cases):
                    elapsed = time.time() - started
                    rate = done / elapsed if elapsed else 0
                    eta = (len(cases) - done) / rate if rate else 0
                    print(
                        f"[runner] {done}/{len(cases)} done "
                        f"({elapsed:.0f}s elapsed, ~{eta:.0f}s left)",
                        flush=True,
                    )

    # Deterministic ordering regardless of completion order.
    results.sort(key=lambda r: r[0].case_id)
    return results


def add_common_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument(
        "--calibrate",
        action="store_true",
        help="write tolerances derived from this run instead of checking them",
    )
    ap.add_argument(
        "--limit", type=int, default=None, help="only use the first N cases"
    )
    ap.add_argument("--jobs", type=int, default=None, help="parallel workers")
    ap.add_argument(
        "--artifacts",
        type=Path,
        default=None,
        help="directory for the summary JSON and per-fit delta CSV",
    )
    ap.add_argument(
        "--kinds",
        default="single,multi",
        help="comma-separated subset of {single,multi}",
    )


def execute(
    suite: str,
    build_cases,
    *,
    need_db: bool,
    args,
    has_truth: bool = False,
) -> int:
    """Shared main() body for the two test scripts."""
    cfg = config.load()
    if args.jobs:
        cfg.jobs = args.jobs

    # ISIS is only needed for references that are not cached yet, so its
    # absence is checked per suite below rather than up front.
    reason = cfg.missing(need_isis=False, need_db=need_db)
    if reason:
        print(f"SKIP: {reason}")
        return SKIP_EXIT_CODE

    kinds = [k.strip() for k in args.kinds.split(",") if k.strip()]
    overall_ok = True
    reports = []

    for kind in kinds:
        try:
            cases = build_cases(cfg, kind, args.limit)
        except Exception as exc:  # noqa: BLE001
            print(f"SKIP: cannot build {suite}/{kind} cases: {exc}")
            return SKIP_EXIT_CODE
        if not cases:
            print(f"SKIP: no cases available for {suite}/{kind}")
            return SKIP_EXIT_CODE

        name = f"{suite}_{kind}"

        # Without ISIS the run is still valid as long as every reference is
        # already cached; if any is missing the suite would silently compare
        # fewer cases, so skip loudly instead.
        if not cfg.isis_bin or not cfg.allow_isis:
            uncached = sum(
                1
                for c in cases
                if isis_ref.load_cached(
                    cfg.cache_dir, isis_ref.cache_key(c, list(cfg.base_paths))
                )
                is None
            )
            if uncached:
                why = (
                    "GAEL_TEST_NO_ISIS is set"
                    if cfg.isis_bin
                    else "ISIS binary not found (set GAEL_TEST_ISIS)"
                )
                print(
                    f"SKIP: {why} and {uncached}/{len(cases)} {name} "
                    f"reference fits are not cached"
                )
                return SKIP_EXIT_CODE
            print(f"[runner] {name}: using {len(cases)} cached ISIS references")
        results = run_cases(cases, cfg)
        coll = compare.collect(results)
        stats = compare.statistics(coll)
        truth = compare.truth_statistics(results) if has_truth else None

        reports.append(compare.report(name, coll, stats, truth))

        if args.artifacts:
            compare.write_artifacts(args.artifacts, name, coll, stats, truth)

        if args.calibrate:
            spec = compare.calibrate(name, coll, stats)
            compare.write_tolerances(name, spec)
            reports.append(
                f"[calibrate] wrote tolerances for {name} to "
                f"{compare.TOLERANCE_FILE}"
            )
        elif args.limit:
            # The thresholds are calibrated for the full suite; a truncated run
            # cannot satisfy their sample-size floors, so report without gating.
            reports.append(
                f"NOTE: --limit {args.limit} is a partial run of {name}; "
                f"reporting only, tolerances not applied"
            )
        else:
            passed, problems = compare.check(
                name, coll, stats, compare.load_tolerances()
            )
            if problems and passed:
                reports.append("\n".join(f"NOTE: {m}" for m in problems))
            elif not passed:
                overall_ok = False
                reports.append(
                    f"FAIL [{name}]\n" + "\n".join(f"  - {m}" for m in problems)
                )
            else:
                reports.append(f"PASS [{name}]")

    print()
    print("\n\n".join(reports))
    sys.stdout.flush()
    return 0 if overall_ok else 1
