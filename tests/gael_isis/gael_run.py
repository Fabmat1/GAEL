"""Drive the GAEL CLI over the same cases and parse ``fit_parameters.csv``.

GAEL results are intentionally *not* cached: they are the thing under test and
must be recomputed on every run.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from .jobs import Case, FitOutcome, ParamResult, split_qualified

DEFAULT_TIMEOUT = int(os.environ.get("GAEL_TEST_GAEL_TIMEOUT", "3600"))


def build_input(case: Case, filenames: list[str], out_dir: str) -> dict:
    s = case.settings
    initial = {}
    for i, comp in enumerate(s.initial_by_component()):
        for pname, pvalue, pfrozen in comp:
            initial[f"c{i + 1}_{pname}"] = {"value": pvalue, "freeze": pfrozen}

    files = []
    for ref, fname in zip(case.spectra, filenames):
        files.append(
            {
                "filename": fname,
                "spectype": "ASCII_with_3_columns",
                "resOffset": ref.res_offset,
                "resSlope": ref.res_slope,
                "barycorr": 0.0,
            }
        )

    return {
        "initialGuess": initial,
        "grids": s.grid_list(),
        "observations": [
            {
                "files": files,
                "ignore": [list(x) for x in s.ignore],
                "csplineAnchorpoints": [list(x) for x in s.anchors],
                "waveCut": list(s.wave_cut),
            }
        ],
        "outputPath": out_dir,
        "saveModel": "none",
    }


def run(case: Case, cfg, work_root: Path | None = None) -> FitOutcome:
    started = time.time()
    workdir = Path(tempfile.mkdtemp(prefix="gaelfit-", dir=work_root))
    try:
        filenames = []
        for i, ref in enumerate(case.spectra):
            name = f"s{i + 1}.txt"
            shutil.copyfile(ref.path, workdir / name)
            filenames.append(str(workdir / name))

        results = workdir / "results"
        (workdir / "fit.json").write_text(
            json.dumps(build_input(case, filenames, str(results)), indent=2)
        )
        # The CLI looks for global_settings.json next to the binary and in the
        # cwd; copying it in makes the run independent of where it was invoked.
        shutil.copyfile(cfg.global_settings, workdir / "global_settings.json")

        proc = subprocess.run(
            [
                str(cfg.gael_bin),
                "--fit",
                "fit.json",
                "--threads",
                "1",
                "--no-plots",
                "--no-pdf",
            ],
            cwd=workdir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=DEFAULT_TIMEOUT,
        )
        log = proc.stdout.decode("utf-8", "replace")
        (workdir / "gael.log").write_text(log)

        if proc.returncode != 0:
            return FitOutcome(
                ok=False,
                backend="gael",
                error=_first_error(log) or f"GAEL exited with {proc.returncode}",
                n_rejected=_rejected_count(log),
                seconds=time.time() - started,
            )

        outcome = parse_csv(results / "fit_parameters.csv", case)
        outcome.n_rejected = _rejected_count(log)
        outcome.seconds = time.time() - started
        return outcome

    except subprocess.TimeoutExpired:
        return FitOutcome(
            ok=False,
            backend="gael",
            error=f"GAEL timed out after {DEFAULT_TIMEOUT}s",
            seconds=time.time() - started,
        )
    except Exception as exc:  # noqa: BLE001
        return FitOutcome(
            ok=False,
            backend="gael",
            error=f"{type(exc).__name__}: {exc}",
            seconds=time.time() - started,
        )
    finally:
        if os.environ.get("GAEL_TEST_KEEP_WORKDIRS"):
            print(f"[gael] kept {workdir}")
        else:
            shutil.rmtree(workdir, ignore_errors=True)


def _first_error(log: str) -> str:
    for line in log.splitlines():
        if line.startswith("Error:") or "error" in line.lower():
            return line.strip()[:300]
    return ""


def _rejected_count(log: str) -> int:
    for line in log.splitlines():
        if "spectra rejected" in line:
            try:
                return int(line.split("=")[-1].strip())
            except ValueError:
                return 0
    return 0


def parse_csv(path: Path, case: Case) -> FitOutcome:
    """Parse ``fit_parameters.csv``.

    Naming follows ``write_fit_parameters_csv`` in ``src/main.cpp``: tied
    parameters are ``c1_<name>``, untied ones ``c1_<name>_d<k>`` (1-based, in
    submission order).
    """
    if not path.exists():
        return FitOutcome(
            ok=False, backend="gael", error=f"{path.name} not written by GAEL"
        )

    rows: dict[str, tuple[float, float]] = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                rows[row["parameter"]] = (
                    float(row["value"]),
                    float(row["error"]),
                )
            except (TypeError, ValueError, KeyError):
                continue

    s = case.settings

    def csv_name(name: str) -> str:
        """Qualified names are already GAEL's CSV spelling; bare ones are c1."""
        comp, bare = split_qualified(name)
        return f"c{comp}_{bare}"

    tied: dict[str, ParamResult] = {}
    for name in s.tied():
        hit = rows.get(csv_name(name))
        if hit is not None:
            tied[name] = ParamResult(value=hit[0], error=hit[1], frozen=hit[1] == 0.0)

    per_spectrum: list[dict[str, ParamResult]] = []
    for d in range(1, case.n_spectra + 1):
        entry: dict[str, ParamResult] = {}
        for name in s.untied():
            base = csv_name(name)
            # Untied only when more than one spectrum survived; a single
            # spectrum is written without the _d suffix.
            hit = rows.get(f"{base}_d{d}")
            if hit is None and d == 1:
                hit = rows.get(base)
            if hit is not None:
                entry[name] = ParamResult(value=hit[0], error=hit[1])
        if entry:
            per_spectrum.append(entry)

    if not tied:
        return FitOutcome(
            ok=False, backend="gael", error="no stellar parameters in fit_parameters.csv"
        )
    return FitOutcome(ok=True, backend="gael", tied=tied, per_spectrum=per_spectrum)
