"""Pick the real-spectra samples from ASTRA's database and export them to ASCII.

The chosen spectra are frozen into ``tests/manifests/*.json`` and committed, so
the sample (and therefore the ISIS reference cache) stays stable even as the
database grows.  Regenerate deliberately with::

    python -m gael_isis.selection --regenerate
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import asd
from .config import TESTS_DIR
from .jobs import Case, FitSettings, SpectrumRef

MANIFEST_DIR = TESTS_DIR / "manifests"

RVVD_PROJECT = "RVVD"

# Instruments the comparison draws from.  Both have a constant resolving power,
# which maps exactly onto GAEL's R(lambda) = resOffset + resSlope*lambda with
# resSlope = 0, so ISIS and GAEL see an identical resolution model.
# SOAR is deliberately excluded: those spectra have known fitting issues.
INSTRUMENTS = {
    "LAMOST/LRS": (1800.0, 0.0),
    "SDSS/BOSS": (2000.0, 0.0),
}

# Selection quality cuts.
MIN_SNR = 20.0
BLUE_REQUIRED = 3900.0  # spectrum must start at least this blue
RED_REQUIRED = 5200.0  # ... and extend at least this far red
SNR_WINDOW = (4000.0, 5000.0)
MIN_POINTS_IN_WINDOW = 100

# Restrict to stars whose catalogued temperature sits comfortably inside the
# sdB grid (15000-55000 K).  RVVD also contains cooler objects, and fitting
# those with this grid makes both codes rail against the 15000 K edge -- they
# then "agree" trivially and the comparison carries no information.
STAR_TEFF_RANGE = (18000.0, 45000.0)

N_SINGLE = 100
N_MULTI = 100
SPECTRA_PER_MULTI = 5


@dataclass
class SpectrumRow:
    spectrum_id: str
    star_id: str
    data_file: str
    instrument: str


def _connect(db: Path) -> sqlite3.Connection:
    # Read-only: the tests must never modify the user's database.
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    return con


def _project_id(con: sqlite3.Connection, name: str) -> str | None:
    row = con.execute("SELECT id FROM projects WHERE name = ?", (name,)).fetchone()
    return row["id"] if row else None


def _candidate_rows(con: sqlite3.Connection, project_id: str) -> list[SpectrumRow]:
    placeholders = ",".join("?" for _ in INSTRUMENTS)
    sql = f"""
        SELECT s.id AS sid, s.star_id AS star, s.data_file AS df,
               s.instrument AS inst
          FROM spectra s
          JOIN stars st ON s.star_id = st.id
         WHERE st.project_id = ?
           AND s.instrument IN ({placeholders})
           AND s.is_flagged = 0
           AND s.data_file IS NOT NULL
           AND st.teff BETWEEN ? AND ?
         ORDER BY s.star_id, s.id
    """
    rows = con.execute(sql, (project_id, *INSTRUMENTS, *STAR_TEFF_RANGE)).fetchall()
    return [SpectrumRow(r["sid"], r["star"], r["df"], r["inst"]) for r in rows]


def _quality(row: SpectrumRow) -> float | None:
    """Median SNR in the fitting window, or ``None`` if the spectrum is unusable."""
    try:
        wl, flux, err = asd.read_spectrum(row.data_file)
    except (OSError, asd.AsdError):
        return None
    if wl.size == 0 or err.size != wl.size:
        return None
    if wl.min() > BLUE_REQUIRED or wl.max() < RED_REQUIRED:
        return None
    sel = (wl > SNR_WINDOW[0]) & (wl < SNR_WINDOW[1]) & (err > 0) & np.isfinite(flux)
    if sel.sum() < MIN_POINTS_IN_WINDOW:
        return None
    snr = float(np.median(flux[sel] / err[sel]))
    if not np.isfinite(snr) or snr < MIN_SNR:
        return None
    return snr


def build_manifests(db: Path, verbose: bool = True) -> dict:
    """Scan the database and choose the single- and multi-spectrum samples."""
    con = _connect(db)
    try:
        pid = _project_id(con, RVVD_PROJECT)
        if pid is None:
            raise RuntimeError(f"project {RVVD_PROJECT!r} not found in {db}")
        rows = _candidate_rows(con, pid)
    finally:
        con.close()

    if verbose:
        print(f"[selection] {len(rows)} candidate spectra in {RVVD_PROJECT}")

    by_star: dict[str, list[dict]] = {}
    checked = 0
    n_usable_stars = 0  # stars with >= SPECTRA_PER_MULTI good spectra
    for row in rows:
        # Stop scanning once both samples can certainly be filled.
        if n_usable_stars >= N_MULTI and len(by_star) >= N_MULTI + N_SINGLE:
            break
        snr = _quality(row)
        checked += 1
        if verbose and checked % 500 == 0:
            print(
                f"[selection] screened {checked} spectra, "
                f"{len(by_star)} stars ({n_usable_stars} with "
                f"{SPECTRA_PER_MULTI}+)...",
                flush=True,
            )
        if snr is None:
            continue
        bucket = by_star.setdefault(row.star_id, [])
        if len(bucket) + 1 == SPECTRA_PER_MULTI:
            n_usable_stars += 1
        bucket.append(
            {
                "spectrum_id": row.spectrum_id,
                "star_id": row.star_id,
                "data_file": row.data_file,
                "instrument": row.instrument,
                "snr": round(snr, 2),
            }
        )

    # Multi sample: stars with at least SPECTRA_PER_MULTI good spectra.
    multi_stars = sorted(
        (s for s, v in by_star.items() if len(v) >= SPECTRA_PER_MULTI)
    )[:N_MULTI]
    multi = [
        {"star_id": s, "spectra": by_star[s][:SPECTRA_PER_MULTI]} for s in multi_stars
    ]

    # Single sample: one spectrum per star, preferring stars not already used
    # by the multi sample so the two tests stay independent.
    used = set(multi_stars)
    single_pool = sorted(s for s in by_star if s not in used)
    single = [
        {"star_id": s, "spectra": [by_star[s][0]]} for s in single_pool[:N_SINGLE]
    ]
    if len(single) < N_SINGLE:  # fall back to reusing stars if the pool is thin
        for s in multi_stars:
            if len(single) >= N_SINGLE:
                break
            single.append({"star_id": s, "spectra": [by_star[s][0]]})

    manifests = {
        "real_single": {
            "description": "One good spectrum each from N distinct RVVD stars",
            "instruments": sorted(INSTRUMENTS),
            "cuts": {
                "min_snr": MIN_SNR,
                "blue_required": BLUE_REQUIRED,
                "red_required": RED_REQUIRED,
                "star_teff_range": list(STAR_TEFF_RANGE),
            },
            "cases": single,
        },
        "real_multi": {
            "description": (
                f"{SPECTRA_PER_MULTI} spectra of the same star, fitted jointly"
            ),
            "instruments": sorted(INSTRUMENTS),
            "cuts": {
                "min_snr": MIN_SNR,
                "blue_required": BLUE_REQUIRED,
                "red_required": RED_REQUIRED,
                "star_teff_range": list(STAR_TEFF_RANGE),
            },
            "cases": multi,
        },
    }
    if verbose:
        print(
            f"[selection] chose {len(single)} single-spectrum and "
            f"{len(multi)} multi-spectrum cases"
        )
    return manifests


def write_manifests(db: Path) -> None:
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    for name, body in build_manifests(db).items():
        path = MANIFEST_DIR / f"{name}.json"
        path.write_text(json.dumps(body, indent=1))
        print(f"[selection] wrote {path} ({len(body['cases'])} cases)")


def load_manifest(name: str) -> dict | None:
    path = MANIFEST_DIR / f"{name}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def cases_from_manifest(name: str, export_dir: Path, limit: int | None = None) -> list[Case]:
    """Export the manifest's spectra to ASCII and turn them into fit cases."""
    manifest = load_manifest(name)
    if manifest is None:
        raise FileNotFoundError(
            f"manifest {name!r} missing; run "
            f"'python -m gael_isis.selection --regenerate' first"
        )

    export_dir.mkdir(parents=True, exist_ok=True)
    cases: list[Case] = []
    settings = FitSettings()

    for entry in manifest["cases"][: limit or None]:
        refs = []
        for spec in entry["spectra"]:
            res_offset, res_slope = INSTRUMENTS[spec["instrument"]]
            out = export_dir / f"{spec['spectrum_id']}.txt"
            if not out.exists():
                wl, flux, err = asd.read_spectrum(spec["data_file"])
                asd.write_ascii3(out, wl, flux, err)
            refs.append(
                SpectrumRef(
                    label=spec["spectrum_id"],
                    path=str(out),
                    res_offset=res_offset,
                    res_slope=res_slope,
                )
            )
        cases.append(
            Case(
                case_id=entry["star_id"],
                kind="single" if len(refs) == 1 else "multi",
                spectra=refs,
                settings=settings,
            )
        )
    return cases


if __name__ == "__main__":
    import argparse

    from .config import load as load_config

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--regenerate", action="store_true", help="rewrite the manifests")
    ap.add_argument("--db", type=Path, default=None)
    args = ap.parse_args()

    cfg = load_config()
    db = args.db or cfg.astra_db
    if db is None:
        raise SystemExit("ASTRA database not found; pass --db")
    if args.regenerate:
        write_manifests(db)
    else:
        for n in ("real_single", "real_multi"):
            m = load_manifest(n)
            print(n, "->", "missing" if m is None else f"{len(m['cases'])} cases")
