"""Reproducible mock spectra via GAEL's own ``mock_data_generator``.

The generator is invoked with a fixed ``--seed``, which makes its output both
reproducible and independent of the thread count.  That is what lets the ISIS
reference cache survive across runs -- without it every invocation would
produce different spectra and every ISIS lookup would miss.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from .jobs import Case, FitSettings, SpectrumRef, stable_json

# Ranges chosen to sit comfortably inside the sdB grid (15000-55000 K,
# logg 4.6-6.6, He -5.05..-0.041) so that fits are not dominated by railing
# against grid edges.
MOCK_SPEC = {
    "wave_start": 3600.0,
    "wave_end": 5300.0,
    "res_offset": 1800.0,  # LAMOST/LRS-like constant resolving power
    "res_slope": 0.0,
    "teff": "22000:38000",
    "logg": "5.0:6.0",
    "he": "-3.0:-1.5",
    "vsini": "7",
    "zeta": "0",
    "xi": "0",
    "z": "0",
    "vrad": "0,60",
    "vrad_scatter": "-25:25",
    "noise": "1:4",  # per cent
    "continuum": "linear",
}

SINGLE_SEED = 20260809
MULTI_SEED = 20260810
N_SINGLE = 100
N_MULTI = 100
SPECTRA_PER_MULTI = 5


def _spec_digest(seed: int, n_sets: int, multiplicity: int, grid: str) -> str:
    payload = stable_json(
        {"spec": MOCK_SPEC, "seed": seed, "n": n_sets, "m": multiplicity, "grid": grid}
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def generate(cfg, seed: int, n_sets: int, multiplicity: int) -> Path:
    """Generate (or reuse) a mock data set; returns its directory."""
    digest = _spec_digest(seed, n_sets, multiplicity, cfg.grid)
    out_dir = cfg.cache_dir / "mockdata" / digest
    stamp = out_dir / ".complete"
    if stamp.exists():
        return out_dir

    if not cfg.mockgen_bin:
        raise RuntimeError("mock_data_generator binary not found")

    # Regenerate from scratch: a partial directory from an interrupted run
    # must never be mistaken for a good one.
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(cfg.global_settings, out_dir / "global_settings.json")

    # "--opt=value" throughout: several of these values start with '-' and
    # cxxopts rejects those in the separate-argument form.  xi/z/zeta are not
    # passed at all -- the generator already defaults them to "0", which is
    # what MOCK_SPEC records.
    cmd = [
        str(cfg.mockgen_bin),
        f"--seed={seed}",
        f"--num-sets={n_sets}",
        f"--multiplicity={multiplicity}",
        "--nyquist",
        f"--wave-start={MOCK_SPEC['wave_start']}",
        f"--wave-end={MOCK_SPEC['wave_end']}",
        f"--res-offset={MOCK_SPEC['res_offset']}",
        f"--res-slope={MOCK_SPEC['res_slope']}",
        f"--teff={MOCK_SPEC['teff']}",
        f"--logg={MOCK_SPEC['logg']}",
        f"--he={MOCK_SPEC['he']}",
        f"--vsini={MOCK_SPEC['vsini']}",
        f"--vrad={MOCK_SPEC['vrad']}",
        f"--vrad-scatter={MOCK_SPEC['vrad_scatter']}",
        f"--noise={MOCK_SPEC['noise']}",
        f"--continuum={MOCK_SPEC['continuum']}",
        f"--grid={cfg.grid}",
        f"--threads={max(1, cfg.jobs)}",
        f"--output={out_dir}",
    ]
    proc = subprocess.run(
        cmd, cwd=out_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    if proc.returncode != 0:
        log = proc.stdout.decode("utf-8", "replace")
        raise RuntimeError(f"mock_data_generator failed:\n{log[-3000:]}")

    stamp.write_text(json.dumps({"cmd": cmd[1:], "digest": digest}, indent=1))
    return out_dir


def cases(cfg, kind: str, limit: int | None = None) -> list[Case]:
    """Build the mock comparison cases for ``kind`` in {"single", "multi"}."""
    if kind == "single":
        data_dir = generate(cfg, SINGLE_SEED, N_SINGLE, 1)
    elif kind == "multi":
        data_dir = generate(cfg, MULTI_SEED, N_MULTI, SPECTRA_PER_MULTI)
    else:
        raise ValueError(kind)

    settings = FitSettings(
        grid=cfg.grid,
        wave_cut=(MOCK_SPEC["wave_start"], 5250.0),
        # Mock spectra contain no interstellar lines or detector artefacts,
        # so nothing needs masking; the anchor grid is kept as in the real fits.
        ignore=(),
    )

    out: list[Case] = []
    for set_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        meta_path = set_dir / "metadata.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        spec_files = sorted(set_dir.glob("spectrum_*.txt"))
        if not spec_files:
            continue

        truth = dict(meta.get("true_parameters", {}))
        # Per-observation radial velocities differ; keep them alongside.
        truth["vrad_per_spectrum"] = [
            s.get("vrad_actual") for s in meta.get("spectra", [])
        ]

        refs = [
            SpectrumRef(
                label=f"{set_dir.name}/{f.name}",
                path=str(f),
                res_offset=MOCK_SPEC["res_offset"],
                res_slope=MOCK_SPEC["res_slope"],
            )
            for f in spec_files
        ]
        out.append(
            Case(
                case_id=set_dir.name,
                kind=kind,
                spectra=refs,
                settings=settings,
                truth=truth,
            )
        )
        if limit and len(out) >= limit:
            break
    return out
