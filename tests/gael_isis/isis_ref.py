"""Generate, run, parse and cache ISIS ``spectroscopy_automated`` reference fits.

The generated script mirrors ASTRA's ``IsisBackend::generateScript`` so that the
reference really is "what ISIS would have produced for this star", with two
deliberate differences:

``plot_pdf = 0``
    Skips the xfig/LaTeX/PDF stage.  That is ~3x faster (6s vs 18s for a single
    spectrum) and removes the TeX toolchain from the test's dependencies.  All
    fitted numbers survive in ``results_conf.fits``.
``error_estimation = 0``
    Uncertainties come from the covariance matrix rather than ``conf_loop``,
    matching how GAEL is normally run.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from .jobs import Case, FitOutcome, ParamResult, file_digest, stable_json

# Bump when the generated script or the parsing changes in a way that
# invalidates previously cached results.
CACHE_VERSION = "isis-ref-v1"

DEFAULT_TIMEOUT = int(os.environ.get("GAEL_TEST_ISIS_TIMEOUT", "3600"))

_STELLAR_RE = re.compile(r"^stellar\((\d+)\)\.d(\d+)_(?:c(\d+)_)?(.+)$")
_REJECT_RE = re.compile(r"^Ignoring spectrum\s+(\d+)\s*\(", re.MULTILINE)
_WS_RE = re.compile(r"\s+")


# --------------------------------------------------------------- script gen
def build_script(case: Case, filenames: list[str], base_paths: list[str]) -> str:
    s = case.settings

    names, values, freeze = [], [], []
    for pname, pvalue, pfrozen in s.initial:
        names.append(f'"{pname}"')
        values.append(_num(pvalue))
        freeze.append("1" if pfrozen else "0")

    ignore = ",".join(f"{{{_num(a)},{_num(b)}}}" for a, b in s.ignore)
    anchors = ",".join(f"[{_num(a)}:{_num(b)}:{_num(c)}]" for a, b, c in s.anchors)

    entries = []
    for ref, fname in zip(case.spectra, filenames):
        fields = [
            f'     filename = "{fname}"',
            '     spectype = "ASCII_with_3_columns"',
        ]
        # S-Lang has no empty-array literal, so these are omitted rather than
        # written as "[]" when nothing is masked (mock spectra need no mask).
        if ignore:
            fields.append(f"     ignore = [{ignore}]")
        if anchors:
            fields.append(f"     cspline_anchorpoints = [{anchors}]")
        fields += [
            f"     res_offset = {_num(ref.res_offset)}",
            f"     res_slope  = {_num(ref.res_slope)}",
            f"     wave_cut = [{_num(s.wave_cut[0])},{_num(s.wave_cut[1])}]",
        ]
        entries.append("   struct{\n" + ",\n".join(fields) + "\n   }")

    untie = ",".join(f'"{u}"' for u in s.untie)
    bpaths = ",\n                   ".join(f'"{p}"' for p in ["./"] + base_paths)

    return f"""require("stellar_isisscripts.sl");
variable tscript_start = _ftime;

variable modelgrid = ["{s.grid}"];

variable initial_guess_params_values = struct{{
    name   = [{", ".join(names)}],
    value  = [{", ".join(values)}],
    freeze = [{", ".join(freeze)}] }};

variable input =
  [
{",\n".join(entries)}
  ];

variable qualies_for_fit = struct{{
  xrange             = 500.,
  error_estimation   = 0,
  auto_freeze_vsini  = {1 if s.auto_freeze_vsini else 0},
  add_telluric_model = 0,
  apply_mask         = 0,
  xfig_ignore        = -1,
  plot_pdf           = 0,
  filter_snr         = {_num(s.filter_snr)},
  save_model         = 0,
  untie = {{{untie}}}
}};

variable bpaths = [{bpaths}];
qualies_for_fit = struct_combine(qualies_for_fit, struct{{bpaths=bpaths}});
modelgrid = search_grid_fit_photometry(bpaths, modelgrid, "grid.fits");

variable sout = spectroscopy_automated(input, modelgrid, initial_guess_params_values;;
                                       qualies_for_fit);
vmessage(sprintf("- script completed in %.1fs", _ftime - tscript_start));
exit;
"""


def _num(x) -> str:
    """Format a number stably (no locale, no float noise) for the script."""
    if isinstance(x, bool):
        return "1" if x else "0"
    f = float(x)
    return str(int(f)) if f == int(f) and abs(f) < 1e15 else repr(f)


# ------------------------------------------------------------------- cache
def cache_key(case: Case, base_paths: list[str]) -> str:
    digests = [file_digest(ref.path) for ref in case.spectra]
    # Filenames are normalised to s1.txt.. so the key is path-independent.
    script = build_script(
        case, [f"s{i + 1}.txt" for i in range(case.n_spectra)], base_paths
    )
    payload = stable_json(
        {
            "version": CACHE_VERSION,
            "script": script,
            "spectra": [
                {
                    "digest": d,
                    "res_offset": r.res_offset,
                    "res_slope": r.res_slope,
                }
                for d, r in zip(digests, case.spectra)
            ],
        }
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def cache_path(cache_dir: Path, key: str) -> Path:
    return cache_dir / "isis" / key[:2] / f"{key}.json"


def load_cached(cache_dir: Path, key: str) -> FitOutcome | None:
    p = cache_path(cache_dir, key)
    if not p.exists():
        return None
    try:
        return FitOutcome.from_json(json.loads(p.read_text())["result"])
    except (OSError, ValueError, KeyError):
        return None  # corrupt entry -> recompute


def store_cached(cache_dir: Path, key: str, case: Case, outcome: FitOutcome) -> None:
    p = cache_path(cache_dir, key)
    p.parent.mkdir(parents=True, exist_ok=True)
    body = {
        "version": CACHE_VERSION,
        "case_id": case.case_id,
        "kind": case.kind,
        "n_spectra": case.n_spectra,
        "spectra": [r.label for r in case.spectra],
        "result": outcome.to_json(),
    }
    # Atomic: concurrent workers must never observe a half-written file.
    tmp = p.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(body, indent=1))
    os.replace(tmp, p)


# --------------------------------------------------------------- execution
def run(case: Case, cfg, work_root: Path | None = None) -> FitOutcome:
    """Run ISIS for *case* in a throwaway directory and parse the results."""
    started = time.time()
    workdir = Path(tempfile.mkdtemp(prefix="isisfit-", dir=work_root))
    try:
        filenames = []
        for i, ref in enumerate(case.spectra):
            name = f"s{i + 1}.txt"
            shutil.copyfile(ref.path, workdir / name)
            filenames.append(name)

        script = build_script(case, filenames, list(cfg.base_paths))
        (workdir / "fit.sl").write_text(script)

        from .config import isis_env

        proc = subprocess.run(
            [str(cfg.isis_bin), "fit.sl"],
            cwd=workdir,
            env=isis_env(cfg.isis_bin),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=DEFAULT_TIMEOUT,
        )
        log = proc.stdout.decode("utf-8", "replace")
        (workdir / "isis.log").write_text(log)

        n_rejected = len(set(_REJECT_RE.findall(log)))

        if proc.returncode != 0:
            return FitOutcome(
                ok=False,
                backend="isis",
                error=_first_error(log) or f"isis exited with {proc.returncode}",
                n_rejected=n_rejected,
                seconds=time.time() - started,
            )

        outcome = parse_outputs(workdir, case)
        outcome.n_rejected = n_rejected
        outcome.seconds = time.time() - started
        return outcome

    except subprocess.TimeoutExpired:
        return FitOutcome(
            ok=False,
            backend="isis",
            error=f"isis timed out after {DEFAULT_TIMEOUT}s",
            seconds=time.time() - started,
        )
    except Exception as exc:  # noqa: BLE001 - reported, never fatal
        return FitOutcome(
            ok=False,
            backend="isis",
            error=f"{type(exc).__name__}: {exc}",
            seconds=time.time() - started,
        )
    finally:
        if os.environ.get("GAEL_TEST_KEEP_WORKDIRS"):
            print(f"[isis] kept {workdir}")
        else:
            shutil.rmtree(workdir, ignore_errors=True)


def _first_error(log: str) -> str:
    for line in log.splitlines():
        low = line.lower()
        if "error" in low or "failed" in low or "usage" in low:
            return line.strip()[:300]
    return ""


# ----------------------------------------------------------------- parsing
def parse_outputs(workdir: Path, case: Case) -> FitOutcome:
    params = _parse_params_dat(workdir / "spectroscopy_spectrum_params.dat")
    if not params:
        return FitOutcome(
            ok=False,
            backend="isis",
            error="spectroscopy_spectrum_params.dat missing or unparsable",
        )
    conf = _parse_results_conf(workdir / "results_conf.fits")

    def build(spec_idx: int, pname: str) -> ParamResult | None:
        entry = params.get((spec_idx, pname.lower()))
        if entry is None:
            return None
        value, frozen, lo, hi = entry
        error = 0.0
        c = conf.get((spec_idx, pname.lower()))
        if c is not None:
            value = c["value"]  # identical, but full precision from the FITS
            error = 0.5 * (c["conf_max"] - c["conf_min"])
            lo, hi = c["min"], c["max"]
        finite = hi > lo and abs(lo) < 1e30 and abs(hi) < 1e30
        return ParamResult(
            value=value,
            error=error,
            frozen=frozen,
            at_bound=_at_bound(value, lo, hi),
            bound_lo=lo if finite else None,
            bound_hi=hi if finite else None,
        )

    from .jobs import TIED_PARAMS, UNTIED_PARAMS

    tied = {}
    for p in TIED_PARAMS:
        r = build(1, p)
        if r is not None:
            tied[p] = r

    n_found = max((idx for idx, _ in params), default=0)
    per_spectrum = []
    for d in range(1, n_found + 1):
        entry = {}
        for p in UNTIED_PARAMS:
            r = build(d, p)
            if r is not None:
                entry[p] = r
        per_spectrum.append(entry)

    if not tied:
        return FitOutcome(
            ok=False, backend="isis", error="no stellar parameters in ISIS output"
        )

    return FitOutcome(ok=True, backend="isis", tied=tied, per_spectrum=per_spectrum)


def _at_bound(value: float, lo: float, hi: float) -> bool:
    """True when the fit railed against a grid edge (so the value is not a fit)."""
    if not (hi > lo) or abs(lo) > 1e30 or abs(hi) > 1e30:
        return False
    eps = 1e-6 * (hi - lo)
    return value <= lo + eps or value >= hi - eps


def _parse_params_dat(path: Path) -> dict[tuple[int, str], tuple[float, bool, float, float]]:
    """``{(spectrum_index, param): (value, frozen, min, max)}`` from the .dat file."""
    out: dict[tuple[int, str], tuple[float, bool, float, float]] = {}
    if not path.exists():
        return out
    for line in path.read_text(errors="replace").splitlines():
        tok = _WS_RE.split(line.strip())
        if len(tok) < 7 or not tok[0].isdigit():
            continue
        m = _STELLAR_RE.match(tok[1])
        if not m:
            continue
        spec_idx = int(m.group(2))
        pname = m.group(4).lower()
        try:
            frozen = tok[3] != "0"
            value, lo, hi = float(tok[4]), float(tok[5]), float(tok[6])
        except ValueError:
            continue
        out[(spec_idx, pname)] = (value, frozen, lo, hi)
    return out


def _parse_results_conf(path: Path) -> dict[tuple[int, str], dict]:
    """Free-parameter values + confidence intervals from ``results_conf.fits``."""
    out: dict[tuple[int, str], dict] = {}
    if not path.exists():
        return out
    from astropy.io import fits

    with fits.open(path) as hdul:
        if len(hdul) < 2 or hdul[1].data is None:
            return out
        for row in hdul[1].data:
            name = str(row["name"]).strip()
            m = _STELLAR_RE.match(name)
            if not m:
                continue
            out[(int(m.group(2)), m.group(4).lower())] = {
                "value": float(row["value"]),
                "min": float(row["min"]),
                "max": float(row["max"]),
                "conf_min": float(row["conf_min"]),
                "conf_max": float(row["conf_max"]),
            }
    return out


# ------------------------------------------------------------------- entry
def get_or_run(case: Case, cfg, work_root: Path | None = None) -> tuple[FitOutcome, bool]:
    """Return ``(outcome, was_cached)``, running ISIS only on a cache miss."""
    key = cache_key(case, list(cfg.base_paths))
    cached = load_cached(cfg.cache_dir, key)
    if cached is not None:
        return cached, True
    if not cfg.allow_isis or not cfg.isis_bin:
        return (
            FitOutcome(
                ok=False,
                backend="isis",
                error="no cached ISIS reference and ISIS is unavailable",
            ),
            False,
        )
    outcome = run(case, cfg, work_root)
    # Failures are cached too: they are reproducible properties of the input
    # (rejected by the SNR filter, outside the grid, ...) and re-running ISIS
    # for them on every invocation would defeat the point of the cache.
    store_cached(cfg.cache_dir, key, case, outcome)
    return outcome, False
