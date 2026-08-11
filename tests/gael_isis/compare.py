"""Turn paired GAEL/ISIS fits into population statistics and a pass/fail verdict.

The gate is deliberately statistical rather than per-fit.  Individual spectra
can legitimately land in a different local chi^2 minimum in either code, so a
single hard-tolerance failure says little; a shifted *median* or an inflated
*scatter* across 100 fits is what actually signals a regression.

Metrics per parameter, over the "clean" subset (see :func:`collect`):

  bias     median(GAEL - ISIS)              -- systematic offset
  scatter  1.4826 * median(|d - median(d)|) -- robust sigma (NMAD)
  outliers fraction with |d - bias| > 5*scatter
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import TESTS_DIR
from .jobs import COMPARED_PARAMS, TIED_PARAMS, UNITS, UNTIED_PARAMS

TOLERANCE_FILE = TESTS_DIR / "tolerances.json"

OUTLIER_NSIGMA = 5.0


@dataclass
class Sample:
    """One comparable number pair."""

    case_id: str
    param: str
    gael: float
    isis: float
    spectrum: str = ""

    @property
    def delta(self) -> float:
        return self.gael - self.isis


@dataclass
class Collection:
    samples: list[Sample] = field(default_factory=list)
    n_cases: int = 0
    n_pairs_ok: int = 0
    n_isis_failed: int = 0
    n_gael_failed: int = 0
    n_rejected_spectra: int = 0
    n_at_bound: int = 0
    n_frozen: int = 0
    isis_cached: int = 0
    isis_computed: int = 0
    failures: list[str] = field(default_factory=list)

    def deltas(self, param: str) -> np.ndarray:
        return np.array(
            [s.delta for s in self.samples if s.param == param], dtype=float
        )


def _bound_flag(value: float, lo, hi) -> bool:
    if lo is None or hi is None or not (hi > lo):
        return False
    eps = 1e-6 * (hi - lo)
    return value <= lo + eps or value >= hi - eps


def collect(results) -> Collection:
    """Build the comparable sample set from ``[(case, isis, gael, cached), ...]``.

    A case contributes only when *both* codes succeeded and neither dropped a
    spectrum, so that untied parameters stay aligned by index.  A parameter is
    additionally skipped when either code froze it or railed against a grid
    edge: neither situation is a fit result, and including them would let a
    real regression hide behind a pile of identical boundary values.
    """
    out = Collection()
    for case, isis, gael, cached in results:
        out.n_cases += 1
        out.isis_cached += 1 if cached else 0
        out.isis_computed += 0 if cached else 1

        if not isis.ok:
            out.n_isis_failed += 1
            out.failures.append(f"{case.case_id}: ISIS: {isis.error}")
            continue
        if not gael.ok:
            out.n_gael_failed += 1
            out.failures.append(f"{case.case_id}: GAEL: {gael.error}")
            continue
        if isis.n_rejected or gael.n_rejected:
            out.n_rejected_spectra += max(isis.n_rejected, gael.n_rejected)
            continue
        if len(isis.per_spectrum) != len(gael.per_spectrum):
            out.failures.append(
                f"{case.case_id}: spectrum count mismatch "
                f"(ISIS {len(isis.per_spectrum)}, GAEL {len(gael.per_spectrum)})"
            )
            continue

        out.n_pairs_ok += 1

        def usable(a, b) -> bool:
            """Both codes actually *fitted* this parameter."""
            if a is None or b is None:
                return False
            # Boundary first: a code that rails at a grid edge also reports the
            # parameter as frozen with zero error, and calling that "frozen"
            # would hide the real reason it is not comparable.
            if a.at_bound or _bound_flag(b.value, a.bound_lo, a.bound_hi):
                out.n_at_bound += 1
                return False
            if a.frozen or b.frozen:
                out.n_frozen += 1
                return False
            return True

        # The parameter set is a property of the case: the abundance and
        # multi-component suites compare component-qualified names, and which
        # elements are free differs per suite.
        for p in case.settings.tied():
            a, b = isis.tied.get(p), gael.tied.get(p)
            if usable(a, b):
                out.samples.append(Sample(case.case_id, p, b.value, a.value))

        for idx, (ai, bi) in enumerate(zip(isis.per_spectrum, gael.per_spectrum)):
            for p in case.settings.untied():
                a, b = ai.get(p), bi.get(p)
                if usable(a, b):
                    label = (
                        case.spectra[idx].label if idx < len(case.spectra) else str(idx)
                    )
                    out.samples.append(
                        Sample(case.case_id, p, b.value, a.value, label)
                    )

    return out


def nmad(x: np.ndarray) -> float:
    if x.size == 0:
        return float("nan")
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def _params_of(coll: Collection) -> list[str]:
    """Parameters actually present, in a stable order (COMPARED_PARAMS first
    so the legacy suites keep their familiar row order)."""
    seen = {s.param for s in coll.samples}
    head = [p for p in COMPARED_PARAMS if p in seen]
    tail = sorted(p for p in seen if p not in set(COMPARED_PARAMS))
    return head + tail


def statistics(coll: Collection) -> dict:
    stats = {}
    for p in _params_of(coll):
        d = coll.deltas(p)
        if d.size == 0:
            stats[p] = {"n": 0}
            continue
        bias = float(np.median(d))
        sc = nmad(d)
        frac = (
            float(np.mean(np.abs(d - bias) > OUTLIER_NSIGMA * sc))
            if sc > 0
            else 0.0
        )
        stats[p] = {
            "n": int(d.size),
            "bias": bias,
            "scatter": sc,
            "outlier_fraction": frac,
            "mean": float(np.mean(d)),
            "max_abs": float(np.max(np.abs(d))),
        }
    return stats


def truth_statistics(results) -> dict:
    """For mock data: how well each code recovers the generating parameters."""
    acc: dict[str, dict[str, list]] = {}
    for case, isis, gael, _ in results:
        if not case.truth:
            continue
        for p in TIED_PARAMS:
            true_val = case.truth.get(p)
            if true_val is None:
                continue
            for backend, res in (("isis", isis), ("gael", gael)):
                if not res.ok:
                    continue
                got = res.tied.get(p)
                if got is None or got.frozen or got.at_bound:
                    continue
                acc.setdefault(p, {}).setdefault(backend, []).append(
                    got.value - float(true_val)
                )
    out = {}
    for p, per_backend in acc.items():
        out[p] = {
            b: {
                "n": len(v),
                "bias": float(np.median(v)),
                "scatter": nmad(np.array(v, dtype=float)),
            }
            for b, v in per_backend.items()
        }
    return out


# ------------------------------------------------------------- tolerances
def load_tolerances() -> dict:
    if not TOLERANCE_FILE.exists():
        return {}
    return json.loads(TOLERANCE_FILE.read_text())


def check(suite: str, coll: Collection, stats: dict, tolerances: dict) -> tuple[bool, list[str]]:
    """Apply the calibrated thresholds. Returns ``(passed, messages)``."""
    problems: list[str] = []
    spec = (tolerances.get("suites") or {}).get(suite)
    if spec is None:
        return True, [
            f"no calibrated tolerances for suite {suite!r} -- reporting only "
            f"(run with --calibrate to create them)"
        ]

    min_pairs = spec.get("min_clean_cases", 0)
    if coll.n_pairs_ok < min_pairs:
        problems.append(
            f"only {coll.n_pairs_ok} comparable cases, expected at least "
            f"{min_pairs} (ISIS failed {coll.n_isis_failed}, "
            f"GAEL failed {coll.n_gael_failed})"
        )

    for p, limits in (spec.get("parameters") or {}).items():
        st = stats.get(p, {})
        if st.get("n", 0) < limits.get("min_n", 0):
            problems.append(
                f"{p}: only {st.get('n', 0)} comparable values, "
                f"expected at least {limits.get('min_n')}"
            )
            continue
        if st.get("n", 0) == 0:
            continue
        from .jobs import split_qualified
        unit = UNITS.get(p, UNITS.get(split_qualified(p)[1], ""))
        if abs(st["bias"]) > limits["max_abs_bias"]:
            problems.append(
                f"{p}: bias {st['bias']:+.4g} {unit} exceeds "
                f"{limits['max_abs_bias']:.4g} {unit}"
            )
        if st["scatter"] > limits["max_scatter"]:
            problems.append(
                f"{p}: scatter {st['scatter']:.4g} {unit} exceeds "
                f"{limits['max_scatter']:.4g} {unit}"
            )
        if st["outlier_fraction"] > limits["max_outlier_fraction"]:
            problems.append(
                f"{p}: outlier fraction {st['outlier_fraction']:.3f} exceeds "
                f"{limits['max_outlier_fraction']:.3f}"
            )

    return not problems, problems


# Smallest thresholds calibration will ever emit, per parameter.  A suite that
# happens to agree very tightly must not lock in a threshold so narrow that
# ordinary numerical noise trips it; differences below these are not worth
# failing a build over.  (bias, scatter) in the parameter's own unit.
THRESHOLD_FLOORS = {
    "teff": (25.0, 50.0),  # K
    "logg": (0.005, 0.010),  # dex
    "he": (0.005, 0.010),  # dex
    "vrad": (0.3, 0.5),  # km/s
}


def calibrate(suite: str, coll: Collection, stats: dict, margin: float = 2.0) -> dict:
    """Derive thresholds from an observed run, with head-room."""
    params = {}
    for p in _params_of(coll):
        st = stats.get(p, {})
        if not st.get("n"):
            continue
        # Floors are keyed by the bare parameter name, so a qualified one
        # ("c2_teff") inherits the floor of its kind; an element abundance is
        # a dex quantity and reuses the "he" floor.
        from .jobs import split_qualified
        _, bare = split_qualified(p)
        bias_floor, scatter_floor = THRESHOLD_FLOORS.get(
            bare, THRESHOLD_FLOORS.get("he", (0.0, 0.0)))
        params[p] = {
            "min_n": int(max(1, 0.6 * st["n"])),
            "max_abs_bias": _round(
                max(abs(st["bias"]) * margin, st["scatter"] * 0.5, bias_floor)
            ),
            "max_scatter": _round(max(st["scatter"] * margin, scatter_floor)),
            "max_outlier_fraction": round(
                min(0.25, max(0.05, st["outlier_fraction"] * 2 + 0.03)), 3
            ),
        }
    return {
        "min_clean_cases": int(0.7 * coll.n_pairs_ok),
        "parameters": params,
    }


def _round(x: float) -> float:
    """Round to two significant digits so the file stays readable."""
    if not np.isfinite(x) or x == 0:
        return float(x)
    from math import floor, log10

    digits = -int(floor(log10(abs(x)))) + 1
    return float(round(x, digits))


def write_tolerances(suite: str, spec: dict) -> None:
    doc = load_tolerances()
    doc.setdefault(
        "_comment",
        "Calibrated GAEL-vs-ISIS agreement thresholds. Regenerate with "
        "tests/test_*_vs_isis.py --calibrate. bias/scatter are in the "
        "parameter's own unit (K, dex, km/s).",
    )
    doc.setdefault("suites", {})[suite] = spec
    TOLERANCE_FILE.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")


# ----------------------------------------------------------------- report
def report(suite: str, coll: Collection, stats: dict, truth: dict | None = None) -> str:
    lines = []
    lines.append("=" * 72)
    lines.append(f"GAEL vs ISIS -- {suite}")
    lines.append("=" * 72)
    lines.append(
        f"cases: {coll.n_cases}   comparable: {coll.n_pairs_ok}   "
        f"ISIS failed: {coll.n_isis_failed}   GAEL failed: {coll.n_gael_failed}"
    )
    lines.append(
        f"ISIS references: {coll.isis_cached} cached, "
        f"{coll.isis_computed} newly computed"
    )
    lines.append(
        f"skipped: {coll.n_rejected_spectra} spectra rejected by a code, "
        f"{coll.n_at_bound} parameters railed at a grid edge, "
        f"{coll.n_frozen} frozen"
    )
    lines.append("")
    header = f"{'param':>6} {'n':>5} {'bias':>12} {'scatter':>12} {'max|d|':>12} {'outliers':>9}"
    lines.append(header)
    lines.append("-" * len(header))
    for p in _params_of(coll):
        st = stats.get(p, {})
        if not st.get("n"):
            lines.append(f"{p:>6} {0:>5}   (no comparable values)")
            continue
        lines.append(
            f"{p:>6} {st['n']:>5} {st['bias']:>+12.4g} {st['scatter']:>12.4g} "
            f"{st['max_abs']:>12.4g} {st['outlier_fraction']:>9.3f}   {UNITS.get(p, '')}"
        )

    if truth:
        lines.append("")
        lines.append("Recovery of the true (generating) parameters:")
        lines.append(
            f"{'param':>6} {'backend':>8} {'n':>5} {'bias':>12} {'scatter':>12}"
        )
        lines.append("-" * 48)
        for p, per_backend in truth.items():
            for b, st in sorted(per_backend.items()):
                lines.append(
                    f"{p:>6} {b:>8} {st['n']:>5} {st['bias']:>+12.4g} "
                    f"{st['scatter']:>12.4g}"
                )

    if coll.failures:
        lines.append("")
        lines.append(f"First failures ({len(coll.failures)} total):")
        for f in coll.failures[:15]:
            lines.append(f"  - {f}")
    return "\n".join(lines)


def write_artifacts(out_dir: Path, suite: str, coll: Collection, stats: dict, truth) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{suite}_summary.json").write_text(
        json.dumps(
            {
                "suite": suite,
                "n_cases": coll.n_cases,
                "n_comparable": coll.n_pairs_ok,
                "n_isis_failed": coll.n_isis_failed,
                "n_gael_failed": coll.n_gael_failed,
                "statistics": stats,
                "truth_recovery": truth,
            },
            indent=1,
        )
    )
    with open(out_dir / f"{suite}_deltas.csv", "w") as fh:
        fh.write("case_id,spectrum,parameter,gael,isis,delta\n")
        for s in coll.samples:
            fh.write(
                f"{s.case_id},{s.spectrum},{s.param},"
                f"{s.gael!r},{s.isis!r},{s.delta!r}\n"
            )
