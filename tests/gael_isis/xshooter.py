"""Cases for the two features that the sdB suites cannot reach: element
abundances and more than one stellar component.

Both need a metal-bearing grid, so these suites run on the Feros grids rather
than ``sdB/processed/``, and the spectra are X-Shooter-like (UVB + VIS arms,
R = 5400 / 8900) produced by ``gen_mock_xshooter``.  As everywhere else in
this harness the reference is ISIS's ``spectroscopy_automated``; the mock's
known truth is reported but not gated.

Two scoping decisions worth knowing:

*Every element the grid resolves is modelled*, at the middle of its own axis,
and only the ones under test are free.  Switching the rest off (ISIS's
"positive abundance" convention) is not usable here for two reasons:
spectroscopy_automated seeds parameters with a bare ``set_par``, which rejects
a value outside the soft limits stellar_default gives an abundance; and ISIS
builds its union wavelength grid over *all* species in the grid whatever their
abundance, so switching one off in GAEL alone would have the two codes
comparing different models.

*Tellurics are absent.*  ISIS's telluric model scales PWV by a factor of ten
against a library whose nodes are PWV*10, so over most of its own allowed
range it extrapolates to negative transmission; comparing GAEL's telluric
component against it would compare against a known-broken model.  The telluric
component is validated separately, directly against ``interpol_telluric`` and
against an independent convolution of the ESO library.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from .jobs import Case, FitSettings, SpectrumRef

# The Feros grids, and coverage that each case has to stay inside.
#   Feros_3: 21000-26000 K, log g 4.8-5.6, log n(He) -3.5..-1.0
#   Feros_5: 25000-29000 K, log g 5.2-5.6, log n(He) -2.5..-1.0
GRID_A = "Feros_3/processed/"
GRID_B = "Feros_5/processed/"

# Elements left free in the abundance suite.  Iron and silicon carry the most
# lines in this temperature range, so they are the ones a real fit would use.
FREE_ELEMENTS = ("FE", "SI")
ARMS = ("uvb", "vis")


def _gen_bin() -> Path | None:
    p = os.environ.get("GAEL_TEST_XSHOOTER_GEN")
    if p and Path(p).exists():
        return Path(p)
    # Next to the GAEL binary, which CMake always passes.
    g = os.environ.get("GAEL_TEST_GAEL_BIN")
    if g:
        cand = Path(g).parent / "gen_mock_xshooter"
        if cand.exists():
            return cand
    return None


def _truth_grid(kind: str, i: int) -> dict:
    """Deterministic, spread over each grid's interior (never on an edge).

    Keys are component-qualified ("c1_teff", "c2_sur_ratio"), i.e. the names
    both codes report, so ``compare.truth_statistics`` can line the truth up
    with the fitted values.
    """
    # Small prime-ish strides so consecutive cases are not correlated.
    t1 = 22200.0 + (i * 431) % 2600          # 22200 .. 24800, inside Feros_3
    g1 = 5.00 + ((i * 7) % 5) * 0.10         # 5.00 .. 5.40
    h1 = -2.60 + ((i * 3) % 6) * 0.25        # -2.60 .. -1.35
    out = {
        "c1_teff": t1, "c1_logg": g1, "c1_he": h1,
        "c1_vsini": 8.0 + (i % 5) * 6.0,     # 8 .. 32 km/s
        "c1_vrad": -60.0 + (i * 17) % 120,
    }
    if kind == "binary":
        out.update({
            "c2_teff": 26200.0 + (i * 313) % 2400,   # inside Feros_5
            "c2_logg": 5.25 + ((i * 5) % 4) * 0.08,  # 5.25 .. 5.49
            "c2_he": -2.30 + ((i * 11) % 5) * 0.25,  # -2.30 .. -1.30
            "c2_vsini": 15.0 + (i % 4) * 10.0,
            "c2_vrad": 40.0 - (i * 23) % 130,
            "c2_sur_ratio": 0.35 + (i % 5) * 0.15,   # 0.35 .. 0.95
        })
    else:
        for e in FREE_ELEMENTS:
            # Interior of the element's own axis (all are 3 nodes wide).
            out[f"c1_{e}"] = (
                {"FE": -5.30, "SI": -5.20}[e] + ((i * 13) % 7) * 0.10
            )
    return out


def _grid_base(cfg) -> str | None:
    """The configured base path that actually holds the Feros grids.

    ``cfg.base_paths`` starts with "./" for ISIS's benefit, which is not where
    the grids live; the generator needs the real one.
    """
    for base in cfg.base_paths:
        if (Path(base) / GRID_A / "grid.fits").exists():
            return str(base)
    return None


def _generate(cfg, kind: str, i: int, out_dir: Path) -> dict | None:
    gen = _gen_bin()
    base = _grid_base(cfg)
    if gen is None or base is None:
        return None
    t = _truth_grid(kind, i)
    cmd = [str(gen), "-o", str(out_dir), "--base", base,
           "--grid", GRID_A, "--seed", str(90000 + i),
           "--teff", str(t["c1_teff"]), "--logg", str(t["c1_logg"]),
           "--he", str(t["c1_he"]), "--vsini", str(t["c1_vsini"]),
           "--vrad", str(t["c1_vrad"])]
    if kind == "binary":
        cmd += ["--grid2", GRID_B,
                "--teff2", str(t["c2_teff"]), "--logg2", str(t["c2_logg"]),
                "--he2", str(t["c2_he"]), "--vsini2", str(t["c2_vsini"]),
                "--vrad2", str(t["c2_vrad"]),
                "--sur-ratio", str(t["c2_sur_ratio"])]
    # Every element is modelled, at its axis midpoint unless the case gives it
    # a truth value; the fit config below seeds the identical set.
    mids = _species_axes(cfg, GRID_A)
    for e, mid in mids.items():
        if kind != "binary" and e in FREE_ELEMENTS:
            cmd += ["--abundance", f"{e}={t[f'c1_{e}']}", "--fit-abundance", e]
        else:
            cmd += ["--abundance", f"{e}={mid}"]

    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if res.returncode != 0:
        return None
    return t


def _inside(value: float, span: tuple[float, float] | None) -> float:
    """Pull a seed back inside the axis it belongs to.

    The seeds below are deliberately offset from the truth so the fit has to
    travel, and near a grid edge that offset can walk off the grid.  That is
    not a weaker test case, it is no test case at all: ISIS's
    spectroscopy_automated seeds parameters with a bare ``set_par``, which
    rejects an out-of-range value and aborts the whole fit.  Six of the twelve
    binary cases died that way -- the secondary sits on Feros_5, whose log g
    axis stops at 5.6 and whose He axis stops at -2.5, and the offsets asked
    for 5.61 and -2.55.

    Inset by 2 % of the span so that neither code reports the *start* as
    railed at a grid edge.
    """
    if span is None:
        return value
    lo, hi = span
    if not (hi > lo):
        return value
    inset = 0.02 * (hi - lo)
    return min(max(value, lo + inset), hi - inset)


def _component_initial(cfg, kind: str, truth: dict, species: list[str],
                       bounds: tuple[dict, ...] = ()):
    """Per-component (name, value, frozen), identical for both backends.

    Seeded away from the truth, but per component: two components started at
    the same point cannot separate, because the chi2 surface is symmetric
    under swapping them.

    ``bounds[i]`` is component i's own grid coverage, used to keep its seed on
    that grid -- the components can sit on grids with different coverage, so
    the clamp has to be per component and not over the intersection.
    """
    comps = []

    def block(idx: int, teff, logg, he, vsini, vrad, sur=None):
        sgn = 1.0 if idx == 0 else -1.0
        b = bounds[idx] if idx < len(bounds) else {}
        ent = [
            ("vrad", vrad + sgn * 12.0, False),
            ("vsini", max(1.0, vsini * 0.6), False),
            ("zeta", 0.0, True),
            ("teff", _inside(teff + sgn * 900.0, b.get("teff")), False),
            ("logg", _inside(logg - sgn * 0.12, b.get("logg")), False),
            ("xi", 0.0, True),
            ("z", 0.0, True),
            ("HE", _inside(he + sgn * 0.25, b.get("he")), False),
        ]
        if sur is not None:
            ent.append(("sur_ratio", sur * 1.5, False))
        # Every element the grid resolves has to be named explicitly: ISIS
        # leaves them all free by default, GAEL freezes them by default, and
        # the two configurations only match if the case states each one.
        for sp, mid in species.items():
            if kind != "binary" and sp in FREE_ELEMENTS and idx == 0:
                ent.append((sp, mid, False))        # free, seeded at the midpoint
            else:
                ent.append((sp, mid, True))         # modelled, frozen
        return tuple(ent)

    comps.append(block(0, truth["c1_teff"], truth["c1_logg"], truth["c1_he"],
                       truth["c1_vsini"], truth["c1_vrad"]))
    if kind == "binary":
        comps.append(block(1, truth["c2_teff"], truth["c2_logg"],
                           truth["c2_he"], truth["c2_vsini"],
                           truth["c2_vrad"], sur=truth["c2_sur_ratio"]))
    return tuple(comps)


def _stellar_bounds(cfg, grid: str) -> dict[str, tuple[float, float]]:
    """``{teff/logg/he: (min, max)}`` of one grid, from its grid.fits.

    Only the three axes a seed is offset along; xi and z are frozen at 0 and
    every grid here pins them there anyway.
    """
    from astropy.io import fits

    names = {"t": "teff", "g": "logg", "HHE": "he"}
    for base in cfg.base_paths:
        p = Path(base) / grid / "grid.fits"
        if not p.exists():
            continue
        out: dict[str, tuple[float, float]] = {}
        with fits.open(p) as h:
            data = h[1].data
            for col in h[1].columns:
                if col.name in names:
                    v = data[col.name][0]
                    out[names[col.name]] = (float(min(v)), float(max(v)))
        return out
    return {}


def _species_of(cfg, grid: str) -> list[str]:
    """Element axes of a grid, read straight from its grid.fits."""
    return list(_species_axes(cfg, grid).keys())


def _species_axes(cfg, grid: str) -> dict[str, float]:
    """``{element: middle of its abundance axis}``.

    The midpoint is ISIS's stellar_default seed for an abundance, so an
    element nobody names ends up at the same value in both codes.
    """
    from astropy.io import fits

    for base in cfg.base_paths:
        p = Path(base) / grid / "grid.fits"
        if not p.exists():
            continue
        out: dict[str, float] = {}
        with fits.open(p) as h:
            data = h[1].data
            for col in h[1].columns:
                if col.name in ("t", "g", "x", "z", "HHE", "HE3", "HE4"):
                    continue
                vals = data[col.name][0]
                out[col.name] = 0.5 * (float(vals[0]) + float(vals[-1]))
        return out
    return {}


def cases(cfg, kind: str, limit: int | None = None):
    """Build the X-Shooter cases for ``kind`` in ("metal", "binary")."""
    n = limit if limit is not None else (12 if kind == "metal" else 12)
    root = Path(cfg.cache_dir) / "xshooter" / kind
    root.mkdir(parents=True, exist_ok=True)

    species = _species_axes(cfg, GRID_A)
    if not species:
        return []

    out = []
    for i in range(n):
        d = root / f"{i:03d}"
        truth = None
        # Regenerate only when missing: the spectra are the cache key for the
        # ISIS reference, so they must be bit-stable across runs.
        if (d / "truth.json").exists() and all(
                (d / f"spectrum_{a}.txt").exists() for a in ARMS):
            truth = _truth_grid(kind, i)
        else:
            truth = _generate(cfg, kind, i, d)
        if truth is None:
            continue

        grids = (GRID_A, GRID_B) if kind == "binary" else (GRID_A,)
        comp_init = _component_initial(
            cfg, kind, truth, species,
            bounds=tuple(_stellar_bounds(cfg, g) for g in grids))

        tied = ["c1_teff", "c1_logg", "c1_he"]
        untied = ["c1_vrad"]
        if kind == "binary":
            tied += ["c2_teff", "c2_logg", "c2_he", "c2_sur_ratio"]
            untied += ["c2_vrad"]
        else:
            tied += [f"c1_{e}" for e in FREE_ELEMENTS]

        settings = FitSettings(
            grid=GRID_A,
            grids=grids,
            component_initial=comp_init,
            tied_params=tuple(tied),
            untied_params=tuple(untied),
            wave_cut=(3605.0, 9395.0),
            ignore=(),
            anchors=((3000.0, 9500.0, 60.0),),
            untie=("vrad",),
            filter_snr=0.0,
            auto_freeze_vsini=False,
            # Both codes must actually fit two components: ISIS retires the
            # secondary by default at any sensible surface ratio, and GAEL's
            # equivalent is off by default.
            auto_freeze_sur_ratio=False if kind == "binary" else None,
        )

        spectra = [
            SpectrumRef(
                label=f"{kind}-{i:03d}-{arm}",
                path=str((d / f"spectrum_{arm}.txt").resolve()),
                res_offset=5400.0 if arm == "uvb" else 8900.0,
                res_slope=0.0,
            )
            for arm in ARMS
        ]
        out.append(Case(case_id=f"xs_{kind}_{i:03d}", kind=kind,
                        spectra=spectra, settings=settings, truth=truth))
    return out
