"""The fit description shared by both backends, plus the result containers.

A :class:`Case` is the unit of comparison: one set of spectra fitted together,
once by ISIS and once by GAEL, with byte-identical inputs and equivalent
settings.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Parameters that are compared between the two codes.  ``vrad`` is untied
# (one value per spectrum); the rest are tied across the spectra of a case.
TIED_PARAMS = ("teff", "logg", "he")
UNTIED_PARAMS = ("vrad",)
COMPARED_PARAMS = TIED_PARAMS + UNTIED_PARAMS

# Human-facing units, used by the report.
UNITS = {"teff": "K", "logg": "dex", "he": "dex", "vrad": "km/s"}


@dataclass(frozen=True)
class SpectrumRef:
    """One observed (or mock) spectrum, already exported to 3-column ASCII."""

    label: str  # stable identity (DB uuid, or mock set/observation)
    path: str  # absolute path to the ASCII file
    res_offset: float  # R(lambda) = res_offset + res_slope * lambda
    res_slope: float


@dataclass
class FitSettings:
    """Everything both codes need to agree on, beyond the spectra themselves."""

    grid: str = "sdB/processed/"
    wave_cut: tuple[float, float] = (3600.0, 5250.0)
    ignore: tuple[tuple[float, float], ...] = (
        (3932.0, 3935.0),  # Ca II K (interstellar)
        (3967.0, 3970.0),  # Ca II H (interstellar)
        (4610.0, 4655.0),  # detector / reduction artefacts
        (5888.0, 5892.0),  # Na I D1 (interstellar)
        (5894.0, 5898.0),  # Na I D2 (interstellar)
    )
    anchors: tuple[tuple[float, float, float], ...] = (
        (3000.0, 3850.0, 50.0),
        (3850.0, 4050.0, 75.0),
        (4050.0, 4400.0, 75.0),
        (4400.0, 5200.0, 100.0),
        (5200.0, 15050.0, 100.0),
    )
    # name -> (start value, frozen)
    initial: tuple[tuple[str, float, bool], ...] = (
        ("vrad", 0.0, False),
        ("vsini", 7.0, True),
        ("zeta", 0.0, True),
        ("teff", 25000.0, False),
        ("logg", 5.5, False),
        ("xi", 0.0, True),
        ("z", 0.0, True),
        ("HE", -2.0, False),
    )
    untie: tuple[str, ...] = ("vrad",)
    filter_snr: float = 5.0
    auto_freeze_vsini: bool = True

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class Case:
    case_id: str
    kind: str  # "single" | "multi"
    spectra: list[SpectrumRef]
    settings: FitSettings = field(default_factory=FitSettings)
    truth: dict | None = None  # mock data only: the generating parameters

    @property
    def n_spectra(self) -> int:
        return len(self.spectra)


# ------------------------------------------------------------------ results
@dataclass
class ParamResult:
    value: float
    error: float = 0.0
    frozen: bool = False
    at_bound: bool = False
    # Grid/parameter bounds as reported by ISIS.  GAEL's CSV does not expose
    # them, so the comparison reuses these to flag GAEL fits that railed
    # against the same edge -- both codes fit the same grid, so they are exact.
    bound_lo: float | None = None
    bound_hi: float | None = None

    def to_json(self) -> dict:
        return {
            "value": self.value,
            "error": self.error,
            "frozen": self.frozen,
            "at_bound": self.at_bound,
            "bound_lo": self.bound_lo,
            "bound_hi": self.bound_hi,
        }

    @staticmethod
    def from_json(d: dict) -> "ParamResult":
        def opt(key):
            v = d.get(key)
            return None if v is None else float(v)

        return ParamResult(
            float(d["value"]),
            float(d.get("error", 0.0)),
            bool(d.get("frozen", False)),
            bool(d.get("at_bound", False)),
            opt("bound_lo"),
            opt("bound_hi"),
        )


@dataclass
class FitOutcome:
    """Normalised result of one fit, from either backend."""

    ok: bool
    backend: str
    error: str = ""
    tied: dict[str, ParamResult] = field(default_factory=dict)
    # one dict per spectrum, in submission order
    per_spectrum: list[dict[str, ParamResult]] = field(default_factory=list)
    n_rejected: int = 0
    seconds: float = 0.0

    def to_json(self) -> dict:
        return {
            "ok": self.ok,
            "backend": self.backend,
            "error": self.error,
            "tied": {k: v.to_json() for k, v in self.tied.items()},
            "per_spectrum": [
                {k: v.to_json() for k, v in d.items()} for d in self.per_spectrum
            ],
            "n_rejected": self.n_rejected,
            "seconds": self.seconds,
        }

    @staticmethod
    def from_json(d: dict) -> "FitOutcome":
        return FitOutcome(
            ok=bool(d["ok"]),
            backend=str(d.get("backend", "?")),
            error=str(d.get("error", "")),
            tied={k: ParamResult.from_json(v) for k, v in d.get("tied", {}).items()},
            per_spectrum=[
                {k: ParamResult.from_json(v) for k, v in entry.items()}
                for entry in d.get("per_spectrum", [])
            ],
            n_rejected=int(d.get("n_rejected", 0)),
            seconds=float(d.get("seconds", 0.0)),
        )


def file_digest(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
