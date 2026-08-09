"""Locating binaries, data and caches.

Every path can be overridden from the environment, which is what the CTest
wrappers do.  Nothing here raises: callers inspect the returned object and skip
the test with a readable message when something is missing.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent.parent
REPO_DIR = TESTS_DIR.parent


def _env_path(name: str) -> Path | None:
    v = os.environ.get(name)
    return Path(v).expanduser() if v else None


def _first_existing(*candidates: Path | None) -> Path | None:
    for c in candidates:
        if c and c.exists():
            return c
    return None


def _expand(value: str) -> str:
    """Expand ``${VAR}`` / ``$VAR`` / ``~`` the way GAEL's own config loader does."""
    return os.path.expanduser(os.path.expandvars(value))


@dataclass
class Config:
    gael_bin: Path | None = None
    mockgen_bin: Path | None = None
    isis_bin: Path | None = None
    global_settings: Path | None = None
    base_paths: list[str] = field(default_factory=list)
    grid: str = "sdB/processed/"
    astra_db: Path | None = None
    cache_dir: Path = TESTS_DIR / "cache"
    work_dir: Path | None = None
    jobs: int = 1
    allow_isis: bool = True

    # ---------------------------------------------------------------- checks
    def grid_available(self) -> bool:
        return self.resolve_grid() is not None

    def resolve_grid(self) -> Path | None:
        """Find the directory holding ``grid.fits`` for :attr:`grid`."""
        for bp in self.base_paths:
            cand = Path(bp) / self.grid / "grid.fits"
            if cand.exists():
                return cand.parent
        return None

    def missing(self, *, need_isis: bool, need_db: bool) -> str | None:
        """Return a human-readable reason to skip, or ``None`` when runnable."""
        if not self.gael_bin:
            return "GAEL binary not found (set GAEL_TEST_GAEL_BIN)"
        if not self.global_settings:
            return "global_settings.json not found (set GAEL_TEST_GLOBAL_SETTINGS)"
        if not self.grid_available():
            return (
                f"model grid {self.grid!r} not found under any of the configured "
                f"basePaths: {self.base_paths}"
            )
        if need_isis and self.allow_isis and not self.isis_bin:
            return "ISIS binary not found (set GAEL_TEST_ISIS)"
        if need_db and not self.astra_db:
            return "ASTRA database not found (set GAEL_TEST_ASTRA_DB)"
        return None


def _normalise_base_paths(paths: list[str]) -> list[str]:
    """Expand, deduplicate and drop dead entries from the grid search path.

    ISIS builds grid locations by plain string concatenation, so every entry
    must end in a separator; and a literal ``${USER}`` reaching the generated
    script would simply never match.  Non-existent directories are dropped to
    keep the script (and the cache key) short and stable.
    """
    out: list[str] = []
    for raw in paths:
        p = _expand(raw)
        if not p:
            continue
        if not p.endswith(os.sep):
            p += os.sep
        if p in out:
            continue
        if os.path.isdir(p):
            out.append(p)
    return out


def load() -> Config:
    cfg = Config()

    build_hint = _env_path("GAEL_TEST_BUILD_DIR") or (REPO_DIR / "build")

    cfg.gael_bin = _first_existing(
        _env_path("GAEL_TEST_GAEL_BIN"),
        build_hint / "GAEL",
    ) or (Path(shutil.which("GAEL")) if shutil.which("GAEL") else None)

    cfg.mockgen_bin = _first_existing(
        _env_path("GAEL_TEST_MOCKGEN_BIN"),
        build_hint / "mock_data_generator",
    ) or (
        Path(shutil.which("mock_data_generator"))
        if shutil.which("mock_data_generator")
        else None
    )

    cfg.isis_bin = _first_existing(
        _env_path("GAEL_TEST_ISIS"),
        Path.home() / "Projects/ISIS_install/bin/isis",
    ) or (Path(shutil.which("isis")) if shutil.which("isis") else None)

    cfg.global_settings = _first_existing(
        _env_path("GAEL_TEST_GLOBAL_SETTINGS"),
        build_hint / "global_settings.json",
        REPO_DIR / "global_settings.json",
    )

    if cfg.global_settings:
        try:
            raw = json.loads(cfg.global_settings.read_text())
            cfg.base_paths = list(raw.get("basePaths", []))
        except (OSError, ValueError):
            cfg.base_paths = []
    # The tests keep their own grid search path too, so a user with a
    # non-standard grid location only has to set one variable.
    extra = os.environ.get("GAEL_TEST_GRID_PATHS", "")
    cfg.base_paths += [p for p in extra.split(os.pathsep) if p]
    cfg.base_paths.append(str(Path.home() / "ISIS_models"))
    cfg.base_paths = _normalise_base_paths(cfg.base_paths)

    cfg.astra_db = _first_existing(
        _env_path("GAEL_TEST_ASTRA_DB"),
        Path.home() / "data/ASTRA/astra.db",
    )

    cfg.cache_dir = _env_path("GAEL_TEST_CACHE") or (TESTS_DIR / "cache")
    cfg.work_dir = _env_path("GAEL_TEST_WORKDIR")
    cfg.jobs = int(os.environ.get("GAEL_TEST_JOBS", "0")) or (os.cpu_count() or 1)
    cfg.allow_isis = os.environ.get("GAEL_TEST_NO_ISIS", "") == ""
    cfg.grid = os.environ.get("GAEL_TEST_GRID", "sdB/processed/")
    return cfg


def isis_env(isis_bin: Path) -> dict:
    """Environment for a child ISIS process.

    Mirrors what a normal interactive ISIS session gets: the install prefix on
    ``PATH`` and its ``lib`` on the loader path.
    """
    env = dict(os.environ)
    prefix = isis_bin.parent.parent
    env["PATH"] = f"{isis_bin.parent}{os.pathsep}" + env.get("PATH", "")
    libdir = prefix / "lib"
    if libdir.exists():
        env["LD_LIBRARY_PATH"] = f"{libdir}{os.pathsep}" + env.get(
            "LD_LIBRARY_PATH", ""
        )
    # Keep ISIS/S-Lang from trying to open an X display for its plot device.
    env.pop("DISPLAY", None)
    return env


_UUID_RE = re.compile(r"^[0-9a-fA-F-]{36}$")


def is_uuid(text: str) -> bool:
    return bool(_UUID_RE.match(text))
