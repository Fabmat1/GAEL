"""A patched copy of ``stellar_isisscripts.sl``, for the cases ISIS cannot run.

``spectroscopy_automated`` aborts on *every* two-component fit that passes
``auto_freeze_sur_ratio = 0``::

    Double_Type > Undefined_Type is not possible
    Binary operation between Array_Type and Undefined_Type failed
    stellar_isisscripts.sl:...:spectroscopy_automated:Type Mismatch

``c2_detection_thres`` is declared *inside* ``if(ngrid==2 &&
auto_freeze_sur_ratio==1)`` (``spectroscopy_automated.sl:1163``) but read at
``:1738``, in a block guarded only by ``if(ngrid>1)``.  S-Lang hoists the
declaration to function scope, so the name exists but is never assigned when
the first block is skipped, and the comparison at ``:1738`` fails.

That single switch is exactly the one the binary suite needs.  Turning it back
on is not a weaker comparison, it is a different one: at ``:386-409`` ISIS
*removes the second grid entirely*, before fitting, whenever the seeded
``c2_sur_ratio`` is at or below ``sur_ratio_thres = 5`` -- true of every
physically sensible binary -- so ``auto_freeze_sur_ratio = 1`` would have ISIS
fit one component while GAEL fits two.

So the reference has to be patched.  The ISIS *installation* is deliberately
left byte-for-byte stock: instead, this module derives a patched copy of the
concatenated script under the test cache and points ISIS at it with ``-i``,
whose init file prepends the copy's directory to the load path.  The patch is
therefore a reviewable part of this repository, survives an ISIS rebuild, and
reproduces on any machine.

It is applied **only** to cases that cannot run without it -- more than one
component *and* ``auto_freeze_sur_ratio`` off -- and :data:`PATCH_ID` enters
those cases' cache keys.  The scope of the patch and the scope of the key
invalidation are the same set, so no already-cached reference is affected.

The change itself is a one-liner and cannot alter any run with
``auto_freeze_sur_ratio = 1``: the declaration moves next to
``sur_ratio_thres``, and what is left behind assigns the same constant at the
same point.
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

# Enters the ISIS cache key of every case that needs the patch.  Bump it when
# the substitutions below change.
PATCH_ID = "c2-detection-thres-v1"

SCRIPT_NAME = "stellar_isisscripts.sl"


class IsisPatchError(RuntimeError):
    """The patched reference could not be prepared."""


# ``(pattern, replacement, expected occurrences)``, applied in order.  The
# demotion comes first: hoisting inserts a second ``variable
# c2_detection_thres`` line, which the demotion pattern would then also match.
_SUBSTITUTIONS = (
    (
        re.compile(r"^(\s*)variable c2_detection_thres = 0\.05;$", re.M),
        r"\1c2_detection_thres = 0.05;",
        1,
    ),
    (
        re.compile(r"^(\s*)variable sur_ratio_thres = 5\.;$", re.M),
        r"\1variable sur_ratio_thres = 5.;\n"
        r"\1% GAEL test harness: hoisted out of the auto_freeze_sur_ratio block\n"
        r"\1% below, which the final-fit block (guarded only by ngrid>1) reads\n"
        r"\1% without ever entering.\n"
        r"\1variable c2_detection_thres = 0.05;",
        1,
    ),
)


def needs_patch(case) -> bool:
    """True when this case is one stock ISIS cannot fit at all."""
    s = case.settings
    return s.n_components > 1 and s.auto_freeze_sur_ratio is False


# ------------------------------------------------------------------ locating
def _load_path_entries() -> list[Path]:
    """Directories ``~/.isisrc`` puts on ISIS's load path, in file order."""
    rc = Path.home() / ".isisrc"
    if not rc.exists():
        return []
    try:
        text = rc.read_text(errors="replace")
    except OSError:
        return []
    return [
        Path(m).expanduser()
        for m in re.findall(r'add_to_isis_load_path\s*\(\s*"([^"]+)"', text)
    ]


def find_scripts(isis_bin: Path) -> Path:
    """Locate the concatenated ``stellar_isisscripts.sl`` ISIS would load."""
    override = os.environ.get("GAEL_TEST_ISIS_SCRIPTS")
    if override:
        p = Path(override).expanduser()
        if p.is_dir():
            p = p / SCRIPT_NAME
        if not p.exists():
            raise IsisPatchError(
                f"GAEL_TEST_ISIS_SCRIPTS points at {p}, which does not exist"
            )
        return p

    prefix = Path(isis_bin).resolve().parent.parent
    candidates = [d / SCRIPT_NAME for d in _load_path_entries()]
    candidates.append(prefix / "src/stellar_isisscripts/share" / SCRIPT_NAME)
    candidates.append(prefix / "share" / SCRIPT_NAME)
    for c in candidates:
        if c.exists():
            return c
    raise IsisPatchError(
        f"cannot find {SCRIPT_NAME} (looked in ~/.isisrc's load path and under "
        f"{prefix}); set GAEL_TEST_ISIS_SCRIPTS"
    )


# ------------------------------------------------------------------ patching
def apply_patch(text: str) -> str:
    for pattern, replacement, expected in _SUBSTITUTIONS:
        text, n = pattern.subn(replacement, text)
        if n != expected:
            raise IsisPatchError(
                f"{PATCH_ID}: expected {expected} match(es) for "
                f"{pattern.pattern!r} in {SCRIPT_NAME}, found {n}. The ISIS "
                f"scripts have changed; re-check spectroscopy_automated.sl "
                f"before trusting this suite."
            )
    # The point of the whole exercise: exactly one declaration, and it is the
    # hoisted one.
    n_decl = len(re.findall(r"^\s*variable c2_detection_thres\b", text, re.M))
    if n_decl != 1:
        raise IsisPatchError(
            f"{PATCH_ID}: {n_decl} declarations of c2_detection_thres after "
            f"patching, expected exactly 1"
        )
    return text


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _symlink(target: Path, link: Path) -> None:
    """Idempotent, and safe when several workers prepare the tree at once."""
    try:
        link.symlink_to(target)
    except FileExistsError:
        pass


def _link_tree(src_root: Path, dst_root: Path) -> None:
    """Mirror *src_root* into *dst_root* with symlinks, except ``share/``.

    The script derives ``ISISSCRIPTS_REFPATH`` from its own ``__FILE__`` and
    then imports ``<refpath>/slirp/c_functions`` and reads ``<refpath>/refdata``,
    so the copy has to sit in the same directory layout as the install.
    """
    (dst_root / "share").mkdir(parents=True, exist_ok=True)
    for entry in src_root.iterdir():
        if entry.name == "share":
            continue
        _symlink(entry, dst_root / entry.name)
    for entry in (src_root / "share").iterdir():
        if entry.name == SCRIPT_NAME:
            continue
        _symlink(entry, dst_root / "share" / entry.name)


def prepare(cfg) -> Path:
    """Build (once) the patched tree and return the ISIS init file to use.

    The result is keyed on the *installed* script's digest, so an ISIS update
    produces a fresh copy rather than a stale patched one.
    """
    if not cfg.isis_bin:
        raise IsisPatchError("no ISIS binary")
    src = find_scripts(Path(cfg.isis_bin))
    root = (
        Path(cfg.cache_dir)
        / "isis_patched"
        / f"{PATCH_ID}-{_digest(src)[:12]}"
    )
    rc = root / "isisrc.sl"
    target = root / "share" / SCRIPT_NAME

    if not (rc.exists() and target.exists()):
        root.mkdir(parents=True, exist_ok=True)
        _link_tree(src.parent.parent, root)
        patched = apply_patch(src.read_text(errors="replace"))
        # Written via a temporary so a concurrent worker never loads a
        # half-written script.
        tmp = target.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(patched)
        os.replace(tmp, target)
        tmp = rc.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(_RC_TEMPLATE.format(share=root / "share"))
        os.replace(tmp, rc)
    return rc


# ``-i FILE`` replaces ~/.isisrc entirely, so load the real one first; then add
# the patched directory, which add_to_isis_load_path puts *ahead* of everything
# already on the path.
_RC_TEMPLATE = """\
% Generated by GAEL's test harness (tests/gael_isis/isis_patch.py).
% Do not edit: it is rewritten whenever the installed ISIS scripts change.
variable _gael_home = getenv("HOME");
if (_gael_home != NULL)
{{
  variable _gael_rc = path_concat(_gael_home, ".isisrc");
  if (stat_file(_gael_rc) != NULL) () = evalfile(_gael_rc);
}}
add_to_isis_load_path("{share}");
"""
