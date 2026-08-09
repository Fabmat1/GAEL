"""Reader for ASTRA's ``.asd`` spectrum files.

Layout (see ASTRA ``src/utils/DataStore.cpp`` and ``src/models/Spectrum.cpp``)::

    'ASTR'                        4 raw bytes
    quint16 format_version        \\  QDataStream, Qt_6_0 => big endian
    quint16 data_type             /
    QByteArray compressed         quint32 length + qCompress() blob

``qCompress`` output is a 4-byte big-endian *uncompressed* size followed by a
plain zlib stream.  The decompressed payload for a spectrum is::

    quint32 n_wavelength, n_flux, n_error
    double  wavelength[n_wavelength]
    double  flux[n_flux]
    double  error[n_error]
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np

MAGIC = b"ASTR"
SPECTRUM_DATA = 1


class AsdError(RuntimeError):
    pass


def read_spectrum(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(wavelength, flux, error)``; ``error`` is empty if not stored."""
    blob = Path(path).read_bytes()
    if blob[:4] != MAGIC:
        raise AsdError(f"{path}: not an ASTRA .asd file (bad magic)")

    off = 4
    version, dtype = struct.unpack_from(">HH", blob, off)
    off += 4
    if version > 1:
        raise AsdError(f"{path}: unsupported .asd format version {version}")
    if dtype != SPECTRUM_DATA:
        raise AsdError(f"{path}: expected spectrum data, got type {dtype}")

    (n_compressed,) = struct.unpack_from(">I", blob, off)
    off += 4
    if n_compressed == 0xFFFFFFFF:
        raise AsdError(f"{path}: null payload")
    # Skip qCompress' 4-byte expected-size prefix; the rest is raw zlib.
    payload = zlib.decompress(blob[off : off + n_compressed][4:])

    n_wl, n_fl, n_er = struct.unpack_from(">III", payload, 0)
    pos = 12

    def take(count: int) -> np.ndarray:
        nonlocal pos
        arr = np.frombuffer(payload, dtype=">f8", count=count, offset=pos)
        pos += 8 * count
        return arr.astype(np.float64)

    return take(n_wl), take(n_fl), take(n_er)


def write_ascii3(path: str | Path, wl, flux, err) -> None:
    """Write the 3-column ASCII form that both GAEL and ISIS consume.

    The exact byte content matters: it feeds the ISIS reference cache key, so
    the formatting must stay stable across runs and machines.
    """
    wl = np.asarray(wl, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)
    err = np.asarray(err, dtype=np.float64)
    if err.size != wl.size:
        raise AsdError("flux errors missing; cannot build a 3-column spectrum")
    with open(path, "w") as fh:
        for w, f, e in zip(wl, flux, err):
            fh.write(f"{w:.6f} {f:.8e} {e:.8e}\n")
