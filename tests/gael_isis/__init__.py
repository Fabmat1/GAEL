"""Comparison harness: GAEL fit results vs. ISIS ``spectroscopy_automated`` fits.

The package is deliberately dependency-light (numpy + astropy only) and drives
both codes through their normal user-facing entry points:

  * ISIS  -- a generated ``fit.sl`` executed by the ``isis`` binary
  * GAEL  -- a generated ``fit.json`` executed by the ``GAEL`` CLI

Both codes are handed the *same* ASCII spectrum files and the *same* fit
configuration, so any difference in the results is a difference between the
fitting codes rather than between their inputs.
"""

SKIP_EXIT_CODE = 77  # matches SKIP_RETURN_CODE in tests/CMakeLists.txt
