# GAEL regression tests: agreement with ISIS

These suites compare GAEL's fit results against reference fits produced by
ISIS's `spectroscopy_automated`, which is the established code these fits are
normally done with.

| test | data | cases |
|---|---|---|
| `gael_vs_isis_mock` | mock spectra from `mock_data_generator` (true parameters known) | 100 single-spectrum + 100 joint 5-spectrum fits |
| `gael_vs_isis_real` | real observed spectra from ASTRA's **RVVD** project | 100 single-spectrum + 100 joint 5-spectrum fits |
| `gael_vs_isis_xshooter` | X-Shooter-like mocks (UVB + VIS) on the metal-bearing Feros grids | 12 abundance + 12 two-component fits |

The sdB grids carry no element axes and one stellar component, so the first two
suites cannot reach either feature. The X-Shooter suite exists for exactly
those two: `metal` leaves Fe and Si free, `binary` fits two components on two
different grids. See `gael_isis/xshooter.py` for what each holds fixed and why.

Both codes are handed the **same** 3-column ASCII files and equivalent fit
settings (grid, wavelength cut, masks, continuum anchors, initial guess, frozen
parameters, `untie = {vrad}`), so a difference in the results is a difference
between the fitting codes, not between their inputs.

```
ctest -R gael_vs_isis            # both suites
ctest -R gael_vs_isis_smoke      # 4 mock cases, for checking the harness itself
./test_mock_vs_isis.py --kinds single --limit 10   # run a script directly
```

`--limit` runs are report-only: the thresholds are calibrated for the full
100-case suites and a truncated run cannot meet their sample-size floors, so
gating it would only produce a misleading failure.

If ISIS is not installed but the reference cache is populated, the suites still
run — set `GAEL_TEST_NO_ISIS=1` to guarantee no ISIS process is launched. Should
any reference be missing in that mode the suite skips rather than quietly
comparing fewer cases.

## The ISIS reference cache

ISIS is slow, so every reference fit is cached under `tests/cache/isis/` and
keyed by a SHA-256 of the generated `fit.sl` **plus the content digest of every
spectrum file**. Change a spectrum, a mask, an initial guess or the script
generator, and the key changes and ISIS reruns; change nothing and no ISIS
process is started at all.

The first run computes ~200 ISIS fits per suite in parallel across all cores
(`GAEL_TEST_JOBS`, default = core count). On a 16-core machine that is roughly

| | ISIS per fit | GAEL per fit |
|---|---|---|
| single spectrum | ~6 s | ~15 s |
| 5 spectra | ~23 s | ~60 s |

GAEL results are deliberately **not** cached — they are what is under test.

Reference fits run with `plot_pdf = 0`, which skips the xfig/LaTeX/PDF stage.
That is ~3x faster and drops the TeX toolchain from the test's dependencies;
every fitted number is still recovered at full precision from
`results_conf.fits` (values, confidence intervals, grid bounds) and
`spectroscopy_spectrum_params.dat` (frozen/tied structure).

The cache is not committed (see `.gitignore`): mock spectra are regenerated
from the model grid, so entries are only valid for a given grid version. Delete
`tests/cache/` to force a full recomputation.

## The patched ISIS (two-component cases only)

Stock `spectroscopy_automated` **cannot run** a two-component fit with
`auto_freeze_sur_ratio = 0`. `c2_detection_thres` is declared inside
`if(ngrid==2 && auto_freeze_sur_ratio==1)` (`spectroscopy_automated.sl:1163`)
but read at `:1738`, in a block guarded only by `if(ngrid>1)`; S-Lang hoists the
declaration to function scope, so the name exists, is never assigned, and the
comparison fails with

```
Binary operation between Array_Type and Undefined_Type failed
stellar_isisscripts.sl:...:spectroscopy_automated:Type Mismatch
```

Leaving the qualifier at its default of 1 is not an alternative: at `:386-409`
ISIS *deletes the second grid outright*, before fitting, whenever the seeded
`c2_sur_ratio` is at or below `sur_ratio_thres = 5` — true of every physically
sensible binary. ISIS would fit one component while GAEL fits two.

So `gael_isis/isis_patch.py` derives a patched copy of the concatenated
`stellar_isisscripts.sl` under `tests/cache/isis_patched/`, and runs those
cases as `isis -i <generated rc> fit.sl`, whose init file loads `~/.isisrc` and
then puts the copy first on the load path. **The ISIS installation is never
modified.** The copy is keyed on the installed script's digest, so an ISIS
update rebuilds it rather than going stale, and the substitutions are checked
by match count — if the upstream lines have moved, the suite fails loudly
instead of silently running something else.

The patch itself is one line, hoisting the declaration next to
`sur_ratio_thres` and leaving a plain assignment of the same constant behind,
so it cannot change any run with `auto_freeze_sur_ratio = 1`. It is applied
only to cases that cannot run without it (more than one component *and* the
qualifier off), and only those cases carry `isis_patch` in their cache key —
the two scopes are the same set, so no previously cached reference is
invalidated.

## What is compared, and what is skipped

Compared: **teff**, **logg**, **he** (tied across a case) and **vrad** (untied,
one value per spectrum, so a 5-spectrum case contributes 5 samples). The
X-Shooter suites compare component-qualified names instead — `c1_teff`,
`c1_FE`, `c1_SI` for `metal`, and `c1_*`, `c2_*` plus `c2_sur_ratio` for
`binary` — since the parameter set is a property of the case.

A case is skipped entirely when either code fails or drops a spectrum — the
untied parameters are matched by index, so a dropped spectrum would misalign
them. An individual parameter is skipped when either code froze it or railed
against a grid edge: a railed parameter is not a fit result, and letting those
in would mean two codes stuck at the same boundary look like perfect agreement.

Because of that last point the real-spectra sample is restricted to stars whose
catalogued temperature lies inside the sdB grid (18000–45000 K). RVVD also
contains cooler objects; fitting those with this grid rails both codes at the
15000 K edge and carries no information.

Spectra come from **LAMOST/LRS** and **SDSS/BOSS**. Both have a constant
resolving power, which maps exactly onto GAEL's `R(λ) = resOffset + resSlope·λ`
with `resSlope = 0`, so both codes see an identical resolution model. **SOAR is
deliberately excluded** — those spectra have known fitting issues.

## Pass/fail

The gate is statistical, not per-fit. Individual spectra can legitimately land
in different local χ² minima in either code; a shifted *median* or an inflated
*scatter* over 100 fits is what actually indicates a regression. Per parameter:

```
|median(GAEL - ISIS)|              <  max_abs_bias
1.4826 * MAD(GAEL - ISIS)          <  max_scatter
fraction(|d - bias| > 5*scatter)   <  max_outlier_fraction
```

plus a floor on how many cases stayed comparable, which catches a regression
that makes fits fail or rail rather than shift.

Thresholds live in `tests/tolerances.json`, calibrated from a measured run with
head-room and clamped to per-parameter floors (25 K, 0.005 dex, 0.3 km/s for
bias) so a suite that happens to agree very tightly cannot lock in a threshold
that ordinary numerical noise would trip. Without that file the suites report
and pass, so a fresh checkout is never blocked by a missing calibration.
Recalibrate after an intentional change to GAEL's fitting behaviour:

```
./test_mock_vs_isis.py --calibrate
./test_real_vs_isis.py --calibrate
```

Review the resulting diff — a widened tolerance is a statement that GAEL's
agreement with ISIS got worse on purpose.

For mock data the true generating parameters are also known, so the report
additionally shows how well *each* code recovers them. That is reported, not
gated: it is a property of the grid and the noise, not of a GAEL change.

## Sample selection

The real-spectra sample is frozen in `manifests/real_single.json` and
`manifests/real_multi.json` (committed) so results and the cache stay stable as
the database grows. Regenerate deliberately:

```
python -m gael_isis.selection --regenerate
```

Selection cuts: unflagged spectra, median SNR ≥ 20 in 4000–5000 Å, coverage
from ≤3900 Å to ≥5200 Å, and the star Teff cut above. The multi-spectrum cases
use 5 spectra of the *same* star; the single-spectrum cases prefer stars not
used by the multi sample so the two are independent.

## Skipping

Both tests exit **77** (CTest `SKIP_RETURN_CODE`) with a readable reason when a
prerequisite is missing — no ISIS, no model grid, no ASTRA database — so a
machine without the full setup still gets a clean `ctest` run.

## Environment variables

| variable | meaning |
|---|---|
| `GAEL_TEST_GAEL_BIN`, `GAEL_TEST_MOCKGEN_BIN` | binaries (CMake passes these) |
| `GAEL_TEST_ISIS` | ISIS binary |
| `GAEL_TEST_ISIS_SCRIPTS` | `stellar_isisscripts.sl` (or its directory) to patch, if it is not on `~/.isisrc`'s load path |
| `GAEL_TEST_GLOBAL_SETTINGS` | `global_settings.json` used for the GAEL runs |
| `GAEL_TEST_ASTRA_DB` | ASTRA database with the real spectra |
| `GAEL_TEST_CACHE` | cache root (default `tests/cache`) |
| `GAEL_TEST_GRID`, `GAEL_TEST_GRID_PATHS` | model grid and extra search paths |
| `GAEL_TEST_JOBS` | parallel workers (default: core count) |
| `GAEL_TEST_NO_ISIS` | never launch ISIS; use only cached references |
| `GAEL_TEST_KEEP_WORKDIRS` | keep the per-fit scratch directories |
| `GAEL_TEST_ISIS_TIMEOUT`, `GAEL_TEST_GAEL_TIMEOUT` | per-fit timeouts (s) |

## Layout

```
tests/
  test_mock_vs_isis.py     entry point: mock spectra
  test_real_vs_isis.py     entry point: real RVVD spectra
  test_xshooter_vs_isis.py entry point: abundances and two-component fits
  tolerances.json          calibrated pass/fail thresholds
  manifests/               frozen real-spectra sample
  gael_isis/
    config.py              binary/data/cache discovery, skip reasons
    asd.py                 reader for ASTRA's .asd spectrum files
    jobs.py                Case/FitOutcome model shared by both backends
    isis_ref.py            fit.sl generation, execution, parsing, caching
    isis_patch.py          patched stellar_isisscripts.sl for binary cases
    xshooter.py            the metal and binary case definitions
    gael_run.py            GAEL CLI driver and fit_parameters.csv parsing
    selection.py           database queries and sample manifests
    mockdata.py            seeded mock_data_generator invocation
    compare.py             statistics, tolerance check, report
    runner.py              parallel execution and the main() body
```
