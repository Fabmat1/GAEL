# GAEL  
Grid-based Atmospheric Estimation Library  
*Modern C++20 framework to fit observed stellar spectra with synthetic model grids.*

Pronounce as : /ɡeɪl/

---

## Table of Contents
1. [Features](#features)  
2. [Quick Start](#quick-start)  
3. [Installation](#installation)  
   * [Ubuntu / Debian](#1-ubuntu)  
   * [Arch Linux](#2-arch-linux)  
   * [macOS](#3-macos)  
   * [Windows](#3b-windows)  
4. [Building GAEL](#4-build-gael)  
5. [Running GAEL](#running-gael)  
6. [Troubleshooting](#troubleshooting)  
7. [License & Citation](#license--citation)  

---

## Features
* Full-spectrum forward modelling with synthetic grids  
* CPU runtime with optional CUDA acceleration  
* Lightweight, header-only third-party libraries wherever possible  
* Modern CMake build system, fully unit-tested (`ctest`)  
* Multithreaded via OpenMP and/or Intel TBB  

---

## Quick Start
```
git clone https://github.com/<your-user>/GAEL.git
cd GAEL
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release           # add -DGAEL_ENABLE_CUDA=OFF to disable GPU
make -j$(nproc)
sudo make install
GAEL --global globals.json --fit run.json --threads 8
```

---

## Installation

### 1. Ubuntu
Tested on 22.04 LTS, 24.04 LTS (or newer)
```
sudo apt update && sudo apt upgrade

# Build & runtime dependencies
sudo apt install \
    build-essential cmake git \
    libopenblas-dev \
    libboost-dev libboost-filesystem-dev libboost-system-dev \
    libcfitsio-dev libccfits-dev \
    libtbb-dev \
    libnlohmann-json3-dev libcxxopts-dev \
    python3-dev python3-numpy \
    libomp-dev                 # OpenMP runtime for clang (gcc already ships it)
```

Optional CUDA back-end:
```
sudo apt install nvidia-cuda-toolkit          # or the official NVIDIA .run installer
```

#### Eigen ≥ 3.4 (only if your distro still ships 3.3.x)
```
git clone https://gitlab.com/libeigen/eigen.git --branch 3.4.0
cd eigen && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
sudo make install                              # installs to /usr/local
```

#### ankerl::unordered_dense (header-only)
```
git clone https://github.com/martinus/unordered_dense.git
cd unordered_dense && mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local
sudo make install
```

---

### 2. Arch Linux
```
sudo pacman -Syu
sudo pacman -S \
     base-devel git cmake \
     openblas \
     boost \
     eigen \
     cfitsio ccfits \
     tbb \
     nlohmann-json \
     cxxopts \
     python python-numpy
```

Optional CUDA back-end:
```
sudo pacman -S cuda
```

ankerl::unordered_dense from the AUR:
```
yay -S unordered_dense-git
```

---

### 3. macOS
Apple Silicon and Intel, with Apple's own compiler (Xcode command-line tools)
and Homebrew:
```
xcode-select --install      # first time only
brew install cmake git eigen boost openblas libomp cfitsio ccfits tbb \
             nlohmann-json cxxopts unordered_dense

# Only for the PDF/plot reports (otherwise configure with -DGAEL_BUILD_REPORT=OFF):
brew install python numpy
```
`openblas` and `libomp` are keg-only in Homebrew; GAEL's CMake looks in their
Homebrew prefixes itself, so no extra flags are needed. CUDA does not exist on
macOS, and the build notices that on its own.

---

### 3b. Windows
Two ways, in order of how little effort they take:

* **WSL2** (recommended). Install Ubuntu from the Microsoft Store, then follow
  the [Ubuntu](#1-ubuntu) instructions inside it. GAEL runs there exactly as on
  Linux, at native speed.
* **Native, MSYS2 UCRT64** (what ASTRA's Windows build uses). In the
  "MSYS2 UCRT64" shell:
  ```
  pacman -S --needed mingw-w64-ucrt-x86_64-{gcc,cmake,ninja,eigen3,openblas,boost,tbb,cfitsio,nlohmann-json,cxxopts,unordered_dense}
  ```
  MSYS2 has no CCfits package; build it once from
  <https://heasarc.gsfc.nasa.gov/fitsio/CCfits/> with
  `cmake -G Ninja -DBUILD_SHARED_LIBS=OFF -DCMAKE_POLICY_VERSION_MINIMUM=3.5`
  and pass its install prefix to GAEL as `-DCMAKE_PREFIX_PATH=<prefix>`.
  Configure GAEL with `-G Ninja -DGAEL_BUILD_REPORT=OFF`.

  Visual Studio (MSVC) is not supported.

---

## 4. Build GAEL (identical on every platform)
```
git clone https://github.com/<your-user>/GAEL.git
cd GAEL
mkdir build && cd build

#   GPU on  (default): -DGAEL_ENABLE_CUDA=ON
#   GPU off           : -DGAEL_ENABLE_CUDA=OFF
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j 8      # works on every platform (macOS has no `nproc`)
ctest -L unit             # quick self-check, a few seconds

sudo cmake --install .    # optional, installs lib + CLI
```

---

## Running GAEL
```
GAEL --global globals.json --fit run.json [--threads N]
```

* `globals.json` – general configuration (paths, hardware, etc.)  
* `run.json`     – parameters of the individual fit  
* `--threads N`  – override automatic core detection  

### Multi-component (binary) fits

The number of stellar components is the number of entries in `grids`, exactly
as in ISIS, where it is the number of grids handed to
`initialize_grid_fit_spectroscopy`. Each component's start values live under
the `cN_` prefix in `initialGuess`:

```jsonc
{
  "grids": ["sdB/processed/", "BG/processed/"],
  "initialGuess": {
    "c1_vrad": {"value":   0.0, "freeze": false},
    "c1_teff": {"value": 25000.0, "freeze": false},
    // ... c1_vsini, c1_zeta, c1_logg, c1_xi, c1_z, c1_HE

    "c2_vrad": {"value":   0.0, "freeze": false},
    "c2_teff": {"value": 15000.0, "freeze": false},
    // ... c2_vsini, c2_zeta, c2_logg, c2_xi, c2_z, c2_HE

    "c2_sur_ratio": {"value": 1.0, "freeze": false}
  }
}
```

`cN_sur_ratio` is the ratio of component *N*'s effective surface area to
component 1's — ISIS's `sur_ratio`. It is optional (default 1, free) and, like
the other stellar parameters, tied across spectra unless listed in
`untieParams`. `c1_sur_ratio` defines the scale and is always 1 and frozen;
the others are bounded to `[0, 1500]`, as `stellar_set_ranges` bounds them.

The components are combined the way ISIS combines them: the *calibrated*
model fluxes (the grid's `c` column) are summed with the surface ratios as
weights and divided by the summed continua,

```
n(λ)  =  Σ s_k · F_k(λ)  /  Σ s_k · C_k(λ)
```

so a line of the secondary is diluted by the primary's continuum flux at that
wavelength rather than by a constant. A one-component fit is unaffected and
still uses the grid's normalised `f` column directly.

Two optional `settings` keys mirror ISIS's `auto_freeze_sur_ratio`, which is
**off by default here** (ISIS has it on):

| key | default | effect |
|---|---|---|
| `autoFreezeSurRatio` | `false` | drop a second grid whose *initial* surface ratio is already at or below `surRatioThres` (or which names the same grid as the first), and, after the first full fit, retire a secondary whose fitted ratio fell below `surRatioThres` or whose peak contribution to the composite stayed below `c2DetectionThres` — setting its ratio to zero and freezing the whole component |
| `surRatioThres` | `5.0` | ISIS's `sur_ratio_thres` |
| `c2DetectionThres` | `0.05` | ISIS's `c2_detection_thres` |

---

## Troubleshooting

| Message                                                       | Solution                                                                                               |
|---------------------------------------------------------------|---------------------------------------------------------------------------------------------------------|
| `Could NOT find Eigen3 (found version … 3.3.x)`               | Install/upgrade to Eigen ≥ 3.4 (see instructions above).                                                |
| `Could NOT find unordered_dense`                              | Ensure the header resides in a CMake search path, e.g. `/usr/local/include/ankerl/unordered_dense.h`.   |
| `OpenMP for C++ not found` (macOS)                            | `brew install libomp`. Apple's compiler has no OpenMP runtime of its own.                              |
| `CUDA toolkit not found`                                      | Install CUDA **or** rebuild with `-DGAEL_ENABLE_CUDA=OFF`.                                           |
| `Warning: model file … is missing; interpolating across it`   | The grid lists a node with no model file. The fit goes on: the missing node is linearly interpolated from its two neighbours along one axis (rule: "Holes in a grid" in `src/ModelGrid.cpp`), and the result's `grid_gaps` / `warnings` name every such file the fit used. |
| `model file … is missing … cannot be interpolated across`     | A missing node without existing neighbours on both sides along any axis. Restore the file or keep the fit away from that corner of the grid. |
| `cannot read model file …`                                    | The file exists but could not be read (I/O error, unreachable mount or host, corrupt file). This is never bridged; fix the storage and rerun. |

---

## License & Citation
GAEL is released under the MIT license.  
If you use this code in a publication, please cite
```
@misc{GAEL2025,
  author  = {Mattig et al.},
  title   = {GAEL – Grid-based Atmospheric Estimation Library},
  year    = {2025},
  url     = {https://github.com/Fabmat1/GAEL}
}
```

Happy fitting!
