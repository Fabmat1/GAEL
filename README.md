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
    build-essential gfortran cmake git \
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
     base-devel git cmake gcc-fortran \
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
```
xcode-select --install      # first time only
brew install cmake git llvm eigen boost openblas cfitsio tbb nlohmann-json cxxopts

# Use Homebrew LLVM (recommended)
export CC=/opt/homebrew/opt/llvm/bin/clang
export CXX=/opt/homebrew/opt/llvm/bin/clang++

# If CMake cannot locate Homebrew packages:
# cmake .. -DCMAKE_PREFIX_PATH="$(brew --prefix)"
```

---

## 4. Build GAEL (identical on every platform)
```
git clone https://github.com/<your-user>/GAEL.git
cd GAEL
mkdir build && cd build

#   GPU on  (default): -DGAEL_ENABLE_CUDA=ON
#   GPU off           : -DGAEL_ENABLE_CUDA=OFF
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

sudo make install         # optional, installs lib + CLI
```

---

## Running GAEL
```
GAEL --global globals.json --fit run.json [--threads N]
```

* `globals.json` – general configuration (paths, hardware, etc.)  
* `run.json`     – parameters of the individual fit  
* `--threads N`  – override automatic core detection  

---

## Troubleshooting

| Message                                                       | Solution                                                                                               |
|---------------------------------------------------------------|---------------------------------------------------------------------------------------------------------|
| `Could NOT find Eigen3 (found version … 3.3.x)`               | Install/upgrade to Eigen ≥ 3.4 (see instructions above).                                                |
| `Could NOT find unordered_dense`                              | Ensure the header resides in a CMake search path, e.g. `/usr/local/include/ankerl/unordered_dense`.     |
| `CUDA toolkit not found`                                      | Install CUDA **or** rebuild with `-DGAEL_ENABLE_CUDA=OFF`.                                           |

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
