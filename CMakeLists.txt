cmake_minimum_required(VERSION 3.18)
cmake_policy(SET CMP0074 NEW)       # honour *_ROOT / CMAKE_PREFIX_PATH

# No Fortran: nothing here is Fortran, and declaring it made a gfortran a
# configure-time requirement on every platform -- on macOS and Windows that was
# a whole extra toolchain installed only to satisfy this line.
project(GAEL LANGUAGES C CXX)

if(POLICY CMP0167)
    cmake_policy(SET CMP0167 OLD)
endif()

# ---------------------------------------------------------------------------
# 0. Tell CMake where Home-brew lives (needed only on macOS)
# ---------------------------------------------------------------------------
#  openblas and libomp are keg-only: Homebrew does not link them into its
#  prefix, so CMake finds them only if their own prefixes are named.
set(_gael_libomp "")
if(APPLE)
    foreach(_p "/opt/homebrew" "/usr/local")
        if(EXISTS "${_p}")
            list(APPEND CMAKE_PREFIX_PATH "${_p}")
            foreach(_keg openblas libomp)
                if(EXISTS "${_p}/opt/${_keg}")
                    list(APPEND CMAKE_PREFIX_PATH "${_p}/opt/${_keg}")
                endif()
            endforeach()
            if(NOT _gael_libomp AND EXISTS "${_p}/opt/libomp/include/omp.h")
                set(_gael_libomp "${_p}/opt/libomp")
            endif()
        endif()
    endforeach()
endif()

# ---------------------------------------------------------------------------
# 1. Build options
# ---------------------------------------------------------------------------
# Without this a plain `cmake ..` leaves CMAKE_BUILD_TYPE empty, which means no
# -O flag and no NDEBUG: the fit then runs ~9x slower with Eigen's per-access
# assertions enabled.  Only applies to single-config generators.
if(PROJECT_IS_TOP_LEVEL AND NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
                 Debug Release RelWithDebInfo MinSizeRel)
    message(STATUS "CMAKE_BUILD_TYPE was empty - defaulting to Release")
endif()

option(GAEL_ENABLE_CUDA "Build CUDA back-end" ON)

option(GAEL_LEGACY_CONVOLUTION "Use legacy spectral-resolution algorithm" OFF)

if(GAEL_LEGACY_CONVOLUTION)
    add_compile_definitions(GAEL_LEGACY_CONVOLUTION)
endif()

# ---------------------------------------------------------------------------
# 2. C++ settings
# ---------------------------------------------------------------------------
if(NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 20)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(PROJECT_IS_TOP_LEVEL)
    #  Tune for the machine doing the build.  Apple's clang on arm64 rejects
    #  -march=native outright and wants -mcpu=native, so ask the compiler
    #  rather than assume.
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag(-march=native GAEL_HAVE_MARCH_NATIVE)
    if(GAEL_HAVE_MARCH_NATIVE)
        add_compile_options(-march=native)
    else()
        check_cxx_compiler_flag(-mcpu=native GAEL_HAVE_MCPU_NATIVE)
        if(GAEL_HAVE_MCPU_NATIVE)
            add_compile_options(-mcpu=native)
        endif()
    endif()
    add_compile_options(
        -funroll-loops
        -ftree-vectorize
    )
endif()
add_compile_options(
    -Wno-deprecated-declarations
)

# ---------------------------------------------------------------------------
# 3. CUDA (optional)
# ---------------------------------------------------------------------------
if(POLICY CMP0104)
    cmake_policy(SET CMP0104 NEW)   # CUDA_ARCHITECTURES must be set explicitly
endif()

# Architectures used when no local GPU can be queried.  The trailing
# "90-virtual" embeds PTX so newer cards (Blackwell and later) still JIT.
set(GAEL_CUDA_ARCHITECTURES_FALLBACK 75-real 80-real 86-real 89-real 90-real 90-virtual
    CACHE STRING "CUDA architectures used when the local GPU cannot be detected")

if(GAEL_ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)            # unlike enable_language(... OPTIONAL) this never hard-errors

    if(CMAKE_CUDA_COMPILER)
        # -- Choose the architectures *before* enable_language/add_library, so
        #    every CUDA target inherits them (CMP0104 forbids leaving it empty).
        #    -DCMAKE_CUDA_ARCHITECTURES=... on the command line still wins.
        if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
            set(_gael_cuda_arch "")
            find_program(GAEL_NVIDIA_SMI nvidia-smi)
            if(GAEL_NVIDIA_SMI)
                execute_process(
                    COMMAND "${GAEL_NVIDIA_SMI}" --query-gpu=compute_cap --format=csv,noheader
                    RESULT_VARIABLE _smi_rc
                    OUTPUT_VARIABLE _smi_out
                    ERROR_VARIABLE  _smi_err
                    OUTPUT_STRIP_TRAILING_WHITESPACE)
                if(_smi_rc EQUAL 0)
                    string(REPLACE "\n" ";" _smi_caps "${_smi_out}")
                    foreach(_cap IN LISTS _smi_caps)
                        string(STRIP "${_cap}" _cap)
                        string(REPLACE "." "" _cap "${_cap}")   # "8.9" -> "89"
                        if(_cap MATCHES "^[0-9]+$")
                            list(APPEND _gael_cuda_arch "${_cap}-real")
                        endif()
                    endforeach()
                    list(REMOVE_DUPLICATES _gael_cuda_arch)
                endif()
            endif()

            if(_gael_cuda_arch)
                message(STATUS "GAEL: building CUDA for the local GPU(s): ${_gael_cuda_arch}")
            else()
                set(_gael_cuda_arch ${GAEL_CUDA_ARCHITECTURES_FALLBACK})
                message(STATUS "GAEL: no usable GPU detected (driver missing or mismatched) – "
                               "building CUDA for ${_gael_cuda_arch}")
            endif()

            set(CMAKE_CUDA_ARCHITECTURES "${_gael_cuda_arch}"
                CACHE STRING "CUDA architectures to generate code for")
        endif()

        enable_language(CUDA)
    else()
        message(STATUS "GAEL: no CUDA compiler found – building without the CUDA back-end")
    endif()
endif()
find_package(CUDAToolkit QUIET)

# ---------------------------------------------------------------------------
# 4. Dependencies
# ---------------------------------------------------------------------------
# The PDF/plot reporting helpers are the only part that needs Python's
# development files and NumPy.  Declared here, ahead of the dependency search,
# so a build without them does not have to have those installed.
option(GAEL_BUILD_REPORT "Build the PDF/plot reporting helpers" ON)

# Apple's clang compiles OpenMP but ships neither the runtime nor the driver
# flag, and CMake's FindOpenMP does not know where Homebrew's libomp lives.
# Values passed on the command line (ASTRA's build-macos.sh does) win.
if(APPLE AND CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang" AND _gael_libomp
   AND NOT DEFINED OpenMP_CXX_FLAGS)
    foreach(_lang C CXX)
        set(OpenMP_${_lang}_FLAGS "-Xpreprocessor -fopenmp -I${_gael_libomp}/include")
        set(OpenMP_${_lang}_LIB_NAMES omp)
    endforeach()
    set(OpenMP_omp_LIBRARY "${_gael_libomp}/lib/libomp.dylib")
endif()

find_package(Eigen3        REQUIRED)
# C too, because some OpenBLAS packages' targets link OpenMP::OpenMP_C.  C++
# is required: GaelAPI.cpp includes <omp.h> unconditionally, and failing here
# names the missing piece where failing at that #include did not.
find_package(OpenMP        COMPONENTS C CXX)
if(NOT OpenMP_CXX_FOUND)
    message(FATAL_ERROR
        "OpenMP for C++ not found. On macOS: brew install libomp (Apple's "
        "clang has no OpenMP runtime of its own).")
endif()
find_package(Boost         REQUIRED CONFIG COMPONENTS filesystem)
find_package(nlohmann_json REQUIRED)
find_package(TBB           REQUIRED)
if(GAEL_BUILD_REPORT)
    find_package(Python3 REQUIRED COMPONENTS Interpreter Development NumPy)
else()
    find_package(Python3 COMPONENTS Interpreter)   # the ctest drivers only
endif()

# ---- 4a. OpenBLAS ---------------------------------------------------------
# Linux distributions ship an OpenBLASConfig.cmake that defines the
# OpenBLAS::OpenBLAS target; Homebrew's and some source builds define only
# variables, or nothing usable.  Make the target from the library and header
# when the package did not.  libopenblas bundles LAPACK and LAPACKE.
#
# Some packages (Arch's, for one) also put OpenMP::OpenMP_Fortran in the
# target's link interface, which is the reason GAEL used to declare Fortran.
# libopenblas already carries its own dependency on the OpenMP runtime, and
# GAEL compiles no Fortran, so an empty stand-in satisfies it without a
# Fortran compiler.  Directory-scoped, so a parent project that does find
# OpenMP for Fortran is unaffected.
if(NOT TARGET OpenMP::OpenMP_Fortran)
    add_library(OpenMP::OpenMP_Fortran INTERFACE IMPORTED)
endif()
find_package(OpenBLAS QUIET)
if(NOT TARGET OpenBLAS::OpenBLAS)
    find_library(GAEL_OPENBLAS_LIBRARY NAMES openblas)
    find_path(GAEL_OPENBLAS_INCLUDE_DIR NAMES cblas.h
              PATH_SUFFIXES openblas
              HINTS ${OpenBLAS_INCLUDE_DIRS})
    if(NOT GAEL_OPENBLAS_LIBRARY OR NOT GAEL_OPENBLAS_INCLUDE_DIR)
        message(FATAL_ERROR
            "OpenBLAS not found. Install it (apt: libopenblas-dev, "
            "brew: openblas, MSYS2: mingw-w64-ucrt-x86_64-openblas) or pass "
            "-DCMAKE_PREFIX_PATH=<its prefix>.")
    endif()
    add_library(OpenBLAS::OpenBLAS UNKNOWN IMPORTED)
    set_target_properties(OpenBLAS::OpenBLAS PROPERTIES
        IMPORTED_LOCATION             "${GAEL_OPENBLAS_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${GAEL_OPENBLAS_INCLUDE_DIR}")
    message(STATUS "GAEL: OpenBLAS from ${GAEL_OPENBLAS_LIBRARY}")
endif()

# ---- 4a. Header-only libraries that may lack a CMake package --------------
find_package(cxxopts QUIET CONFIG)
if(NOT cxxopts_FOUND)
    message(STATUS "cxxoptsConfig.cmake not found – creating IMPORTED target")
    find_path(CXXOPTS_INCLUDE_DIR cxxopts.hpp REQUIRED)
    add_library(cxxopts::cxxopts INTERFACE IMPORTED)
    set_target_properties(cxxopts::cxxopts
        PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${CXXOPTS_INCLUDE_DIR}")
endif()

find_package(unordered_dense QUIET CONFIG)
if(NOT unordered_dense_FOUND)
    message(STATUS "unordered_denseConfig.cmake not found – creating IMPORTED target")
    # The sources include <ankerl/unordered_dense.h>, so that is what has to
    # be found -- the directory above ankerl/, not the one holding the file.
    find_path(UNORD_DENSE_INCLUDE_DIR ankerl/unordered_dense.h REQUIRED)
    add_library(unordered_dense::unordered_dense INTERFACE IMPORTED)
    set_target_properties(unordered_dense::unordered_dense
        PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${UNORD_DENSE_INCLUDE_DIR}")
endif()

find_library(CCFITS_LIB  CCfits   REQUIRED)
find_library(CFITSIO_LIB cfitsio  REQUIRED)
# The headers too: they sit on the default include path on Linux and under the
# Homebrew prefix on macOS, but a CCfits built from source into its own prefix
# (MSYS2 has no package) was found as a library and then failed to compile.
find_path(CCFITS_INCLUDE_DIR CCfits/CCfits)

# ---------------------------------------------------------------------------
# 5. Core library (no Python, no matplotlib) – this is what ASTRA links
# ---------------------------------------------------------------------------
add_library(GAELcore
    src/Spectrum.cpp
    src/MemoryBudget.cpp
    src/SpectrumCache.cpp
    src/NyquistGrid.cpp
    src/Rebin.cpp
    src/AkimaSpline.cpp
    src/RotationalConvolution.cpp
    src/ContinuumModel.cpp
    src/ModelGrid.cpp
    src/TelluricGrid.cpp
    src/SyntheticModel.cpp
    src/FitParameters.cpp
    src/Observation.cpp
    src/JsonUtils.cpp
    src/MultiDatasetCost.cpp
    src/ContinuumUtils.cpp
    src/Resolution.cpp
    src/Chi2Utils.cpp
    src/FitProgress.cpp
    src/UnifiedFitWorkflow.cpp
    src/SpectrumLoaders.cpp
    src/GaelAPI.cpp
    src/GaelAPI_json.cpp       # JSON adapters (separate TU)
    src/GaelAPI_result_json.cpp # FitResult <-> JSON for out-of-process fits
    src/GridDataProvider.cpp   # pluggable remote grid file source
    src/Platform.cpp           # executable path, per OS
)
add_library(GAEL::core ALIAS GAELcore)

target_include_directories(GAELcore
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE extern)
if(CCFITS_INCLUDE_DIR)
    target_include_directories(GAELcore PRIVATE ${CCFITS_INCLUDE_DIR})
endif()

target_compile_definitions(GAELcore PUBLIC
    EIGEN_USE_BLAS EIGEN_USE_LAPACKE)

target_link_libraries(GAELcore PUBLIC
    Eigen3::Eigen
    OpenBLAS::OpenBLAS
    ${CCFITS_LIB}
    ${CFITSIO_LIB}
    nlohmann_json::nlohmann_json
    Boost::boost Boost::filesystem
    TBB::tbb
    unordered_dense::unordered_dense
    OpenMP::OpenMP_CXX)

# ---------------------------------------------------------------------------
# 6. Optional CUDA attach to core
# ---------------------------------------------------------------------------
if(GAEL_ENABLE_CUDA AND CUDAToolkit_FOUND AND CMAKE_CUDA_COMPILER)
    target_sources(GAELcore PRIVATE src/ResolutionCUDA.cu)
    set_source_files_properties(src/ResolutionCUDA.cu PROPERTIES
        LANGUAGE CUDA CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON
        CUDA_SEPARABLE_COMPILATION ON)
    target_compile_options(GAELcore PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr
            --diag-suppress=20014 --diag-suppress=20015
            -Wno-deprecated-gpu-targets>)
    target_compile_definitions(GAELcore PRIVATE GAEL_USE_CUDA)
    target_link_libraries(GAELcore PRIVATE CUDA::cudart)
endif()

# ---------------------------------------------------------------------------
# 7. Report library (LaTeX + matplotlib-cpp). Optional – ASTRA doesn't need it.
# ---------------------------------------------------------------------------
if(GAEL_BUILD_REPORT)
    add_library(GAELreport src/ReportUtils.cpp)
    add_library(GAEL::report ALIAS GAELreport)
    target_include_directories(GAELreport PRIVATE extern)
    target_link_libraries(GAELreport
        PUBLIC  GAELcore
        PRIVATE Python3::Python Python3::NumPy)
    target_compile_definitions(GAELreport PRIVATE
        GAEL_INSTALL_DATADIR=\"${CMAKE_INSTALL_FULL_DATADIR}/GAEL\")
endif()

# ---------------------------------------------------------------------------
# 8. CLI executable
# ---------------------------------------------------------------------------
add_executable(GAEL src/main.cpp)
target_include_directories(GAEL PRIVATE extern)
target_link_libraries(GAEL PRIVATE GAELcore cxxopts::cxxopts)

# Version stamp for `GAEL --version`, which host applications use to check
# worker-bundle compatibility.  git describe when available, else a fallback.
execute_process(COMMAND git describe --tags --always --dirty
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                OUTPUT_VARIABLE GAEL_GIT_DESCRIBE
                OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET)
if(NOT GAEL_GIT_DESCRIBE)
    set(GAEL_GIT_DESCRIBE "unknown")
endif()
target_compile_definitions(GAEL PRIVATE
    GAEL_VERSION_STRING="${GAEL_GIT_DESCRIBE}")
if(GAEL_BUILD_REPORT)
    target_link_libraries(GAEL PRIVATE GAELreport Python3::Python)
    target_compile_definitions(GAEL PRIVATE GAEL_HAVE_REPORT)
endif()

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/global_settings.json
               ${CMAKE_CURRENT_BINARY_DIR}/global_settings.json COPYONLY)

add_executable(mock_data_generator src/mock_data_generator.cpp)
target_link_libraries(mock_data_generator PRIVATE GAELcore cxxopts::cxxopts)

# Recovery-test data for the features that have no ISIS reference: element
# abundances and multi-component fits.  See tests/gen_mock_xshooter.cpp.
add_executable(gen_mock_xshooter tests/gen_mock_xshooter.cpp)
target_link_libraries(gen_mock_xshooter PRIVATE GAELcore cxxopts::cxxopts
                      nlohmann_json::nlohmann_json)

# ---------------------------------------------------------------------------
# 8b. Tests (ctest)
# ---------------------------------------------------------------------------
option(GAEL_BUILD_TESTS "Register the ctest suite" ${PROJECT_IS_TOP_LEVEL})
if(GAEL_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# ---------------------------------------------------------------------------
# 9. Install & export
# ---------------------------------------------------------------------------
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(_gael_targets GAELcore)
if(GAEL_BUILD_REPORT)
    list(APPEND _gael_targets GAELreport)
endif()

install(TARGETS ${_gael_targets} GAEL mock_data_generator
        EXPORT  GAELTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# FIX: headers are .hpp, not .h
install(DIRECTORY include/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        FILES_MATCHING PATTERN "*.hpp")

install(FILES global_settings.json DESTINATION ${CMAKE_INSTALL_BINDIR})
if(GAEL_BUILD_REPORT)
    install(PROGRAMS src/GAEL_multiplot.py
            DESTINATION ${CMAKE_INSTALL_DATADIR}/GAEL)
endif()

install(EXPORT GAELTargets
        FILE       GAELTargets.cmake
        NAMESPACE  GAEL::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/GAEL)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/GAELConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/GAELConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/GAEL)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/GAELConfigVersion.cmake
    VERSION 0.1.0 COMPATIBILITY SameMajorVersion)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/GAELConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/GAELConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/GAEL)
