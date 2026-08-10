cmake_minimum_required(VERSION 3.18)
cmake_policy(SET CMP0074 NEW)       # honour *_ROOT / CMAKE_PREFIX_PATH

project(GAEL LANGUAGES C CXX Fortran)

if(POLICY CMP0167)
    cmake_policy(SET CMP0167 OLD)
endif()

# ---------------------------------------------------------------------------
# 0. Tell CMake where Home-brew lives (needed only on macOS)
# ---------------------------------------------------------------------------
if(APPLE)
    foreach(_p "/opt/homebrew" "/usr/local")
        if(EXISTS "${_p}")
            list(APPEND CMAKE_PREFIX_PATH "${_p}")
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
    add_compile_options(
        -march=native
        -funroll-loops
        -ftree-vectorize
    )
endif()
add_compile_options(
    -Wno-deprecated-declarations
    -Wno-dev
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
find_package(Eigen3        REQUIRED)
find_package(OpenMP        COMPONENTS C CXX Fortran)   # “REQUIRED” later
find_package(OpenBLAS      REQUIRED)
find_package(Boost         REQUIRED CONFIG COMPONENTS filesystem)
find_package(nlohmann_json REQUIRED)
find_package(Python3       REQUIRED COMPONENTS Interpreter Development NumPy)
find_package(TBB           REQUIRED)

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
    find_path(UNORD_DENSE_INCLUDE_DIR unordered_dense.h REQUIRED)
    add_library(unordered_dense::unordered_dense INTERFACE IMPORTED)
    set_target_properties(unordered_dense::unordered_dense
        PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${UNORD_DENSE_INCLUDE_DIR}")
endif()

find_library(CCFITS_LIB  CCfits   REQUIRED)
find_library(CFITSIO_LIB cfitsio  REQUIRED)

# ---------------------------------------------------------------------------
# 5. Core library (no Python, no matplotlib) – this is what ASTRA links
# ---------------------------------------------------------------------------
add_library(GAELcore
    src/Spectrum.cpp
    src/SpectrumCache.cpp
    src/NyquistGrid.cpp
    src/Rebin.cpp
    src/AkimaSpline.cpp
    src/RotationalConvolution.cpp
    src/ContinuumModel.cpp
    src/ModelGrid.cpp
    src/SyntheticModel.cpp
    src/FitParameters.cpp
    src/Observation.cpp
    src/JsonUtils.cpp
    src/MultiDatasetCost.cpp
    src/ContinuumUtils.cpp
    src/Resolution.cpp
    src/Chi2Utils.cpp
    src/UnifiedFitWorkflow.cpp
    src/SpectrumLoaders.cpp
    src/GaelAPI.cpp
    src/GaelAPI_json.cpp       # JSON adapters (separate TU)
)
add_library(GAEL::core ALIAS GAELcore)

target_include_directories(GAELcore
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE extern)

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
    unordered_dense::unordered_dense)

if(OpenMP_CXX_FOUND)
    target_link_libraries(GAELcore PUBLIC OpenMP::OpenMP_CXX)
endif()

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
option(GAEL_BUILD_REPORT "Build the PDF/plot reporting helpers" ON)
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
if(GAEL_BUILD_REPORT)
    target_link_libraries(GAEL PRIVATE GAELreport Python3::Python)
    target_compile_definitions(GAEL PRIVATE GAEL_HAVE_REPORT)
endif()

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/global_settings.json
               ${CMAKE_CURRENT_BINARY_DIR}/global_settings.json COPYONLY)

add_executable(mock_data_generator src/mock_data_generator.cpp)
target_link_libraries(mock_data_generator PRIVATE GAELcore cxxopts::cxxopts)

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
