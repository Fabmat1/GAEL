# ---------------------------------------------------------------------------
# GAEL regression tests: fit results compared against ISIS reference fits.
#
# Both tests are slow on their very first run (they have to compute the ISIS
# references) and fast afterwards, because those references are cached under
# ${GAEL_TEST_CACHE}.  They report exit code 77 -> "skipped" whenever a
# prerequisite (ISIS, the model grid, the ASTRA database) is unavailable, so a
# machine without ISIS still gets a clean `ctest` run.
# ---------------------------------------------------------------------------

set(GAEL_TEST_CACHE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/cache"
    CACHE PATH "Where cached ISIS reference fits and mock data are stored")

set(GAEL_TEST_ISIS_BIN "" CACHE FILEPATH
    "ISIS binary used to produce reference fits (auto-detected when empty)")

set(GAEL_TEST_ASTRA_DB "" CACHE FILEPATH
    "ASTRA database holding the real RVVD spectra (auto-detected when empty)")

# The harness resolves everything else itself; we only pass what CMake knows
# better, i.e. the freshly built binaries and the configured settings file.
set(_gael_test_env
    "GAEL_TEST_GAEL_BIN=$<TARGET_FILE:GAEL>"
    "GAEL_TEST_MOCKGEN_BIN=$<TARGET_FILE:mock_data_generator>"
    "GAEL_TEST_XSHOOTER_GEN=$<TARGET_FILE:gen_mock_xshooter>"
    "GAEL_TEST_GLOBAL_SETTINGS=${CMAKE_BINARY_DIR}/global_settings.json"
    "GAEL_TEST_CACHE=${GAEL_TEST_CACHE_DIR}"
)
if(GAEL_TEST_ISIS_BIN)
    list(APPEND _gael_test_env "GAEL_TEST_ISIS=${GAEL_TEST_ISIS_BIN}")
endif()
if(GAEL_TEST_ASTRA_DB)
    list(APPEND _gael_test_env "GAEL_TEST_ASTRA_DB=${GAEL_TEST_ASTRA_DB}")
endif()

function(gael_add_comparison_test name script)
    add_test(NAME ${name}
             COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/${script}"
             WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
    set_tests_properties(${name} PROPERTIES
        ENVIRONMENT "${_gael_test_env}"
        SKIP_RETURN_CODE 77
        # First run computes ~200 ISIS fits; later runs hit the cache.
        TIMEOUT 21600
        LABELS "comparison;slow")
endfunction()

# Fast, hermetic unit test: GridDataProvider routing and caching.
add_executable(test_grid_provider test_grid_provider.cpp)
target_link_libraries(test_grid_provider PRIVATE GAELcore)
add_test(NAME grid_provider COMMAND test_grid_provider)
set_tests_properties(grid_provider PROPERTIES TIMEOUT 120 LABELS "unit")

# Fast, hermetic unit test: missing files inside a model grid are bridged,
# unreadable ones still fail the fit.
add_executable(test_grid_gaps test_grid_gaps.cpp)
target_link_libraries(test_grid_gaps PRIVATE GAELcore)
add_test(NAME grid_gaps COMMAND test_grid_gaps)
set_tests_properties(grid_gaps PROPERTIES TIMEOUT 300 LABELS "unit")

# Fast, hermetic unit test: cgroup v1/v2 limits and headroom for the budget.
add_executable(test_memory_budget test_memory_budget.cpp)
target_link_libraries(test_memory_budget PRIVATE GAELcore)
add_test(NAME memory_budget COMMAND test_memory_budget)
set_tests_properties(memory_budget PROPERTIES TIMEOUT 60 LABELS "unit")

# Fast, hermetic unit test: JSON round trips for out-of-process fitting.
add_executable(test_json_roundtrip test_json_roundtrip.cpp)
target_link_libraries(test_json_roundtrip PRIVATE GAELcore
                      nlohmann_json::nlohmann_json)
add_test(NAME json_roundtrip COMMAND test_json_roundtrip)
set_tests_properties(json_roundtrip PROPERTIES TIMEOUT 120 LABELS "unit")

# Fast, hermetic unit test: the Levenberg-Marquardt stopping rules.
add_executable(test_lm_convergence test_lm_convergence.cpp)
target_link_libraries(test_lm_convergence PRIVATE GAELcore)
add_test(NAME lm_convergence COMMAND test_lm_convergence)
set_tests_properties(lm_convergence PROPERTIES TIMEOUT 120 LABELS "unit")

gael_add_comparison_test(gael_vs_isis_mock test_mock_vs_isis.py)
gael_add_comparison_test(gael_vs_isis_real test_real_vs_isis.py)

# Element abundances and multi-component fits.  These need a metal-bearing
# grid (the Feros ones), which the sdB suites above do not have, so they skip
# cleanly on a machine that only has sdB/processed.
gael_add_comparison_test(gael_vs_isis_xshooter test_xshooter_vs_isis.py)

# A fast subset for "did I break the harness itself" checks.
add_test(NAME gael_vs_isis_smoke
         COMMAND ${Python3_EXECUTABLE}
                 "${CMAKE_CURRENT_SOURCE_DIR}/test_mock_vs_isis.py"
                 --kinds single --limit 4
         WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
set_tests_properties(gael_vs_isis_smoke PROPERTIES
    ENVIRONMENT "${_gael_test_env}"
    SKIP_RETURN_CODE 77
    TIMEOUT 1800
    LABELS "comparison")

# Fast, hermetic unit test: fits in one process opening the same FITS files
# concurrently (cfitsio shares one FITSfile per name across handles).
add_executable(test_fits_concurrency test_fits_concurrency.cpp)
target_link_libraries(test_fits_concurrency PRIVATE GAELcore)
add_test(NAME fits_concurrency COMMAND test_fits_concurrency)
set_tests_properties(fits_concurrency PROPERTIES TIMEOUT 300 LABELS "unit")

# Measurement tool, not a test: accuracy of the metal-free model pipeline.
add_executable(bench_hhe_order bench_hhe_order.cpp)
target_link_libraries(bench_hhe_order PRIVATE GAELcore)

# Measurement tool: dump model spectra to compare two builds bit for bit.
add_executable(dump_models dump_models.cpp)
target_link_libraries(dump_models PRIVATE GAELcore)
