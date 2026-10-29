#pragma once
#include <array>
#include <cstddef>
#include <functional>
#include <mutex>
#include <string>

namespace specfit {

/* ------------------------------------------------------------------ *
 *  Hold this from before a CCfits::FITS is constructed until after it is
 *  destroyed.
 *
 *  cfitsio shares one FITSfile -- buffers, current-HDU state -- between all
 *  handles a process has open on the same file name (fits_already_open),
 *  and a reentrant build only guards its global tables, not that shared
 *  structure: "multiple threads must not access the same FITS file at the
 *  same time".  Fits running side by side in one process open the same
 *  grid.fits, lambda.fits and corner files constantly, and that aborted a
 *  12-fit ASTRA run with "malloc(): unaligned tcache chunk detected" while
 *  one thread read grid.fits and five others closed their handles on it.
 *
 *  Striped by the path actually opened (after localize_grid_path), so
 *  different files are still read in parallel.  Recursive, because two
 *  files can share a stripe and nothing should deadlock on that.
 * ------------------------------------------------------------------ */
inline std::unique_lock<std::recursive_mutex>
lock_fits_file(const std::string& local_path)
{
    static std::array<std::recursive_mutex, 64> stripes;
    const std::size_t i = std::hash<std::string>{}(local_path) % stripes.size();
    return std::unique_lock<std::recursive_mutex>(stripes[i]);
}

} // namespace specfit
