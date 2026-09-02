#pragma once
#include <memory>
#include <string>
#include <vector>

namespace specfit {

/*  Pluggable source for grid files that do not live on the local filesystem.
 *
 *  Grid base paths may carry a scheme prefix (e.g. "ssh://host/abs/path");
 *  the provider recognises such paths via handles() and materialises the
 *  files behind them on local disk on demand.  GAEL itself stays free of any
 *  transport dependency: the host application implements this interface and
 *  installs it either process-wide via set_grid_data_provider() or per
 *  session through api::GlobalSettings::grid_provider.
 *
 *  Plain local paths are never routed through a provider, so all existing
 *  behaviour is unchanged when none is installed.
 *
 *  Thread safety: localize()/exists()/prefetch() are called concurrently
 *  from worker threads (OpenMP/TBB); implementations must be thread-safe and
 *  should single-flight concurrent requests for the same path.
 */
class GridDataProvider {
public:
    virtual ~GridDataProvider() = default;

    /*  True if this provider serves the given path (scheme match on the
     *  full virtual path, e.g. anything starting "ssh://").               */
    virtual bool handles(const std::string& path) const = 0;

    /*  Existence probe used by grid resolution; must not throw for a
     *  merely missing file.                                               */
    virtual bool exists(const std::string& virtual_path) = 0;

    /*  Blocking: ensure the file is available on local disk and return the
     *  local filesystem path to open.  Throws std::runtime_error on
     *  unrecoverable failure (the fit surfaces it as an error).           */
    virtual std::string localize(const std::string& virtual_path) = 0;

    /*  Non-blocking batch hint: the given files are about to be
     *  localize()d.  Lets the provider pipeline transfers.  Default no-op. */
    virtual void prefetch(const std::vector<std::string>& virtual_paths)
    { (void)virtual_paths; }
};

/*  Process-wide provider slot (thread-safe).  A null pointer clears it.   */
void set_grid_data_provider(std::shared_ptr<GridDataProvider> provider);
std::shared_ptr<GridDataProvider> grid_data_provider();

/*  Helpers used by every grid read chokepoint.  For a path no provider
 *  handles they are exactly the identity / std::filesystem behaviour.     */
std::string localize_grid_path(const std::string& path);
bool        grid_path_exists(const std::string& path);
void        prefetch_grid_paths(const std::vector<std::string>& paths);

} // namespace specfit
