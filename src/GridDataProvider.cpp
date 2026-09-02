#include "specfit/GridDataProvider.hpp"

#include <filesystem>
#include <mutex>

namespace specfit {

namespace {
std::mutex                        g_provider_mtx;
std::shared_ptr<GridDataProvider> g_provider;
} // namespace

void set_grid_data_provider(std::shared_ptr<GridDataProvider> provider)
{
    std::lock_guard<std::mutex> lk(g_provider_mtx);
    g_provider = std::move(provider);
}

std::shared_ptr<GridDataProvider> grid_data_provider()
{
    std::lock_guard<std::mutex> lk(g_provider_mtx);
    return g_provider;
}

std::string localize_grid_path(const std::string& path)
{
    if (auto p = grid_data_provider(); p && p->handles(path))
        return p->localize(path);
    return path;
}

bool grid_path_exists(const std::string& path)
{
    if (auto p = grid_data_provider(); p && p->handles(path))
        return p->exists(path);
    return std::filesystem::exists(path);
}

void prefetch_grid_paths(const std::vector<std::string>& paths)
{
    if (paths.empty()) return;
    auto p = grid_data_provider();
    if (!p) return;
    /*  All chokepoints prefetch corners of one grid at a time, so testing
     *  the first path is enough to skip the call for local grids.         */
    if (!p->handles(paths.front())) return;
    p->prefetch(paths);
}

} // namespace specfit
