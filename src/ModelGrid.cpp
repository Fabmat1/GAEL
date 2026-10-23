#include "specfit/ModelGrid.hpp"
#include "specfit/GridDataProvider.hpp"
#include "specfit/Resolution.hpp"
#include "specfit/RotationalConvolution.hpp"  // Added include
#include "specfit/SpectrumCache.hpp"

#include <CCfits/CCfits>
#include <filesystem>
#include <sstream>
#include <stdexcept>
#include <iostream>
#include <iomanip>
#include <unordered_map>
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <array>
#include <atomic>
#include <functional>
#include <limits>
#include <mutex>
#include <optional>
#include <system_error>
#include <unordered_set>

namespace fs = std::filesystem;

/* ---------- little helpers outside namespace ----------------------- */
static std::string fmt(double v, int prec)
{
    std::ostringstream s; s << std::fixed << std::setprecision(prec) << v;
    return s.str();
}

/* ---------- hashing helpers for (path,vsini,resOffset,resSlope) ---------- */
static std::size_t hash_combine(std::size_t seed, std::size_t v) noexcept
{
    seed ^= v + 0x9E3779B97F4A7C15ULL + (seed << 6) + (seed >> 2);
    return seed;
}
static std::size_t hash_double(double x) noexcept
{
    std::uint64_t bits; std::memcpy(&bits, &x, sizeof bits);
    return std::hash<std::uint64_t>{}(bits);
}

/* =================================================================== */
namespace specfit {

/* -------- small Eigen wrapper -------------------------------------- */
static Vector to_eigen(const std::vector<Real>& v)
{ return Eigen::Map<const Vector>(v.data(), v.size()); }

/* -------- resolve grid base path ----------------------------------- */
static std::string
resolve_grid(const std::vector<std::string>& bases,
             const std::string& rel_path)
{
    for (const auto& b : bases) {
        fs::path p = fs::path(b) / rel_path;
        /*  grid_path_exists routes bases a GridDataProvider handles (e.g.
         *  "ssh://host/...") through the provider; plain paths keep the
         *  fs::exists behaviour.  base_ then stays the virtual path and
         *  every later read localizes it just before opening.             */
        if (grid_path_exists((p / "grid.fits").string())) return p.string();
    }
    throw std::runtime_error("Grid '" + rel_path + "' not found.");
}

/*  The missing files this grid object has interpolated across. */
struct ModelGrid::GapLog {
    std::mutex           mtx;
    std::vector<GridGap> used;
};

/* =======================  ctor helpers  ============================ */
ModelGrid::ModelGrid(const std::vector<std::string>& bases,
                     const std::string& rel_path)
: base_(resolve_grid(bases, rel_path)),
  gap_log_(std::make_shared<GapLog>())
{
    CCfits::FITS f(localize_grid_path(base_ + "/grid.fits"), CCfits::Read);
    CCfits::ExtHDU& ext = f.extension(1);
    for (const auto& [name, col] : ext.column()) {
        std::vector<Real> buf;
        /* --- keep the robust logic from the original implementation --- */
        if (col->varLength())               col->read(buf, 1);      // variable-length vector
        else if (col->repeat() == 1)        col->read(buf, 1, 1);   // scalar column
        else                                col->read(buf, 1);      // fixed-length vector
        axes_.push_back({name, to_eigen(buf)});
        if (!is_stellar_axis(name)) species_.push_back(name);
    }
}
ModelGrid::ModelGrid(std::string abs)
: base_(std::move(abs)), gap_log_(std::make_shared<GapLog>()) {}

/* -------------------- misc internal helpers ------------------------ */
static double param_for_axis(const std::string& n,
                             double teff,double logg,double z,
                             double he,double xi)
{
    if (n=="t") return teff; if (n=="g") return logg;
    if (n=="z") return z;    if (n=="HHE") return he;
    if (n=="x") return xi;
    throw std::runtime_error("Unsupported axis '"+n+"'.");
}

/* ------------------------------------------------------------------ *
 *  Generic corner enumeration over an ordered list of (axis, value).
 *
 *  This used to be five hard-coded fields in a `struct Node`, which is why a
 *  grid.fits carrying an element column crashed the loader.  A corner is now
 *  just a coordinate vector, so the same routine serves the 5-axis HHE cube
 *  and the 6-axis (5 stellar + abundance) cube of one element.
 * ------------------------------------------------------------------ */
namespace {

struct CubeCorner {
    std::vector<double> coord;   // one value per axis, in axis order
    double              weight;
};

std::vector<CubeCorner>
build_corners(const std::vector<const GridAxis*>& axes,
              const std::vector<double>&          values)
{
    std::vector<CubeCorner> nodes;
    nodes.push_back({std::vector<double>(axes.size(), 0.0), 1.0});

    for (std::size_t a = 0; a < axes.size(); ++a) {
        const Vector& grid = axes[a]->values;
        const double  p    = values[a];

        if (grid.size() == 1) {                    // pinned axis
            for (auto& nd : nodes) nd.coord[a] = grid[0];
            continue;
        }

        auto it = std::lower_bound(grid.data(), grid.data() + grid.size(), p);
        int hi = (it == grid.data() + grid.size()) ? int(grid.size()) - 1
                                                   : int(it - grid.data());
        int lo = (hi == 0) ? 0 : hi - 1;

        const double a_hi = (grid[hi] == grid[lo]) ? 0.0
                          : (p - grid[lo]) / (grid[hi] - grid[lo]);
        const double a_lo = 1.0 - a_hi;

        const std::size_t n = nodes.size();
        nodes.resize(n * 2);
        for (std::size_t i = 0; i < n; ++i) {
            nodes[n + i] = nodes[i];
            nodes[i    ].coord[a] = grid[lo];  nodes[i    ].weight *= a_lo;
            nodes[n + i].coord[a] = grid[hi];  nodes[n + i].weight *= a_hi;
        }
    }
    return nodes;
}

/* =================================================================== *
 *  Holes in a grid
 *
 *  grid.fits lists the values of every axis, but nothing guarantees a model
 *  file exists for every combination: a failed synthesis run leaves a single
 *  hole, and an index value can have no files at all.  A corner without a
 *  file used to abort the fit.
 *
 *  The rule.  A missing node m is given the value linear interpolation
 *  along one axis would give it,
 *
 *      f(m) = [(v+ - v_m) f(m-) + (v_m - v-) f(m+)] / (v+ - v-),
 *
 *  where m- and m+ are its two neighbours on that axis (v- and v+ their
 *  coordinates on it).  Both neighbours must exist; a neighbour that is
 *  itself missing is not looked through, so the substitute is never more than
 *  one node away.  Among the axes where that works, the one with the
 *  narrowest bridged interval wins, measured in units of the axis's mean
 *  node spacing so that K and dex compare.  Within 1e-9 of a tie the cube's
 *  first axis wins: grid.fits column order for the HHE cube, directory order
 *  (z, HHE, x, g, t, abundance) for an element cube.  No such axis: the
 *  evaluation fails with a message naming the grid point.
 *
 *  Why a value per node rather than a per-cell patch: the choice depends
 *  only on the node and on which files exist, never on the cell being
 *  evaluated or the order things were discovered in, so every cell touching
 *  m sees the same f(m) and the interpolant stays continuous across all cell
 *  boundaries -- the solver sees no step anywhere.  It is also exactly
 *  "stretch the bracket over the wider interval": in a cell with m at one
 *  end of an edge along the bridged axis, (1 - a) f(m) + a f(partner)
 *  is linear interpolation from m- to m+.  So an index value with no files
 *  at all -- every node in it bridged along its own axis -- behaves as if
 *  grid.fits did not list it (to rounding; the weights are summed in a
 *  different order).
 *
 *  Mechanically the missing corner is replaced by its two neighbours with
 *  its weight split between them, merged into the corner list where a
 *  neighbour is already a corner of the cell.  A grid without holes never
 *  enters any of this and computes exactly what it always did.
 *
 *  Missing is not unreadable.  Only "this file does not exist" (ENOENT, or
 *  a GridDataProvider whose exists() says no) is a hole.  Anything else --
 *  EIO, a dead sshfs mount, a truncated file, a provider that cannot reach
 *  its host -- fails the fit as before; bridging those would silently turn
 *  an outage into different models.
 *
 *  What is learnt is kept for the life of the process: which files are
 *  missing, which neighbours were found present, and the bridge chosen for
 *  each hole.  A hole costs one failed read and a few existence probes the
 *  first time and a lookup afterwards, which matters most through a remote
 *  provider where each probe is a round trip.
 * =================================================================== */

/*  The corner file is definitely not there.  Internal: the corner loops
 *  catch it, register the hole and plan the cube again.                    */
struct MissingGridFile : std::runtime_error {
    explicit MissingGridFile(const std::string& path)
        : std::runtime_error("model file '" + path + "' does not exist") {}
};

enum class Presence { Present, Absent };

/*  Absent only when that is certain; throws when it cannot tell.          */
Presence probe_grid_file(const std::string& path)
{
    if (auto p = grid_data_provider(); p && p->handles(path))
        return p->exists(path) ? Presence::Present : Presence::Absent;

    std::error_code ec;
    const fs::file_status st = fs::status(path, ec);
    if (!ec)
        return st.type() == fs::file_type::not_found ? Presence::Absent
                                                     : Presence::Present;
    if (ec == std::errc::no_such_file_or_directory) return Presence::Absent;
    throw std::runtime_error("cannot tell whether model file '" + path +
                             "' exists: " + ec.message());
}

using Coord  = std::vector<double>;
using PathOf = std::function<std::string(const Coord&)>;

struct Bridge { std::size_t axis; double lo, hi; };

/*  One cube's holes, keyed by the directory its files live under
 *  ("<grid>/HHE", "<grid>/FE"): two ModelGrid objects reading the same
 *  files share what either learnt.                                         */
struct CubeGaps {
    std::vector<Coord>                    absent;
    std::vector<std::pair<Coord, Bridge>> bridges;
};

std::mutex                                g_gap_mtx;
std::unordered_map<std::string, CubeGaps> g_gaps;
std::unordered_map<std::string, bool>     g_probed;   // neighbour path -> exists
std::unordered_set<std::string>           g_warned;
std::atomic<bool>                         g_any_gaps{false};

bool contains(const std::vector<Coord>& v, const Coord& c)
{
    return std::find(v.begin(), v.end(), c) != v.end();
}

bool cube_has_gaps(const std::string& root)
{
    std::lock_guard<std::mutex> lk(g_gap_mtx);
    auto it = g_gaps.find(root);
    return it != g_gaps.end() && !it->second.absent.empty();
}

void register_absent(const std::string& root, const Coord& c,
                     const std::string& path)
{
    std::lock_guard<std::mutex> lk(g_gap_mtx);
    CubeGaps& cg = g_gaps[root];
    g_probed[path] = false;
    if (!contains(cg.absent, c)) {
        cg.absent.push_back(c);
        /*  A bridge chosen earlier may lean on this file, if it was probed
         *  present and has since gone; the next lookup chooses again.     */
        cg.bridges.clear();
    }
    g_any_gaps.store(true, std::memory_order_release);
}

/*  Does the node at `c` have a file?  Probed once per process.            */
bool node_present(const std::string& root, const Coord& c,
                  const std::string& path)
{
    {
        std::lock_guard<std::mutex> lk(g_gap_mtx);
        if (auto it = g_gaps.find(root);
            it != g_gaps.end() && contains(it->second.absent, c))
            return false;
        if (auto it = g_probed.find(path); it != g_probed.end())
            return it->second;
    }
    /*  Outside the lock: through a remote provider this is a round trip. */
    if (probe_grid_file(path) == Presence::Absent) {
        register_absent(root, c, path);
        return false;
    }
    std::lock_guard<std::mutex> lk(g_gap_mtx);
    g_probed[path] = true;
    return true;
}

std::string describe_point(const std::vector<const GridAxis*>& axes,
                           const Coord& c)
{
    std::ostringstream s;
    for (std::size_t a = 0; a < axes.size(); ++a)
        s << (a ? ", " : "") << axes[a]->name << '=' << c[a];
    return s.str();
}

/*  The bridge for the missing node `m`: the rule at the top of this
 *  section.  Warns the first time a hole is bridged in this process.      */
Bridge bridge_for(const std::string& root,
                  const std::vector<const GridAxis*>& axes,
                  const Coord& m, const PathOf& path_of)
{
    {
        std::lock_guard<std::mutex> lk(g_gap_mtx);
        if (auto it = g_gaps.find(root); it != g_gaps.end())
            for (const auto& [node, b] : it->second.bridges)
                if (node == m) return b;
    }

    std::optional<Bridge> best;
    double best_score = 0.0;
    for (std::size_t a = 0; a < axes.size(); ++a) {
        const Vector& v = axes[a]->values;
        const Eigen::Index n = v.size();
        if (n < 3) continue;                       // no node has two neighbours

        const Real* it = std::find(v.data(), v.data() + n, m[a]);
        const Eigen::Index k = it - v.data();
        if (k <= 0 || k >= n - 1) continue;        // at the edge of the axis

        Coord lo = m; lo[a] = v[k - 1];
        Coord hi = m; hi[a] = v[k + 1];
        if (!node_present(root, lo, path_of(lo)) ||
            !node_present(root, hi, path_of(hi)))
            continue;

        const double score = (v[k + 1] - v[k - 1]) /
                             ((v[n - 1] - v[0]) / double(n - 1));
        if (!best || score < best_score * (1.0 - 1e-9)) {
            best       = Bridge{a, v[k - 1], v[k + 1]};
            best_score = score;
        }
    }

    const std::string path = path_of(m);
    if (!best)
        throw std::runtime_error(
            "model file '" + path + "' is missing (grid point " +
            describe_point(axes, m) + ") and cannot be interpolated "
            "across: no grid axis has existing models on both sides of it");

    std::lock_guard<std::mutex> lk(g_gap_mtx);
    g_gaps[root].bridges.emplace_back(m, *best);
    if (g_warned.insert(path).second)
        std::cerr << "Warning: model file " << path << " is missing; "
                  << "interpolating across it along " << axes[best->axis]->name
                  << " between " << best->lo << " and " << best->hi << '\n';
    return *best;
}

/*  Replace every weighted corner that is a known hole by its bridge's two
 *  neighbours.  Returns the list untouched -- same corners, same order, same
 *  weights -- when this cube has no known holes.                           */
std::vector<CubeCorner>
bridge_gaps(const std::string& root,
            const std::vector<const GridAxis*>& axes,
            std::vector<CubeCorner> nodes,
            const PathOf& path_of,
            std::vector<GridGap>& used)
{
    if (!g_any_gaps.load(std::memory_order_acquire)) return nodes;

    std::vector<Coord> absent;
    {
        std::lock_guard<std::mutex> lk(g_gap_mtx);
        auto it = g_gaps.find(root);
        if (it == g_gaps.end() || it->second.absent.empty()) return nodes;
        absent = it->second.absent;
    }

    auto add = [&](const Coord& c, double w) {
        for (auto& nd : nodes)
            if (nd.coord == c) { nd.weight += w; return; }
        nodes.push_back({c, w});
    };

    const std::size_t n0 = nodes.size();          // neighbours are never holes
    for (std::size_t i = 0; i < n0; ++i) {
        if (nodes[i].weight == 0.0 || !contains(absent, nodes[i].coord))
            continue;
        const Coord  m = nodes[i].coord;
        const Bridge b = bridge_for(root, axes, m, path_of);

        const double w     = nodes[i].weight;
        const double f_hi  = (m[b.axis] - b.lo) / (b.hi - b.lo);
        nodes[i].weight = 0.0;
        Coord lo = m; lo[b.axis] = b.lo;
        Coord hi = m; hi[b.axis] = b.hi;
        add(lo, w * (1.0 - f_hi));
        add(hi, w * f_hi);

        GridGap g;
        g.file = path_of(m);
        for (std::size_t a = 0; a < axes.size(); ++a)
            g.point.emplace_back(axes[a]->name, m[a]);
        g.axis = axes[b.axis]->name;
        g.from = b.lo;
        g.to   = b.hi;
        used.push_back(std::move(g));
    }
    return nodes;
}

/*  Open one corner file and hand extension 1 to `read`.  A failure is
 *  classified before it propagates: a file that is not there becomes
 *  MissingGridFile, which the corner loops bridge; one that is there but
 *  cannot be read -- or whose existence cannot be established -- fails the
 *  fit.  CCfits's exceptions do not derive from std::exception, so they are
 *  translated here; left alone they went past every handler to
 *  std::terminate.                                                         */
template <typename Read>
void read_corner_file(const std::string& path, Read&& read)
{
    std::string why;
    try {
        CCfits::FITS f(localize_grid_path(path), CCfits::Read);
        read(f.extension(1));
        return;
    }
    catch (const CCfits::FitsException& e) { why = e.message(); }
    catch (const std::exception& e)        { why = e.what(); }

    if (probe_grid_file(path) == Presence::Absent) throw MissingGridFile(path);
    throw std::runtime_error("cannot read model file '" + path + "': " + why);
}

/*  Guards the lazily-filled per-species wavelength grids of every ModelGrid;
 *  a mutex member would make ModelGrid non-movable.                        */
std::mutex g_lambda_mtx;

/*  Which rotational routine the metal-free path uses; see the call site.
 *
 *  Off by default: the segment routine computes the same integral but sums
 *  the duplicate kernel taps in a different order, so it does not reproduce
 *  the numbers the ISIS comparison tests in tests/ were recorded against.
 *  It is the cheaper of the two in both time and memory, so this is worth
 *  turning on once a run of those tests says by how much the results move.
 *
 *  Read once -- it is consulted inside the corner loop.                    */
bool use_rot_segments()
{
    static const bool v = [] {
        const char* e = std::getenv("GAEL_ROT_SEGMENTS");
        return e && std::string(e) == "1";
    }();
    return v;
}

/*  The HHE cube spans the stellar axes in grid.fits order; its file path
 *  names them in directory order, Z/HE/X/G/T.  An axis the grid lacks
 *  reads as 0, as it always did.                                           */
struct HheNode { double t = 0, g = 0, z = 0, h = 0, x = 0; };

struct HheCube {
    std::vector<const GridAxis*> axes;
    int it = -1, ig = -1, iz = -1, ih = -1, ix = -1;   // position in `axes`

    HheNode node(const Coord& c) const
    {
        HheNode n;
        if (it >= 0) n.t = c[it];
        if (ig >= 0) n.g = c[ig];
        if (iz >= 0) n.z = c[iz];
        if (ih >= 0) n.h = c[ih];
        if (ix >= 0) n.x = c[ix];
        return n;
    }
};

HheCube hhe_cube_axes(const std::vector<GridAxis>& all)
{
    HheCube cube;
    for (const auto& ax : all) {
        if (!ModelGrid::is_stellar_axis(ax.name)) continue;
        const int pos = static_cast<int>(cube.axes.size());
        if      (ax.name == "t")   cube.it = pos;
        else if (ax.name == "g")   cube.ig = pos;
        else if (ax.name == "z")   cube.iz = pos;
        else if (ax.name == "HHE") cube.ih = pos;
        else if (ax.name == "x")   cube.ix = pos;
        cube.axes.push_back(&ax);
    }
    return cube;
}

std::string hhe_corner_path(const std::string& base, const HheCube& cube,
                            const Coord& c)
{
    const HheNode nd = cube.node(c);
    std::ostringstream oss;
    oss<<base<<"/HHE"
       <<"/Z"<<fmt(nd.z,2)
       <<"/HE"<<fmt(nd.h,3)
       <<"/X"<<fmt(nd.x,2)
       <<"/G"<<fmt(nd.g,3)
       <<"/T"<<int(std::lround(nd.t))<<".fits";
    return oss.str();
}

/*  <S>/Z%.2f/HE%.3f/X%.2f/G%.3f/T%.0f/<A>.fits, coordinates in that order. */
std::string element_corner_path(const std::string& base,
                                const std::string& name, const Coord& c)
{
    std::ostringstream oss;
    oss << base << '/' << name
        << "/Z"  << fmt(c[0], 2)
        << "/HE" << fmt(c[1], 3)
        << "/X"  << fmt(c[2], 2)
        << "/G"  << fmt(c[3], 3)
        << "/T"  << int(std::lround(c[4]))
        << '/'   << fmt(c[5], 3) << ".fits";
    return oss.str();
}

} // namespace

/* =================================================================== *
 *  Which gaps a fit used
 * =================================================================== */
void ModelGrid::record_gaps(const std::vector<GridGap>& used) const
{
    if (used.empty()) return;
    std::lock_guard<std::mutex> lk(gap_log_->mtx);
    for (const auto& g : used) {
        const bool seen = std::any_of(
            gap_log_->used.begin(), gap_log_->used.end(),
            [&](const GridGap& h) { return h.file == g.file; });
        if (!seen) gap_log_->used.push_back(g);
    }
}

std::vector<GridGap> ModelGrid::gaps_used() const
{
    std::lock_guard<std::mutex> lk(gap_log_->mtx);
    return gap_log_->used;
}

bool ModelGrid::any_gaps_known()
{
    return g_any_gaps.load(std::memory_order_acquire);
}

void ModelGrid::note_gaps(double teff, double logg, double z, double he,
                          double xi,
                          const std::vector<int>&    species,
                          const std::vector<double>& abundances) const
{
    if (!any_gaps_known()) return;

    /*  Corners are only built for cubes known to have holes: once one grid
     *  in the process has one, every model evaluation of every grid passes
     *  through here.                                                       */
    std::vector<GridGap> used;
    if (cube_has_gaps(base_ + "/HHE")) {
        const auto cube = hhe_cube_axes(axes_);
        std::vector<double> vals;
        for (const GridAxis* ax : cube.axes)
            vals.push_back(param_for_axis(ax->name, teff, logg, z, he, xi));
        (void)bridge_gaps(base_ + "/HHE", cube.axes,
                          build_corners(cube.axes, vals),
                          [&](const Coord& c) {
                              return hhe_corner_path(base_, cube, c);
                          },
                          used);
    }
    for (int s : species) {
        if (s < 0 || s >= static_cast<int>(species_.size()) ||
            s >= static_cast<int>(abundances.size()))
            continue;
        const std::string& name = species_[static_cast<std::size_t>(s)];
        if (!cube_has_gaps(base_ + '/' + name)) continue;
        const auto axes = element_cube_axes(s);
        const std::vector<double> vals = {
            z, he, xi, logg, teff, abundances[static_cast<std::size_t>(s)] };
        (void)bridge_gaps(base_ + '/' + name, axes, build_corners(axes, vals),
                          [&](const Coord& c) {
                              return element_corner_path(base_, name, c);
                          },
                          used);
    }
    record_gaps(used);
}

/*  The axes an element cube spans: the five stellar ones, then the
 *  element's own abundance axis (ISIS's interpol_syn appends "/%.3f" to the
 *  path for exactly this reason).  Path order is fixed by the directory
 *  layout, so the axis order is too.                                       */
std::vector<const GridAxis*> ModelGrid::element_cube_axes(int s) const
{
    const std::string& name = species_[static_cast<std::size_t>(s)];
    const GridAxis* ax_z = nullptr; const GridAxis* ax_he = nullptr;
    const GridAxis* ax_x = nullptr; const GridAxis* ax_g  = nullptr;
    const GridAxis* ax_t = nullptr; const GridAxis* ax_a  = nullptr;

    for (const auto& ax : axes_) {
        if      (ax.name == "z")   ax_z  = &ax;
        else if (ax.name == "HHE") ax_he = &ax;
        else if (ax.name == "x")   ax_x  = &ax;
        else if (ax.name == "g")   ax_g  = &ax;
        else if (ax.name == "t")   ax_t  = &ax;
        else if (ax.name == name)  ax_a  = &ax;
    }
    if (!ax_z || !ax_he || !ax_x || !ax_g || !ax_t || !ax_a)
        throw std::runtime_error("grid '" + base_ + "' lacks an axis needed "
                                 "for element '" + name + "'");
    return { ax_z, ax_he, ax_x, ax_g, ax_t, ax_a };
}

/* =================================================================== */
Spectrum ModelGrid::load_spectrum(double teff,double logg,double z,
                                  double he,double xi,double vsini,
                                  double resOffset,double resSlope,
                                  bool   with_continuum) const
{
    //std::cout << "[LoadSpec] Building Interpolation Hypercube." << std::endl;
    /* ---------- build interpolation hyper-cube -------------------- */
    /*  Element axes are not dimensions of *this* cube: ISIS factorises them
     *  out into one cube per element whose spectra multiply onto this one
     *  (see load_element_factor).                                         */
    const auto cube = hhe_cube_axes(axes_);
    std::vector<double> vals;
    for (const GridAxis* ax : cube.axes)
        vals.push_back(param_for_axis(ax->name,teff,logg,z,he,xi));

    const std::vector<CubeCorner> corners = build_corners(cube.axes, vals);
    //std::cout << "[LoadSpec] Corner fetching." << std::endl;

    const PathOf corner_path = [&](const Coord& c) {
        return hhe_corner_path(base_, cube, c);
    };

    /* ---------- helper to fetch / cache one corner ---------------- */
    /*  nullptr: the file turned out not to exist (now registered).       */
    auto fetch = [&](const CubeCorner& cn)->SpectrumPtr {
        //std::cout << "[LoadSpec](fetch) Fetching..." << std::endl;
        const HheNode nd = cube.node(cn.coord);

        int Ti=int(std::lround(nd.t));
        int Gi=int(std::lround(nd.g*100));
        int Zi=int(std::lround(nd.z*100));
        int Hi=int(std::lround(nd.h*1000));
        int Xi=int(std::lround(nd.x*100));

        using Key = std::uint64_t;
        Key k = Key(Ti) | (Key(Gi)<<12) | (Key(Zi)<<24)
                        | (Key(Hi)<<36) | (Key(Xi)<<48);
        /*  The node coordinates alone do not identify a corner: two grids
         *  with the same axis values (sdB/processed vs processed_25, or a
         *  local and a streamed copy of one grid) would collide in the
         *  process-wide cache and serve each other's spectra.              */
        k = hash_combine(k, std::hash<std::string>{}(base_));
        k = hash_combine(k, hash_combine(0x5152A710ULL, std::size_t(with_continuum)));
        k = hash_combine(k, hash_double(vsini));      // Include vsini in cache key
        k = hash_combine(k, hash_double(resOffset));
        k = hash_combine(k, hash_double(resSlope));
        k = hash_combine(k, window_key_);             // slice is part of the key

        //std::cout << "[LoadSpec](fetch) Entry hashes done..." << std::endl;
        if (auto sp = SpectrumCache::instance().try_get(k))
            return sp;                                          // global hit

        //std::cout << "[LoadSpec](fetch) Cache miss, calculating." << std::endl;
        const std::string path = corner_path(cn.coord);
        try {
                return SpectrumCache::instance()
                         .insert_if_absent(k,[&]{
                             //std::cout << "[LoadSpec](fetch) Reading .fits ." << std::endl;
                             Spectrum raw = read_fits(path, with_continuum);
                             //std::cout << "[LoadSpec](fetch) Broadening." << vsini << std::endl;

                             // Apply rotational broadening FIRST
                             Vector rot_flux;
                             if (vsini >= 0.1){
                                //std::cout << "[LoadSpec](fetch) Broadeningggg." << vsini << std::endl;
                                /*  Two ways to compute the same integral.
                                 *
                                 *  The single-grid routine keeps a
                                 *  (index, weight) pair per kernel tap in its
                                 *  own std::vector per model pixel.  On a grid
                                 *  as finely sampled as synthe's, with vsini at
                                 *  100 km/s, the kernel spans ~220 pixels and
                                 *  that table is ~600 MB -- and vsini is part of
                                 *  its cache key, so the solver builds a fresh
                                 *  one every iteration and uses it once.
                                 *
                                 *  The two-grid routine is the same integral
                                 *  stored as one contiguous run of weights per
                                 *  output point in a single flat array, built
                                 *  with a monotone cursor instead of a binary
                                 *  search and a sort per point.  It is what the
                                 *  metal path already uses.  Accumulating the
                                 *  duplicate taps rather than sort-then-merging
                                 *  them reorders the floating-point sum, so this
                                 *  is not bit-identical to what the ISIS
                                 *  comparison tests were recorded against --
                                 *  hence the switch rather than a replacement.
                                 *  GAEL_ROT_SEGMENTS=0 restores the old path.  */
                                 rot_flux = use_rot_segments()
                                     ? rotational_broaden(raw.lambda, raw.flux,
                                                          raw.lambda, vsini)
                                     : rotational_broaden(raw.lambda, raw.flux,
                                                          vsini);
                             }
                             else{
                                 rot_flux = raw.flux;
                             }
                             //std::cout << "[LoadSpec](fetch) Degrading resolution." << std::endl;
                             // Then apply spectral degradation.  The rotated flux is
                             // kept even when no degradation is requested - otherwise
                             // vsini would be silently dropped at infinite resolution.
                             Spectrum result = raw;
                             /*  ISIS never broadens the continuum -- it divides the
                              *  broadened calibrated flux by the *unbroadened*
                              *  continuum and only resamples the latter onto the
                              *  convolved grid, which here is the same grid.  Keep
                              *  the untouched calibrated flux as the numerator.   */
                             if (with_continuum) result.cont = raw.flux;
                             result.flux = rot_flux;
                             if (resOffset != 0 || resSlope != 0){
                             result.flux = degrade_resolution(raw.lambda, rot_flux,
                                                               resOffset, resSlope);
                                                               //std::cout << "[LoadSpec](fetch) returning result." << std::endl;
                             }
                             return result;
                         });
        } catch (const MissingGridFile&) {
            register_absent(base_ + "/HHE", cn.coord, path);
            return nullptr;
        }
    };
    //std::cout << "[LoadSpec] Interpolating Cube." << std::endl;

    /* ---------- weighted sum over the cube ------------------------ */
    /*  Planned again whenever a corner turns out to be a hole: each round
     *  registers one more missing file, so this ends.                     */
    Spectrum out; double wsum=0.0;
    for (bool complete = false; !complete; ) {
        std::vector<GridGap> used;
        const std::vector<CubeCorner> nodes =
            bridge_gaps(base_ + "/HHE", cube.axes, corners, corner_path, used);

        /*  A remote provider can pipeline all corner files of this
         *  evaluation in one round trip instead of paying the latency per
         *  corner.  Known holes are left out; their substitutes are in.   */
        if (auto prov = grid_data_provider(); prov && prov->handles(base_)) {
            std::vector<std::string> want;
            want.reserve(nodes.size());
            for (const auto& nd : nodes)
                if (nd.weight != 0.0) want.push_back(corner_path(nd.coord));
            prov->prefetch(want);
        }

        out = Spectrum{}; wsum = 0.0; complete = true;
        bool first=true;
        for (const auto& nd : nodes){
            /* A parameter sitting exactly on a grid node gives its partner
               corner weight 0; fetching it would read (and convolve) a file
               that cannot contribute.  With one pinned axis that is half of
               all corners.                                                 */
            if (nd.weight == 0.0) continue;
            //std::cout << "[LoadSpec] Fetching Spec." << std::endl;
            const SpectrumPtr ptr = fetch(nd);
            if (!ptr) { complete = false; break; }
            const Spectrum& sp = *ptr;
            //std::cout << "[LoadSpec] Fetched." << std::endl;
            if(first){
                out = sp; out.flux.setZero();
                if (with_continuum) { out.cont.setZero(); out.cont_den.setZero(); }
                first=false;
            }
            out.flux += sp.flux * nd.weight;
            if (with_continuum) {
                out.cont     += sp.cont     * nd.weight;
                out.cont_den += sp.cont_den * nd.weight;
            }
            wsum     += nd.weight;
        }
        if (complete) record_gaps(used);
    }
    //std::cout << "[LoadSpec] Done!" << std::endl;
    if(wsum>0.0) out.flux.array() /= wsum;

    /*  The continuum is interp(c)/interp(f), not interp(c/f): ISIS forms the
     *  ratio *after* interpolating both columns across the hypercube
     *  (spectroscopic_fitting.sl, `c[k] = f[k]/interpol_syn(...)`).  The two
     *  differ because the weights that mix the corners' continua are then the
     *  corners' *line* depths, so the result carries a faint line imprint --
     *  small, but it is what the reference does.                            */
    if (with_continuum) {
        if (wsum > 0.0) { out.cont.array() /= wsum; out.cont_den.array() /= wsum; }
        for (Eigen::Index i = 0; i < out.cont.size(); ++i) {
            const double d = out.cont_den[i];
            out.cont[i] = (d > 0.0) ? out.cont[i] / d : 0.0;
        }
        out.cont_den.resize(0);          // scratch, never leaves this function
    }
    return out;
}

/* =================================================================== *
 *  Per-species wavelength grids
 * =================================================================== */
static Vector read_lambda_column(const std::string& path)
{
    CCfits::FITS f(localize_grid_path(path), CCfits::Read);
    CCfits::ExtHDU& e = f.extension(1);
    std::vector<Real> tmp;
    e.column("l").read(tmp, 1, e.rows());
    return Eigen::Map<Vector>(tmp.data(), tmp.size());
}

/*  Slice to the active window with one point of margin, matching what
 *  read_fits() does to the flux arrays so the two stay index-aligned.     */
Vector ModelGrid::slice_to_window(const Vector& lam) const
{
    if (lam.size() == 0) return lam;
    if (window_lo_ == -std::numeric_limits<double>::infinity()) return lam;

    const Real* b = lam.data();
    const Real* e = lam.data() + lam.size();
    Eigen::Index i0 = std::lower_bound(b, e, window_lo_) - b;
    Eigen::Index i1 = std::upper_bound(b, e, window_hi_) - b;
    if (i0 > 0)          --i0;
    if (i1 < lam.size()) ++i1;
    if (i1 <= i0) { i0 = 0; i1 = lam.size(); }
    return lam.segment(i0, i1 - i0);
}

const Vector& ModelGrid::hhe_lambda() const
{
    std::lock_guard<std::mutex> lk(g_lambda_mtx);
    if (hhe_lambda_.size() == 0)
        hhe_lambda_ = slice_to_window(
            read_lambda_column(base_ + "/HHE/lambda.fits"));
    return hhe_lambda_;
}

const Vector& ModelGrid::species_lambda(int s) const
{
    std::lock_guard<std::mutex> lk(g_lambda_mtx);
    if (species_lambda_.size() != species_.size())
        species_lambda_.resize(species_.size());
    if (s < 0 || s >= static_cast<int>(species_.size()))
        throw std::runtime_error("species index out of range");
    Vector& v = species_lambda_[static_cast<std::size_t>(s)];
    if (v.size() == 0)
        v = slice_to_window(read_lambda_column(
                base_ + "/" + species_[static_cast<std::size_t>(s)] +
                "/lambda.fits"));
    return v;
}

/*  Guards union_lambda_; separate from g_lambda_mtx because building a union
 *  calls species_lambda(), which takes that one.                            */
static std::mutex g_union_mtx;

const Vector& ModelGrid::union_lambda(const std::vector<int>& species) const
{
    {
        std::lock_guard<std::mutex> lk(g_union_mtx);
        for (const auto& [key, lam] : union_lambda_)
            if (key == species) return lam;
    }

    /*  Built outside the lock: species_lambda() may read 25 FITS files the
     *  first time round.                                                    */
    std::vector<double> u(hhe_lambda().data(),
                          hhe_lambda().data() + hhe_lambda().size());
    for (int s : species) {
        const Vector& l = species_lambda(s);
        const std::size_t n0 = u.size();
        u.insert(u.end(), l.data(), l.data() + l.size());
        std::inplace_merge(u.begin(), u.begin() + n0, u.end());
    }
    u.erase(std::unique(u.begin(), u.end()), u.end());

    std::lock_guard<std::mutex> lk(g_union_mtx);
    for (const auto& [key, lam] : union_lambda_)
        if (key == species) return lam;           // lost a race; keep the first
    union_lambda_.emplace_back(
        species, Eigen::Map<Vector>(u.data(), static_cast<Eigen::Index>(u.size())));
    return union_lambda_.back().second;
}

/* =================================================================== *
 *  One element's f_S/f_HHE line-correction spectrum
 * =================================================================== */
Vector ModelGrid::load_element_factor(int    s,
                                      double teff, double logg, double z,
                                      double he,   double xi,
                                      double abundance) const
{
    if (s < 0 || s >= static_cast<int>(species_.size()))
        throw std::runtime_error("species index out of range");
    const std::string& name = species_[static_cast<std::size_t>(s)];

    const std::vector<const GridAxis*> axes = element_cube_axes(s);
    const std::vector<double> vals = { z, he, xi, logg, teff, abundance };

    const auto corners = build_corners(axes, vals);
    const std::string root = base_ + '/' + name;
    const PathOf element_path = [&](const Coord& c) {
        return element_corner_path(base_, name, c);
    };

    /*  Holes are bridged exactly as in load_spectrum, over all six axes.  */
    Vector out; double wsum = 0.0;
    for (bool complete = false; !complete; ) {
        std::vector<GridGap> used;
        const auto nodes = bridge_gaps(root, axes, corners, element_path, used);

        /*  Same batching hint as load_spectrum: a remote provider fetches
         *  all corners of this element cube in one round trip.            */
        if (auto prov = grid_data_provider(); prov && prov->handles(base_)) {
            std::vector<std::string> want;
            want.reserve(nodes.size());
            for (const auto& nd : nodes)
                if (nd.weight != 0.0) want.push_back(element_path(nd.coord));
            prov->prefetch(want);
        }

        out = Vector(); wsum = 0.0; complete = true;
        bool first = true;
        for (const auto& nd : nodes) {
            if (nd.weight == 0.0) continue;      // pinned axis: half the corners

            const std::string epath = element_path(nd.coord);

            std::size_t key = std::hash<std::string>{}(epath);
            key = hash_combine(key, 0xE1E3E27ULL);
            key = hash_combine(key, window_key_);

            SpectrumPtr sp;
            try {
                sp = SpectrumCache::instance().insert_if_absent(key, [&] {
                    return read_element_fits(epath, s);
                });
            } catch (const MissingGridFile&) {
                register_absent(root, nd.coord, epath);
                complete = false;
                break;
            }

            if (first) { out = Vector::Zero(sp->flux.size()); first = false; }
            out += sp->flux * nd.weight;
            wsum     += nd.weight;
        }
        if (complete) record_gaps(used);
    }
    if (wsum > 0.0) out.array() /= wsum;
    return out;
}

/*  An element file holds a single column "f": the ratio flux, already on the
 *  species' own wavelength grid.
 *
 *  Only `flux` is filled.  The wavelength grid is `species_lambda(s)` for
 *  every corner of every element, so storing a copy of it -- plus a vector of
 *  ones for `sigma` -- in each cache entry tripled what a corner costs for
 *  nothing: 25 species is ~1 M points, and at eight live corners each that is
 *  600 MB of duplicate lambda and unit sigma against a 128 MiB budget.      */
Spectrum ModelGrid::read_element_fits(const std::string& path, int s) const
{
    const Vector& lam = species_lambda(s);

    std::vector<Real> fl;
    read_corner_file(path, [&](CCfits::ExtHDU& ext) {
        ext.column("f").read(fl, 1, ext.rows());
    });

    /*  The window slice applied to lambda has to be applied identically
     *  here; species_lambda() sliced with the same bounds, so recover the
     *  offset by locating the sliced grid inside the full one.            */
    Eigen::Index i0 = 0;
    if (static_cast<Eigen::Index>(fl.size()) != lam.size()) {
        const Vector full = read_lambda_column(
            base_ + "/" + species_[static_cast<std::size_t>(s)] + "/lambda.fits");
        const Real* b = full.data();
        i0 = std::lower_bound(b, b + full.size(), lam[0]) - b;
        if (i0 + lam.size() > static_cast<Eigen::Index>(fl.size()))
            throw std::runtime_error("element file '" + path +
                                     "' does not match its lambda grid");
    }

    Spectrum sp;
    sp.flux = Eigen::Map<Vector>(fl.data(), fl.size()).segment(i0, lam.size());
    return sp;
}

/* ------------------------------------------------------------------ */
void ModelGrid::set_wavelength_window(double lambda_min, double lambda_max)
{
    if (!(lambda_max > lambda_min)) return;      // nonsense: keep everything

    window_lo_  = lambda_min;
    window_hi_  = lambda_max;
    window_key_ = hash_combine(hash_double(lambda_min), hash_double(lambda_max));
}

/* ------------------------------------------------------------------ *
 *                read_fits()                                          *
 * ------------------------------------------------------------------ */
Spectrum ModelGrid::read_fits(const std::string& path, bool with_continuum) const
{
    using Vec = std::vector<Real>;
    /*  Corner spectra are read from several OpenMP threads at once
        (SpectrumCache builds outside its lock on purpose), so the shared
        wavelength cache needs its own guard.                            */
    static std::unordered_map<std::string,Vector> wave_cache;
    static std::mutex                             wave_mtx;

    Vector lam;
    {
        std::lock_guard<std::mutex> lk(wave_mtx);
        if (auto it = wave_cache.find(base_); it!=wave_cache.end()) lam=it->second;
    }
    if (lam.size()==0) {
        fs::path lp = fs::path(base_) / "HHE" / "lambda.fits";
        if (grid_path_exists(lp.string())){
            CCfits::FITS fl(localize_grid_path(lp.string()), CCfits::Read);
            CCfits::ExtHDU& e = fl.extension(1);
            Vec tmp; e.column("l").read(tmp,1,e.rows());
            lam = Eigen::Map<Vector>(tmp.data(),tmp.size());
            std::lock_guard<std::mutex> lk(wave_mtx);
            wave_cache.emplace(base_,lam);
        }
    }

    /*  Column "f" is the normalised flux, column "c" the flux-calibrated one
     *  (SURFACE, erg/s/cm^2/A) -- ISIS's `nonorm` qualifier picks "c".  A
     *  binary fit is built from "c" and keeps "f" as the divisor that turns
     *  it into a continuum; a single-component fit only ever needs "f".     */
    Vec fl, fn, lt;
    read_corner_file(path, [&](CCfits::ExtHDU& ext) {
        ext.column(with_continuum ? "c" : "f").read(fl,1,ext.rows());
        if (with_continuum) ext.column("f").read(fn,1,ext.rows());
        if (lam.size()==0) ext.column("l").read(lt,1,ext.rows());
    });
    if(lam.size()==0){
        lam = Eigen::Map<Vector>(lt.data(),lt.size());
        std::lock_guard<std::mutex> lk(wave_mtx);
        wave_cache.emplace(base_,lam);
    }

    Spectrum sp;

    /* ---- keep only the window the fit can see ---------------------- *
     *  One extra grid point on each side so that the rebin onto the
     *  observed bins, and the convolution kernels, still have neighbours
     *  at the window edges.                                            */
    Eigen::Index i0 = 0, i1 = lam.size();          // [i0, i1)
    if (window_lo_ > -std::numeric_limits<double>::infinity() &&
        lam.size() > 0)
    {
        const Real* b = lam.data();
        const Real* e = lam.data() + lam.size();
        i0 = std::lower_bound(b, e, window_lo_) - b;
        i1 = std::upper_bound(b, e, window_hi_) - b;
        if (i0 > 0)          --i0;
        if (i1 < lam.size()) ++i1;
        if (i1 <= i0) { i0 = 0; i1 = lam.size(); }  // window misses the grid
    }
    const Eigen::Index n = i1 - i0;

    sp.lambda = lam.segment(i0, n);
    sp.flux   = Eigen::Map<Vector>(fl.data(), fl.size()).segment(i0, n);
    sp.sigma  = Vector::Ones(n);
    if (with_continuum)
        sp.cont_den = Eigen::Map<Vector>(fn.data(), fn.size()).segment(i0, n);
    return sp;
}

ModelGrid::ParameterBounds ModelGrid::get_parameter_bounds() const {
    ParameterBounds bounds;
    
    for (const auto& axis : axes_) {
        if (axis.values.size() == 0) continue;
        
        double min_val = axis.values.minCoeff();
        double max_val = axis.values.maxCoeff();
        
        if (axis.name == "t") {
            bounds.teff_min = min_val;
            bounds.teff_max = max_val;
        }
        else if (axis.name == "g") {
            bounds.logg_min = min_val;
            bounds.logg_max = max_val;
        }
        else if (axis.name == "z") {
            bounds.z_min = min_val;
            bounds.z_max = max_val;
        }
        else if (axis.name == "HHE") {
            bounds.he_min = min_val;
            bounds.he_max = max_val;
        }
        else if (axis.name == "x") {
            bounds.xi_min = min_val;
            bounds.xi_max = max_val;
        }
    }
    
    return bounds;
}

ModelGrid::ParameterBounds
ModelGrid::intersect(const std::vector<ParameterBounds>& all)
{
    ParameterBounds out;
    bool first = true;
    for (const auto& b : all) {
        if (first) { out = b; first = false; continue; }
        out.teff_min = std::max(out.teff_min, b.teff_min);
        out.teff_max = std::min(out.teff_max, b.teff_max);
        out.logg_min = std::max(out.logg_min, b.logg_min);
        out.logg_max = std::min(out.logg_max, b.logg_max);
        out.z_min    = std::max(out.z_min,    b.z_min);
        out.z_max    = std::min(out.z_max,    b.z_max);
        out.he_min   = std::max(out.he_min,   b.he_min);
        out.he_max   = std::min(out.he_max,   b.he_max);
        out.xi_min   = std::max(out.xi_min,   b.xi_min);
        out.xi_max   = std::min(out.xi_max,   b.xi_max);
    }
    return out;
}

} // namespace specfit