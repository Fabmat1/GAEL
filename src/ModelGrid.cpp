#include "specfit/ModelGrid.hpp"
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

namespace fs = std::filesystem;

/* ---------- little helpers outside namespace ----------------------- */
static std::string fmt(double v, int prec)
{
    std::ostringstream s; s << std::fixed << std::setprecision(prec) << v;
    return s.str();
}
static double to_linear(const std::string& ax, double v) { return v; }

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
        if (fs::exists(p / "grid.fits")) return p.string();
    }
    throw std::runtime_error("Grid '" + rel_path + "' not found.");
}

/* =======================  ctor helpers  ============================ */
ModelGrid::ModelGrid(const std::vector<std::string>& bases,
                     const std::string& rel_path)
: base_(resolve_grid(bases, rel_path))
{
    CCfits::FITS f(base_ + "/grid.fits", CCfits::Read);
    CCfits::ExtHDU& ext = f.extension(1);
    for (const auto& [name, col] : ext.column()) {
        std::vector<Real> buf;
        /* --- keep the robust logic from the original implementation --- */
        if (col->varLength())               col->read(buf, 1);      // variable-length vector
        else if (col->repeat() == 1)        col->read(buf, 1, 1);   // scalar column
        else                                col->read(buf, 1);      // fixed-length vector
        axes_.push_back({name, to_eigen(buf)});
    }
}
ModelGrid::ModelGrid(std::string abs) : base_(std::move(abs)) {}

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

/* =================================================================== */
Spectrum ModelGrid::load_spectrum(double teff,double logg,double z,
                                  double he,double xi,double vsini,
                                  double resOffset,double resSlope) const
{   
    //std::cout << "[LoadSpec] Building Interpolation Hypercube." << std::endl;
    /* ---------- build interpolation hyper-cube -------------------- */
    struct Node { double t,g,z,h,x,w; };
    std::array<Node,32> nodes{}; nodes[0] = {0,0,0,0,0,1}; int nN=1;

    for (const auto& ax: axes_) {
        const double p = param_for_axis(ax.name,teff,logg,z,he,xi);
        const Vector& grid = ax.values;

        if (grid.size()==1) {                 // constant axis
            for (int i=0;i<nN;++i) {
                if      (ax.name=="t") nodes[i].t = grid[0];
                else if (ax.name=="g") nodes[i].g = grid[0];
                else if (ax.name=="z") nodes[i].z = grid[0];
                else if (ax.name=="HHE")nodes[i].h = grid[0];
                else if (ax.name=="x") nodes[i].x = grid[0];
            }
            continue;
        }

        auto it = std::lower_bound(grid.data(),grid.data()+grid.size(),p);
        int hi = (it==grid.data()+grid.size())?grid.size()-1:int(it-grid.data());
        int lo = (hi==0)?0:hi-1;

        double a_hi = (grid[hi]==grid[lo])?0.0
                      :(to_linear(ax.name,p)-to_linear(ax.name,grid[lo]))
                       /(to_linear(ax.name,grid[hi])-to_linear(ax.name,grid[lo]));
        double a_lo = 1.0-a_hi;

        for (int i=nN-1;i>=0;--i) nodes[i+nN]=nodes[i];
        for (int i=0;i<nN;++i) {
            Node& A = nodes[i]; Node& B = nodes[i+nN];
            if      (ax.name=="t"){A.t=grid[lo];B.t=grid[hi];}
            else if (ax.name=="g"){A.g=grid[lo];B.g=grid[hi];}
            else if (ax.name=="z"){A.z=grid[lo];B.z=grid[hi];}
            else if (ax.name=="HHE"){A.h=grid[lo];B.h=grid[hi];}
            else if (ax.name=="x"){A.x=grid[lo];B.x=grid[hi];}
            A.w*=a_lo; B.w*=a_hi;
        }
        nN*=2;
    }
    //std::cout << "[LoadSpec] Building Tiny Local Cache." << std::endl;

    /* ---------- tiny per-call local cache (≤32) -------------------- */
    using Key = std::uint64_t;
    auto pack = [](int t,int g,int z,int h,int x)->Key{
        return Key(t) | (Key(g)<<12) | (Key(z)<<24)
                     | (Key(h)<<36) | (Key(x)<<48);
    };
    struct Entry { SpectrumPtr sp; bool ok=false; };

    struct LocalCache {
        std::array<Key,32> keys{};
        std::array<Entry,32> ent{};
        std::size_t n=0;
        Entry& operator[](Key k){
            for(std::size_t i=0;i<n;++i) if(keys[i]==k) return ent[i];
            keys[n]=k; return ent[n++];                // new slot
        }
    } lc;
    //std::cout << "[LoadSpec] Corner fetching." << std::endl;

    /* ---------- helper to fetch / cache one corner ---------------- */
    auto fetch = [&](const Node& nd)->const Spectrum& {
        //std::cout << "[LoadSpec](fetch) Fetching..." << std::endl;

        int Ti=int(std::lround(nd.t));
        int Gi=int(std::lround(nd.g*100));
        int Zi=int(std::lround(nd.z*100));
        int Hi=int(std::lround(nd.h*1000));
        int Xi=int(std::lround(nd.x*100));

        Key k = pack(Ti,Gi,Zi,Hi,Xi);
        k = hash_combine(k, hash_double(vsini));      // Include vsini in cache key
        k = hash_combine(k, hash_double(resOffset));
        k = hash_combine(k, hash_double(resSlope));

        Entry& e = lc[k];
        //std::cout << "[LoadSpec](fetch) Entry hashes done..." << std::endl;
        if (!e.ok) {
            if (auto sp = SpectrumCache::instance().try_get(k))
                e.sp = std::move(sp);                           // global hit
            else {
                //std::cout << "[LoadSpec](fetch) Cache miss, calculating." << std::endl;
                std::ostringstream oss;
                oss<<base_<<"/HHE"
                   <<"/Z"<<fmt(nd.z,2)
                   <<"/HE"<<fmt(nd.h,3)
                   <<"/X"<<fmt(nd.x,2)
                   <<"/G"<<fmt(nd.g,3)
                   <<"/T"<<Ti<<".fits";
                const std::string path = oss.str();

                e.sp = SpectrumCache::instance()
                         .insert_if_absent(k,[&]{
                             //std::cout << "[LoadSpec](fetch) Reading .fits ." << std::endl;
                             Spectrum raw = read_fits(path);
                             //std::cout << "[LoadSpec](fetch) Broadening." << vsini << std::endl;

                             // Apply rotational broadening FIRST
                             Vector rot_flux;
                             if (vsini >= 0.1){
                                //std::cout << "[LoadSpec](fetch) Broadeningggg." << vsini << std::endl;
                                 rot_flux = rotational_broaden(raw.lambda,
                                                                    raw.flux,
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
                             result.flux = rot_flux;
                             if (resOffset != 0 || resSlope != 0){
                             result.flux = degrade_resolution(raw.lambda, rot_flux,
                                                               resOffset, resSlope);
                                                               //std::cout << "[LoadSpec](fetch) returning result." << std::endl;
                             }
                             return result;
                         });
            }
            e.ok = true;
        }
        return *e.sp;
    };
    //std::cout << "[LoadSpec] Interpolating Cube." << std::endl;

    /* ---------- weighted sum over the cube ------------------------ */
    Spectrum out; bool first=true; double wsum=0.0;

    for(int i=0;i<nN;++i){
        /* A parameter sitting exactly on a grid node gives its partner corner
           weight 0; fetching it would read (and convolve) a file that cannot
           contribute.  With one pinned axis that is half of all corners.      */
        if (nodes[i].w == 0.0) continue;
        //std::cout << "[LoadSpec] Fetching Spec." << std::endl;
        const Spectrum& sp = fetch(nodes[i]);
        //std::cout << "[LoadSpec] Fetched." << std::endl;
        if(first){ out = sp; out.flux.setZero(); first=false; }
        out.flux += sp.flux * nodes[i].w;
        wsum     += nodes[i].w;
    }
    //std::cout << "[LoadSpec] Done!" << std::endl;
    if(wsum>0.0) out.flux.array() /= wsum;
    return out;
}

/* ------------------------------------------------------------------ *
 *                read_fits()   (unchanged)                           *
 * ------------------------------------------------------------------ */
Spectrum ModelGrid::read_fits(const std::string& path) const
{
    using Vec = std::vector<Real>;
    static std::unordered_map<std::string,Vector> wave_cache;

    Vector lam;
    if (auto it = wave_cache.find(base_); it!=wave_cache.end()) lam=it->second;
    else {
        fs::path lp = fs::path(base_) / "HHE" / "lambda.fits";
        if (fs::exists(lp)){
            CCfits::FITS fl(lp.string(), CCfits::Read);
            CCfits::ExtHDU& e = fl.extension(1);
            Vec tmp; e.column("l").read(tmp,1,e.rows());
            lam = Eigen::Map<Vector>(tmp.data(),tmp.size());
            wave_cache.emplace(base_,lam);
        }
    }

    CCfits::FITS f(path, CCfits::Read);
    CCfits::ExtHDU& ext = f.extension(1);

    Vec fl; ext.column("f").read(fl,1,ext.rows());
    if(lam.size()==0){
        Vec lt; ext.column("l").read(lt,1,ext.rows());
        lam = Eigen::Map<Vector>(lt.data(),lt.size());
        wave_cache.emplace(base_,lam);
    }

    Spectrum sp;
    sp.lambda = lam;
    sp.flux   = Eigen::Map<Vector>(fl.data(),fl.size());
    sp.sigma  = Vector::Ones(lam.size());
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

} // namespace specfit