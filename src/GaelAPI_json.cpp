#include "specfit/GaelAPI.hpp"
#include "specfit/JsonUtils.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <stdexcept>

using nlohmann::json;

namespace specfit::api {

GlobalSettings global_settings_from_json_file(const std::string& path)
{
    json j = load_json(path);
    expand_env(j);

    GlobalSettings gs;
    if (j.contains("basePaths"))
        gs.base_paths = j["basePaths"].get<std::vector<std::string>>();

    const auto& s = j.value("settings", json::object());

    auto get_d = [&](const char* k, double& dst){
        if (s.contains(k)) dst = s[k].get<double>(); };
    auto get_i = [&](const char* k, int& dst){
        if (s.contains(k)) dst = s[k].get<int>(); };
    auto get_b = [&](const char* k, bool& dst){
        if (s.contains(k)) dst = s[k].get<bool>(); };

    get_d("xrange",         gs.xrange);
    get_d("filter_snr",     gs.filter_snr);
    get_d("requireBlue",    gs.require_blue);
    get_b("autoFreezeVsini",gs.auto_freeze_vsini);

    if (s.contains("untieParams"))
        gs.untie_params = s["untieParams"].get<std::vector<std::string>>();

    get_i("nitNoiseMax",    gs.nit_noise_max);
    get_i("nitFitMax",      gs.nit_fit_max);
    get_i("widthBoxPx",     gs.width_box_px);
    get_d("outlierSigmaLo", gs.outlier_sigma_lo);
    get_d("outlierSigmaHi", gs.outlier_sigma_hi);
    get_d("convRangeLo",    gs.conv_range_lo);
    get_d("convRangeHi",    gs.conv_range_hi);
    get_d("convFraction",   gs.conv_fraction);
    get_i("contJitterK",    gs.cont_jitter_K);

    return gs;
}

FitInput fit_input_from_json_file(const std::string& path)
{
    json j = load_json(path);
    expand_env(j);

    FitInput fi;

    const auto grids = j.at("grids").get<std::vector<std::string>>();
    const auto ig    = j.at("initialGuess");

    fi.components.resize(grids.size());
    for (std::size_t c = 0; c < grids.size(); ++c) {
        auto& comp = fi.components[c];
        comp.grid_relative_path = grids[c];

        const std::string pre = "c" + std::to_string(c + 1) + "_";
        auto pv = [&](const char* k){
            return ig.at(pre + k).at("value").get<double>(); };
        auto pf = [&](const char* k){
            return ig.at(pre + k).at("freeze").get<bool>(); };

        comp.vrad  = pv("vrad");   comp.freeze_vrad  = pf("vrad");
        comp.vsini = pv("vsini");  comp.freeze_vsini = pf("vsini");
        comp.zeta  = pv("zeta");   comp.freeze_zeta  = pf("zeta");
        comp.teff  = pv("teff");   comp.freeze_teff  = pf("teff");
        comp.logg  = pv("logg");   comp.freeze_logg  = pf("logg");
        comp.xi    = pv("xi");     comp.freeze_xi    = pf("xi");
        comp.z     = pv("z");      comp.freeze_z     = pf("z");
        comp.he    = pv("HE");     comp.freeze_he    = pf("HE");
    }

    if (j.contains("outputPath"))
        fi.output_path = j["outputPath"].get<std::string>();

    for (const auto& obs : j.at("observations")) {
        ObservationInput oi;

        if (obs.contains("waveCut"))
            oi.waveCut = obs["waveCut"].get<std::array<double,2>>();
        if (obs.contains("ignore"))
            oi.ignore  = obs["ignore"].get<std::vector<std::array<double,2>>>();
        if (obs.contains("csplineAnchorpoints"))
            oi.cspline_anchorpoints =
                obs["csplineAnchorpoints"].get<std::vector<std::array<double,3>>>();

        for (const auto& fj : obs.at("files")) {
            SpectrumFileInput f;
            f.filename  = fj.at("filename").get<std::string>();
            f.spectype  = fj.at("spectype").get<std::string>();
            f.resOffset = fj.value("resOffset", 0.0);
            f.resSlope  = fj.value("resSlope",  0.0);
            f.barycorr  = fj.value("barycorr",  0.0);

            if (fj.contains("waveCut"))
                f.waveCut = fj["waveCut"].get<std::array<double,2>>();
            if (fj.contains("ignore"))
                f.ignore  = fj["ignore"].get<std::vector<std::array<double,2>>>();
            if (fj.contains("csplineAnchorpoints"))
                f.cspline_anchorpoints =
                    fj["csplineAnchorpoints"].get<std::vector<std::array<double,3>>>();

            oi.files.push_back(std::move(f));
        }
        fi.observations.push_back(std::move(oi));
    }

    return fi;
}

} // namespace specfit::api