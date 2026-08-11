#include "specfit/GaelAPI.hpp"
#include "specfit/JsonUtils.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <set>
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

    get_b("autoFreezeSurRatio", gs.auto_freeze_sur_ratio);
    get_d("surRatioThres",      gs.sur_ratio_thres);
    get_d("c2DetectionThres",   gs.c2_detection_thres);

    get_b("addTelluricModel",   gs.add_telluric_model);
    get_b("telluricIsisPwvScale", gs.telluric_isis_pwv_scale);

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

        /*  cN_sur_ratio is optional: single-component configs never had one,
         *  and for a binary the ISIS defaults (1, free) are the sensible
         *  start.  Component 1's is pinned to 1 whatever the file says.     */
        comp.sur_ratio        = 1.0;
        comp.freeze_sur_ratio = (c == 0);
        if (c > 0 && ig.contains(pre + "sur_ratio")) {
            const auto& sr = ig.at(pre + "sur_ratio");
            if (sr.contains("value"))  comp.sur_ratio        = sr["value"].get<double>();
            if (sr.contains("freeze")) comp.freeze_sur_ratio = sr["freeze"].get<bool>();
        }

        /* ---- element abundances: any other cN_<KEY> entry ------------- *
         *  The grid decides which names are meaningful, so anything that is
         *  not one of the fixed parameters above is taken to be an element
         *  and validated later against the grid's species list.  ISIS names
         *  them in upper case (cN_FE, cN_SI); the check is case-sensitive
         *  because the grid's directory names are.                        */
        static const std::set<std::string> fixed = {
            "vrad","vsini","zeta","teff","logg","xi","z","HE","sur_ratio" };

        for (const auto& [key, val] : ig.items()) {
            if (key.rfind(pre, 0) != 0) continue;        // other component
            const std::string name = key.substr(pre.size());
            if (fixed.count(name)) continue;
            if (!val.is_object() || !val.contains("value")) continue;

            comp.abundances[name] = val.at("value").get<double>();
            comp.freeze_abundances[name] = val.value("freeze", false);
        }
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

            /*  Telluric seeds; ISIS's defaults (airmass 1, pwv 1 mm).  An
             *  airmass of 0, or "fitTelluric": false, switches the component
             *  off for this spectrum.                                       */
            f.airmass      = fj.value("airmass", 1.0);
            f.pwv          = fj.value("pwv",     1.0);
            f.fit_telluric = fj.value("fitTelluric", true);

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