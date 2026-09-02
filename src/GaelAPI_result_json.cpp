/*  Lossless FitResult <-> JSON, for running fits out of process: a worker
 *  writes the file, the host application reads it back and continues exactly
 *  as if GaelSession::run() had returned in process.
 *
 *  The schema is versioned ("gael_result_version"); the reader refuses
 *  versions it does not know rather than guessing.
 */
#include "specfit/GaelAPI.hpp"
#include "specfit/JsonUtils.hpp"

#include <nlohmann/json.hpp>

#include <fstream>
#include <stdexcept>

using nlohmann::json;

namespace specfit::api {

namespace {

constexpr int kResultVersion = 1;

json vec_to_json(const Vector& v)
{
    json a = json::array();
    for (Eigen::Index i = 0; i < v.size(); ++i) a.push_back(v[i]);
    return a;
}

Vector vec_from_json(const json& a)
{
    Vector v(static_cast<Eigen::Index>(a.size()));
    Eigen::Index i = 0;
    for (const auto& x : a) v[i++] = x.get<double>();
    return v;
}

const char* status_name(Status s)
{
    switch (s) {
        case Status::Ok:                  return "Ok";
        case Status::InvalidInput:        return "InvalidInput";
        case Status::PreprocessingFailed: return "PreprocessingFailed";
        case Status::FitFailed:           return "FitFailed";
        case Status::InternalError:       return "InternalError";
        case Status::Aborted:             return "Aborted";
    }
    return "InternalError";
}

Status status_from_name(const std::string& n)
{
    if (n == "Ok")                  return Status::Ok;
    if (n == "InvalidInput")        return Status::InvalidInput;
    if (n == "PreprocessingFailed") return Status::PreprocessingFailed;
    if (n == "FitFailed")           return Status::FitFailed;
    if (n == "InternalError")       return Status::InternalError;
    if (n == "Aborted")             return Status::Aborted;
    throw std::runtime_error("unknown fit status '" + n + "'");
}

json param_to_json(const StellarParamResult& p)
{
    return json{{"value", p.value},
                {"error", p.error},
                {"frozen", p.frozen},
                {"boundary_side", p.boundary_side}};
}

StellarParamResult param_from_json(const json& j)
{
    StellarParamResult p;
    p.value         = j.at("value").get<double>();
    p.error         = j.at("error").get<double>();
    p.frozen        = j.at("frozen").get<bool>();
    p.boundary_side = j.at("boundary_side").get<int>();
    p.at_boundary   = (p.boundary_side != 0);
    return p;
}

json params_to_json(const std::vector<StellarParamResult>& v)
{
    json a = json::array();
    for (const auto& p : v) a.push_back(param_to_json(p));
    return a;
}

std::vector<StellarParamResult> params_from_json(const json& a)
{
    std::vector<StellarParamResult> v;
    v.reserve(a.size());
    for (const auto& j : a) v.push_back(param_from_json(j));
    return v;
}

} // namespace

void fit_result_to_json_file(const FitResult& r, const std::string& path)
{
    json j;
    j["gael_result_version"] = kResultVersion;
    j["status"]              = status_name(r.status);
    j["error_message"]       = r.error_message;
    j["warnings"]            = r.warnings;

    j["converged"]         = r.converged;
    j["iterations"]        = r.iterations;
    j["final_chi2"]        = r.final_chi2;
    j["n_free_parameters"] = r.n_free_parameters;
    j["n_data_points"]     = r.n_data_points;
    j["cont_jitter_K"]     = r.cont_jitter_K;

    json comps = json::array();
    for (const auto& c : r.components) {
        json cj;
        cj["vrad"]      = params_to_json(c.vrad);
        cj["vsini"]     = params_to_json(c.vsini);
        cj["zeta"]      = params_to_json(c.zeta);
        cj["teff"]      = params_to_json(c.teff);
        cj["logg"]      = params_to_json(c.logg);
        cj["xi"]        = params_to_json(c.xi);
        cj["z"]         = params_to_json(c.z);
        cj["he"]        = params_to_json(c.he);
        cj["sur_ratio"] = params_to_json(c.sur_ratio);
        json ab = json::object();
        for (const auto& [name, v] : c.abundances) ab[name] = params_to_json(v);
        cj["abundances"]  = std::move(ab);
        cj["param_order"] = c.param_order;
        comps.push_back(std::move(cj));
    }
    j["components"] = std::move(comps);

    json specs = json::array();
    for (const auto& s : r.spectra) {
        json sj;
        sj["source_filename"] = s.source_filename;
        sj["input_index"]     = s.input_index;
        sj["lambda"]          = vec_to_json(s.lambda);
        sj["flux"]            = vec_to_json(s.flux);
        sj["sigma"]           = vec_to_json(s.sigma);
        sj["ignoreflag"]      = s.ignoreflag;
        sj["model"]           = vec_to_json(s.model);
        sj["continuum"]       = vec_to_json(s.continuum);
        json cm = json::array();
        for (const auto& m : s.component_models) cm.push_back(vec_to_json(m));
        sj["component_models"] = std::move(cm);
        sj["cont_x"]          = vec_to_json(s.cont_x);
        sj["cont_y"]          = vec_to_json(s.cont_y);
        sj["telluric"]        = vec_to_json(s.telluric);
        sj["telluric_params"] = params_to_json(s.telluric_params);
        specs.push_back(std::move(sj));
    }
    j["spectra"] = std::move(specs);

    j["rejected_files"]    = r.rejected_files;
    j["rejected_indices"]  = r.rejected_indices;
    j["raw_params"]        = r.raw_params;
    j["raw_uncertainties"] = r.raw_uncertainties;
    j["raw_free_mask"]     = r.raw_free_mask;

    std::ofstream out(path);
    if (!out)
        throw std::runtime_error("cannot write result JSON '" + path + "'");
    out << j.dump() << '\n';
}

FitResult fit_result_from_json_file(const std::string& path)
{
    /*  Deliberately NOT expand_env'd: result payloads are data, and log or
     *  file-name strings may legitimately contain "${".                    */
    json j = load_json(path);

    const int ver = j.at("gael_result_version").get<int>();
    if (ver != kResultVersion)
        throw std::runtime_error("result JSON '" + path + "' has version " +
                                 std::to_string(ver) + ", expected " +
                                 std::to_string(kResultVersion));

    FitResult r;
    r.status        = status_from_name(j.at("status").get<std::string>());
    r.error_message = j.value("error_message", std::string{});
    if (j.contains("warnings"))
        r.warnings = j["warnings"].get<std::vector<std::string>>();

    r.converged         = j.at("converged").get<bool>();
    r.iterations        = j.at("iterations").get<int>();
    r.final_chi2        = j.at("final_chi2").get<double>();
    r.n_free_parameters = j.at("n_free_parameters").get<int>();
    r.n_data_points     = j.at("n_data_points").get<int>();
    r.cont_jitter_K     = j.at("cont_jitter_K").get<int>();

    for (const auto& cj : j.at("components")) {
        ComponentResult c;
        c.vrad      = params_from_json(cj.at("vrad"));
        c.vsini     = params_from_json(cj.at("vsini"));
        c.zeta      = params_from_json(cj.at("zeta"));
        c.teff      = params_from_json(cj.at("teff"));
        c.logg      = params_from_json(cj.at("logg"));
        c.xi        = params_from_json(cj.at("xi"));
        c.z         = params_from_json(cj.at("z"));
        c.he        = params_from_json(cj.at("he"));
        c.sur_ratio = params_from_json(cj.at("sur_ratio"));
        for (const auto& [name, v] : cj.at("abundances").items())
            c.abundances[name] = params_from_json(v);
        c.param_order = cj.at("param_order").get<std::vector<std::string>>();
        r.components.push_back(std::move(c));
    }

    for (const auto& sj : j.at("spectra")) {
        SpectrumResult s;
        s.source_filename = sj.at("source_filename").get<std::string>();
        s.input_index     = sj.at("input_index").get<int>();
        s.lambda          = vec_from_json(sj.at("lambda"));
        s.flux            = vec_from_json(sj.at("flux"));
        s.sigma           = vec_from_json(sj.at("sigma"));
        s.ignoreflag      = sj.at("ignoreflag").get<std::vector<int>>();
        s.model           = vec_from_json(sj.at("model"));
        s.continuum       = vec_from_json(sj.at("continuum"));
        for (const auto& m : sj.at("component_models"))
            s.component_models.push_back(vec_from_json(m));
        s.cont_x          = vec_from_json(sj.at("cont_x"));
        s.cont_y          = vec_from_json(sj.at("cont_y"));
        s.telluric        = vec_from_json(sj.at("telluric"));
        s.telluric_params = params_from_json(sj.at("telluric_params"));
        r.spectra.push_back(std::move(s));
    }

    if (j.contains("rejected_files"))
        r.rejected_files = j["rejected_files"].get<std::vector<std::string>>();
    if (j.contains("rejected_indices"))
        r.rejected_indices = j["rejected_indices"].get<std::vector<int>>();
    r.raw_params        = j.at("raw_params").get<std::vector<double>>();
    r.raw_uncertainties = j.at("raw_uncertainties").get<std::vector<double>>();
    r.raw_free_mask     = j.at("raw_free_mask").get<std::vector<bool>>();

    return r;
}

} // namespace specfit::api
