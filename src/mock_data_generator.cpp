// src/mock_data_generator.cpp

#include "specfit/ModelGrid.hpp"
#include "specfit/SyntheticModel.hpp"
#include "specfit/JsonUtils.hpp"
#include "specfit/NyquistGrid.hpp"
#include "specfit/Rebin.hpp"
#include "specfit/AkimaSpline.hpp"

#include <cxxopts.hpp>
#include <nlohmann/json.hpp>

#include <random>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <thread>
#include <mutex>
#include <atomic>
#include <sstream>
#include <iostream>

namespace fs = std::filesystem;
using namespace specfit;

inline double median(Vector v)               // pass *by value*  → cheap copy
{
    const Eigen::Index n = v.size();
    if (n == 0)
        throw std::runtime_error("median(): empty vector");

    Eigen::Index k = n / 2;
    std::nth_element(v.data(), v.data() + k, v.data() + n);   // k-th element

    double m = v[k];
    if ((n & 1) == 0) {                                       // even length
        const double max_lo = *std::max_element(v.data(), v.data() + k);
        m = 0.5 * (m + max_lo);
    }
    return m;
}

// Parameter distribution types
enum class DistributionType {
    Fixed,
    Gaussian,
    Uniform
};

// Parameter configuration
struct ParameterConfig {
    DistributionType type = DistributionType::Fixed;
    double value = 0.0;
    double error = 0.0;  // for Gaussian
    double min = 0.0;    // for Uniform
    double max = 0.0;    // for Uniform
    
    double sample(std::mt19937& rng) const {
        switch(type) {
            case DistributionType::Fixed:
                return value;
            case DistributionType::Gaussian: {
                std::normal_distribution<> dist(value, error);
                return dist(rng);
            }
            case DistributionType::Uniform: {
                std::uniform_real_distribution<> dist(min, max);
                return dist(rng);
            }
        }
        return value;
    }
    
    static ParameterConfig from_string(const std::string& str) {
        ParameterConfig config;
        
        // Parse format: "value" for fixed, "mean,sigma" for gaussian, "min:max" for uniform
        if (str.find(',') != std::string::npos) {
            // Gaussian: mean,sigma
            auto pos = str.find(',');
            config.type = DistributionType::Gaussian;
            config.value = std::stod(str.substr(0, pos));
            config.error = std::stod(str.substr(pos + 1));
        } else if (str.find(':') != std::string::npos) {
            // Uniform: min:max
            auto pos = str.find(':');
            config.type = DistributionType::Uniform;
            config.min = std::stod(str.substr(0, pos));
            config.max = std::stod(str.substr(pos + 1));
        } else {
            // Fixed: single value
            config.type = DistributionType::Fixed;
            config.value = std::stod(str);
        }
        
        return config;
    }
    
    std::string to_string() const {
        switch(type) {
            case DistributionType::Fixed:
                return "Fixed(" + std::to_string(value) + ")";
            case DistributionType::Gaussian:
                return "Gaussian(" + std::to_string(value) + "±" + std::to_string(error) + ")";
            case DistributionType::Uniform:
                return "Uniform[" + std::to_string(min) + "," + std::to_string(max) + "]";
        }
        return "";
    }
};

// Continuum types
enum class ContinuumType {
    None,
    RandomLinear,
    RandomQuadratic,
    RandomPolynomial
};

// Main configuration structure
struct MockDataConfig {
    // Spectrum configuration
    int num_parameter_sets = 10;
    double wave_start = 3800.0;
    double wave_end = 6800.0;
    double wave_sampling = 0.1;
    double res_offset = 0.0;
    double res_slope = 0.3;
    
    // Noise configuration
    double noise_min = 0.01;  // 1% minimum
    double noise_max = 0.05;  // 5% maximum
    
    // Continuum configuration
    ContinuumType continuum_type = ContinuumType::None;
    
    // Atmospheric parameters
    ParameterConfig teff{DistributionType::Fixed, 25000.0};
    ParameterConfig logg{DistributionType::Fixed, 5.5};
    ParameterConfig z{DistributionType::Fixed, 0.0};
    ParameterConfig he{DistributionType::Fixed, -2.0};
    ParameterConfig xi{DistributionType::Fixed, 0.0};
    
    // Kinematic parameters
    ParameterConfig vrad{DistributionType::Gaussian, 0.0, 10.0};
    ParameterConfig vsini{DistributionType::Fixed, 10.0};
    ParameterConfig zeta{DistributionType::Fixed, 0.0};
    ParameterConfig vrad_scatter{DistributionType::Uniform, 0.0, 0.0, -5.0, 5.0};

    // Multiplicity configuration
    int multiplicity_min = 1;
    int multiplicity_max = 5;
    
    // Grid selection
    std::string grid_path = "sdB/processed/";
    std::string output_dir = "./mock_data/";

    // Additional options
    bool verbose = false;
    int num_threads = 1;

    // RNG seed.  0 keeps the historical behaviour (std::random_device, i.e. a
    // different data set on every invocation).  Any non-zero value makes the
    // whole generation reproducible *and* independent of num_threads, which is
    // what the GAEL-vs-ISIS regression tests rely on for caching.
    unsigned seed = 0;

    // Initial guess configuration for fitting
    struct InitialGuess {
        double value = 0.0;
        bool freeze = false;
    };
    
    std::map<std::string, InitialGuess> initial_guesses = {
        {"c1_vrad",  {0.0, false}},
        {"c1_vsini", {10.0, false}},
        {"c1_zeta",  {0.0, true}},
        {"c1_teff",  {25000.0, false}},
        {"c1_logg",  {5.5, false}},
        {"c1_xi",    {0.0, true}},
        {"c1_z",     {0.0, true}},
        {"c1_HE",    {-2.0, false}}
    };
    
    bool use_nyquist_sampling = true; 
};

// Progress tracking
struct GenerationProgress {
    std::atomic<int> current{0};
    std::atomic<int> total{0};
    std::atomic<bool> should_stop{false};
    std::mutex status_mutex;
    std::string status_message;
    bool verbose = false;
    
    void update_status(const std::string& msg) {
        std::lock_guard<std::mutex> lock(status_mutex);
        status_message = msg;
        if (verbose) {
            std::cout << "\r" << msg << std::flush;
        }
    }
    
    std::string get_status() {
        std::lock_guard<std::mutex> lock(status_mutex);
        return status_message;
    }
};

// Generate random continuum
Vector generate_continuum(const Vector& lambda, ContinuumType type, std::mt19937& rng) {
    const int n = lambda.size();
    Vector continuum = Vector::Ones(n);

    if (type == ContinuumType::None) {
        // Return flat continuum at mid-range
        return continuum * 5000.0;  // Middle of 100-10000 range
    }

    // Normalize wavelength to [0,1] for numerical stability
    const double lmin = lambda.minCoeff();
    const double lmax = lambda.maxCoeff();
    Vector x = (lambda.array() - lmin) / (lmax - lmin);

    // Define the count range
    const double MIN_COUNTS = 100.0;
    const double MAX_COUNTS = 10000.0;
    
    // Random baseline level (where the continuum is centered)
    std::uniform_real_distribution<> baseline_dist(2000.0, 8000.0);
    double baseline = baseline_dist(rng);
    
    // Random variation amplitude (how much the continuum varies)
    std::uniform_real_distribution<> amplitude_dist(0.1, 0.5);
    double amplitude = amplitude_dist(rng);

    switch(type) {
        case ContinuumType::RandomLinear: {
            // Generate linear trend
            std::uniform_real_distribution<> slope_dist(-1.0, 1.0);
            double slope = slope_dist(rng);
            
            // Create normalized continuum shape (centered at 0, range ~[-1, 1])
            Vector shape = slope * (x.array() - 0.5);
            
            // Scale and shift to desired range
            continuum = baseline + amplitude * baseline * shape.array();
            break;
        }
        case ContinuumType::RandomQuadratic: {
            // Generate quadratic coefficients
            std::uniform_real_distribution<> coeff_dist(-1.0, 1.0);
            double a = coeff_dist(rng);
            double b = coeff_dist(rng) * 0.5;
            
            // Create normalized continuum shape
            Vector shape = a * (x.array() - 0.5).square() + b * (x.array() - 0.5);
            
            // Normalize shape to [-1, 1] range
            double shape_max = shape.cwiseAbs().maxCoeff();
            if (shape_max > 0) {
                shape /= shape_max;
            }
            
            // Scale and shift to desired range
            continuum = baseline + amplitude * baseline * shape.array();
            break;
        }
        case ContinuumType::RandomPolynomial: {
            // 4th order polynomial with controlled coefficients
            std::uniform_real_distribution<> coeff_dist(-1.0, 1.0);
            double coeffs[5];
            
            // Decreasing influence for higher orders
            coeffs[0] = coeff_dist(rng);           // linear term
            coeffs[1] = coeff_dist(rng) * 0.5;     // quadratic
            coeffs[2] = coeff_dist(rng) * 0.25;    // cubic
            coeffs[3] = coeff_dist(rng) * 0.125;   // quartic
            coeffs[4] = 0.0;                       // constant (we'll use baseline instead)

            // Build polynomial shape
            Vector shape = Vector::Zero(n);
            for(int i = 0; i < n; ++i) {
                double xi = x[i] - 0.5;  // Center at 0.5
                double val = 0.0;
                double xi_pow = xi;
                for(int j = 0; j < 4; ++j) {
                    val += coeffs[j] * xi_pow;
                    xi_pow *= xi;
                }
                shape[i] = val;
            }
            
            // Normalize shape to [-1, 1] range
            double shape_max = shape.cwiseAbs().maxCoeff();
            if (shape_max > 0) {
                shape /= shape_max;
            }
            
            // Scale and shift to desired range
            continuum = baseline + amplitude * baseline * shape.array();
            break;
        }
        default:
            continuum = continuum * baseline;
            break;
    }

    // Hard clamp to ensure we stay within bounds
    continuum = continuum.cwiseMax(MIN_COUNTS).cwiseMin(MAX_COUNTS);
    
    return continuum;
}

// Generate one mock spectrum
void generate_mock_spectrum(
    const MockDataConfig& config,
    const ModelGrid& grid,
    const StellarParams& params,
    int set_index,
    int spec_index,
    double noise_level,
    std::mt19937& rng,
    const std::string& output_path,
    nlohmann::json& metadata
) {
    // Generate wavelength grid
    Vector wave_obs;
    if (config.use_nyquist_sampling) {
        wave_obs = build_nyquist_grid(
            config.wave_start, 
            config.wave_end,
            config.res_offset,
            config.res_slope
        );
    } else {
        // Linear sampling using wave_sampling
        int n_points = static_cast<int>((config.wave_end - config.wave_start) / config.wave_sampling) + 1;
        wave_obs = Vector::LinSpaced(n_points, config.wave_start, config.wave_end);
    }
    
    // Generate synthetic spectrum
    Spectrum synth = compute_synthetic(
        grid, params, wave_obs,
        config.res_offset, config.res_slope
    );
    
    // Apply continuum
    Vector continuum = generate_continuum(wave_obs, config.continuum_type, rng);
    synth.flux = synth.flux.cwiseProduct(continuum);
    
    // Add Gaussian noise
    std::normal_distribution<> noise_dist(0.0, 1.0);
    const double median_flux = median(synth.flux);
    const double sigma = noise_level * median_flux;
    
    for(int i = 0; i < synth.flux.size(); ++i) {
        synth.flux[i] += sigma * noise_dist(rng);
    }
    
    // Generate realistic uncertainties (slightly underestimated)
    std::uniform_real_distribution<> err_scale(0.8, 1.2);
    synth.sigma = Vector::Constant(synth.flux.size(), sigma * err_scale(rng));
    
    // Save spectrum to file
    std::ofstream file(output_path);
    file << std::scientific << std::setprecision(10);
    for(int i = 0; i < wave_obs.size(); ++i) {
        file << wave_obs[i] << " " 
             << synth.flux[i] << " " 
             << synth.sigma[i] << "\n";
    }
    file.close();
    
    // Update metadata
    metadata["noise_level"] = noise_level;
    metadata["continuum_type"] = static_cast<int>(config.continuum_type);
}

// Worker thread for generation
void generation_worker(
    const MockDataConfig& config,
    ModelGrid* grid,
    const std::vector<StellarParams>& param_sets,
    const std::vector<int>& multiplicities,
    GenerationProgress& progress,
    int thread_id,
    int start_idx,
    int end_idx
) {
    std::random_device rd;
    const bool deterministic = (config.seed != 0);
    // Only used when config.seed == 0 (legacy, non-reproducible mode).
    std::mt19937 shared_rng(rd() + thread_id);
    std::uniform_real_distribution<> noise_dist(config.noise_min, config.noise_max);

    for(int set_idx = start_idx; set_idx < end_idx; ++set_idx) {
        if(progress.should_stop) break;

        // Seeding per *set* rather than per thread keeps the generated data
        // identical no matter how the sets are distributed over threads.
        std::mt19937 per_set_rng(config.seed +
                                 0x9E3779B9u * static_cast<unsigned>(set_idx + 1));
        std::mt19937& rng = deterministic ? per_set_rng : shared_rng;

        // Create directory for this parameter set
        std::stringstream ss;
        ss << config.output_dir << "/" 
           << std::setfill('0') << std::setw(4) << (set_idx + 1);
        std::string set_dir = ss.str();
        fs::create_directories(set_dir);
        
        // Metadata for this set
        nlohmann::json metadata;
        metadata["set_index"] = set_idx + 1;
        metadata["true_parameters"] = {
            {"teff", param_sets[set_idx].teff},
            {"logg", param_sets[set_idx].logg},
            {"z", param_sets[set_idx].z},
            {"he", param_sets[set_idx].he},
            {"xi", param_sets[set_idx].xi},
            {"vrad", param_sets[set_idx].vrad},
            {"vsini", param_sets[set_idx].vsini},
            {"zeta", param_sets[set_idx].zeta}
        };
        metadata["spectra"] = nlohmann::json::array();
        
        // Generate multiple observations
        const int n_obs = multiplicities[set_idx];
        std::vector<std::string> spectrum_files;
        
        for(int obs_idx = 0; obs_idx < n_obs; ++obs_idx) {
            // Vary RV slightly for each observation
            StellarParams obs_params = param_sets[set_idx];
            if(obs_idx > 0) {
                obs_params.vrad += config.vrad_scatter.sample(rng);
            }
            
            // Generate noise level
            double noise = noise_dist(rng);
            
            // Generate spectrum
            std::stringstream spec_ss;
            spec_ss << "spectrum_" << std::setfill('0') << std::setw(2) << (obs_idx + 1) << ".txt";
            std::string spec_filename = spec_ss.str();
            std::string spec_path = set_dir + "/" + spec_filename;
            
            nlohmann::json spec_meta;
            spec_meta["filename"] = spec_filename;
            spec_meta["observation_index"] = obs_idx + 1;
            spec_meta["vrad_actual"] = obs_params.vrad;
            
            generate_mock_spectrum(
                config, *grid, obs_params, 
                set_idx, obs_idx, noise, rng,
                spec_path, spec_meta
            );
            
            metadata["spectra"].push_back(spec_meta);
            spectrum_files.push_back(spec_path);
            
            progress.current++;
            
            std::stringstream status;
            status << "Thread " << thread_id << ": Set " << (set_idx + 1) 
                   << "/" << param_sets.size() 
                   << ", Spectrum " << (obs_idx + 1) << "/" << n_obs 
                   << " (" << (100 * progress.current / progress.total) << "%)";
            progress.update_status(status.str());
        }
        
        // Save metadata
        std::ofstream meta_file(set_dir + "/metadata.json");
        meta_file << std::setw(2) << metadata;
        meta_file.close();
        
        // Generate input.json for fitting
        nlohmann::json input_json;
        nlohmann::json initial_guess_json;
        for (const auto& [key, guess] : config.initial_guesses) {
            initial_guess_json[key] = {
                {"value", guess.value},
                {"freeze", guess.freeze}
            };
        }
        input_json["initialGuess"] = initial_guess_json;
        
        input_json["grids"] = {config.grid_path};
        
        // Build files array for observation
        nlohmann::json files_array = nlohmann::json::array();
        for(const auto& file : spectrum_files) {
            files_array.push_back({
                {"filename", fs::absolute(file).string()},
                {"spectype", "ASCII_with_3_columns"},
                {"resOffset", config.res_offset},
                {"resSlope", config.res_slope},
                {"barycorr", 0.0}
            });
        }
        
        input_json["observations"] = {{
            {"files", files_array},
            {"ignore", nlohmann::json::array()},
            {"csplineAnchorpoints", {
                {config.wave_start, config.wave_start + 500, 50},
                {config.wave_start + 500, config.wave_end - 500, 100},
                {config.wave_end - 500, config.wave_end, 50}
            }},
            {"waveCut", {config.wave_start, config.wave_end}}
        }};
        
        std::stringstream result_ss;
        result_ss << "./fit_results/" << std::setfill('0') << std::setw(4) << (set_idx + 1);
        input_json["outputPath"] = result_ss.str();
        input_json["saveModel"] = "fits";
        
        std::ofstream input_file(set_dir + "/input.json");
        input_file << std::setw(2) << input_json;
        input_file.close();
    }
}

// Load configuration from JSON file
MockDataConfig load_config_from_json(const std::string& filename) {
    MockDataConfig config;
    
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open config file: " + filename);
    }
    
    nlohmann::json j;
    file >> j;
    
    // Load basic settings
    if (j.contains("num_parameter_sets")) config.num_parameter_sets = j["num_parameter_sets"];
    if (j.contains("wave_start")) config.wave_start = j["wave_start"];
    if (j.contains("wave_end")) config.wave_end = j["wave_end"];
    if (j.contains("wave_sampling")) config.wave_sampling = j["wave_sampling"];
    if (j.contains("res_offset")) config.res_offset = j["res_offset"];
    if (j.contains("res_slope")) config.res_slope = j["res_slope"];
    
    // Load noise config
    if (j.contains("noise")) {
        if (j["noise"].is_array() && j["noise"].size() == 2) {
            config.noise_min = j["noise"][0];
            config.noise_max = j["noise"][1];
        } else if (j["noise"].is_number()) {
            config.noise_min = config.noise_max = j["noise"];
        }
    }
    
    // Load continuum type
    if (j.contains("continuum")) {
        std::string cont = j["continuum"];
        if (cont == "linear") config.continuum_type = ContinuumType::RandomLinear;
        else if (cont == "quadratic") config.continuum_type = ContinuumType::RandomQuadratic;
        else if (cont == "polynomial") config.continuum_type = ContinuumType::RandomPolynomial;
        else config.continuum_type = ContinuumType::None;
    }
    
    // Load parameter distributions
    if (j.contains("parameters")) {
        auto& p = j["parameters"];
        if (p.contains("teff")) config.teff = ParameterConfig::from_string(p["teff"]);
        if (p.contains("logg")) config.logg = ParameterConfig::from_string(p["logg"]);
        if (p.contains("z")) config.z = ParameterConfig::from_string(p["z"]);
        if (p.contains("he")) config.he = ParameterConfig::from_string(p["he"]);
        if (p.contains("xi")) config.xi = ParameterConfig::from_string(p["xi"]);
        if (p.contains("vrad")) config.vrad = ParameterConfig::from_string(p["vrad"]);
        if (p.contains("vrad_scatter")) config.vrad_scatter = ParameterConfig::from_string(p["vrad_scatter"]);
        if (p.contains("vsini")) config.vsini = ParameterConfig::from_string(p["vsini"]);
        if (p.contains("zeta")) config.zeta = ParameterConfig::from_string(p["zeta"]);
    }
    
    // Load multiplicity
    if (j.contains("multiplicity")) {
        if (j["multiplicity"].is_array() && j["multiplicity"].size() == 2) {
            config.multiplicity_min = j["multiplicity"][0];
            config.multiplicity_max = j["multiplicity"][1];
        } else if (j["multiplicity"].is_number()) {
            config.multiplicity_min = config.multiplicity_max = j["multiplicity"];
        }
    }
    
    // Load paths
    if (j.contains("grid_path")) config.grid_path = j["grid_path"];
    if (j.contains("output_dir")) config.output_dir = j["output_dir"];
    if (j.contains("num_threads")) config.num_threads = j["num_threads"];
    if (j.contains("seed")) config.seed = j["seed"];
    
    // Load wavelength sampling mode
    if (j.contains("use_nyquist_sampling")) {
        config.use_nyquist_sampling = j["use_nyquist_sampling"];
    }
    
    // Load initial guesses
    if (j.contains("initialGuess")) {
        for (auto& [key, val] : j["initialGuess"].items()) {
            MockDataConfig::InitialGuess guess;
            if (val.contains("value")) guess.value = val["value"];
            if (val.contains("freeze")) guess.freeze = val["freeze"];
            config.initial_guesses[key] = guess;
        }
    }
    
    return config;
}

MockDataConfig::InitialGuess parse_initial_guess(const std::string& str) {
    MockDataConfig::InitialGuess guess;
    auto pos = str.find(',');
    if (pos != std::string::npos) {
        guess.value = std::stod(str.substr(0, pos));
        std::string freeze_str = str.substr(pos + 1);
        guess.freeze = (freeze_str == "true" || freeze_str == "1");
    } else {
        guess.value = std::stod(str);
        guess.freeze = false;
    }
    return guess;
}


int main(int argc, char** argv) {
    // Parse command line arguments
    cxxopts::Options options("mock_data_generator", 
                            "Generate mock spectra for testing fitting routines");
    
    options.add_options()
        ("c,config", "Configuration JSON file", cxxopts::value<std::string>())
        ("n,num-sets", "Number of parameter sets to generate", cxxopts::value<int>()->default_value("10"))
        ("wave-start", "Start wavelength (Angstrom)", cxxopts::value<double>()->default_value("3800"))
        ("wave-end", "End wavelength (Angstrom)", cxxopts::value<double>()->default_value("6800"))
        ("wave-sampling", "Wavelength sampling (Angstrom)", cxxopts::value<double>()->default_value("0.1"))
        ("res-offset", "Resolution offset", cxxopts::value<double>()->default_value("0"))
        ("res-slope", "Resolution slope", cxxopts::value<double>()->default_value("0.3"))
        ("noise", "Noise level(s) as percentage (single value or min:max)", cxxopts::value<std::string>()->default_value("1:5"))
        ("continuum", "Continuum type: none|linear|quadratic|polynomial", cxxopts::value<std::string>()->default_value("none"))
        ("multiplicity", "Observations per set (single value or min:max)", cxxopts::value<std::string>()->default_value("1:5"))
        ("teff", "Teff distribution (value | mean,sigma | min:max)", cxxopts::value<std::string>()->default_value("25000"))
        ("logg", "log g distribution", cxxopts::value<std::string>()->default_value("5.5"))
        ("z", "Metallicity distribution", cxxopts::value<std::string>()->default_value("0"))
        ("he", "Helium distribution", cxxopts::value<std::string>()->default_value("-2"))
        ("xi", "Microturbulence distribution", cxxopts::value<std::string>()->default_value("0"))
        ("vrad", "Radial velocity distribution", cxxopts::value<std::string>()->default_value("0,10"))
        ("vrad-scatter", "RV scatter between observations (km/s)", cxxopts::value<std::string>()->default_value("-5:5"))
        ("vsini", "v sin i distribution", cxxopts::value<std::string>()->default_value("10"))
        ("zeta", "Macroturbulence distribution", cxxopts::value<std::string>()->default_value("0"))
        ("g,grid", "Model grid path", cxxopts::value<std::string>()->default_value("sdB/processed/"))
        ("o,output", "Output directory", cxxopts::value<std::string>()->default_value("./mock_data/"))
        ("j,threads", "Number of threads", cxxopts::value<int>()->default_value("1"))
        ("v,verbose", "Verbose output", cxxopts::value<bool>()->default_value("false"))
        ("nyquist", "Use Nyquist sampling (default: false)", cxxopts::value<bool>()->default_value("false"))
        ("init-vrad", "Initial vrad guess (value,freeze)", cxxopts::value<std::string>()->default_value("0,false"))
        ("init-vsini", "Initial vsini guess (value,freeze)", cxxopts::value<std::string>()->default_value("10,false"))
        ("init-zeta", "Initial zeta guess (value,freeze)", cxxopts::value<std::string>()->default_value("0,true"))
        ("init-teff", "Initial teff guess (value,freeze)", cxxopts::value<std::string>()->default_value("25000,false"))
        ("init-logg", "Initial logg guess (value,freeze)", cxxopts::value<std::string>()->default_value("5.5,false"))
        ("init-xi", "Initial xi guess (value,freeze)", cxxopts::value<std::string>()->default_value("0,true"))
        ("init-z", "Initial z guess (value,freeze)", cxxopts::value<std::string>()->default_value("0,true"))
        ("init-he", "Initial HE guess (value,freeze)", cxxopts::value<std::string>()->default_value("-2,false"))
        ("seed", "RNG seed: 0 = random (default), non-zero = fully reproducible",
         cxxopts::value<unsigned>()->default_value("0"))
        ("h,help", "Print usage");
    
    auto result = options.parse(argc, argv);
    
    if (result.count("help")) {
        std::cout << options.help() << std::endl;
        std::cout << "\nParameter distribution formats:\n"
                  << "  Fixed value:    25000\n"
                  << "  Gaussian:       25000,1000  (mean,sigma)\n"
                  << "  Uniform:        20000:30000  (min:max)\n"
                  << "\nExample config.json:\n"
                  << "{\n"
                  << "  \"num_parameter_sets\": 100,\n"
                  << "  \"wave_start\": 3800,\n"
                  << "  \"wave_end\": 6800,\n"
                  << "  \"noise\": [0.01, 0.05],\n"
                  << "  \"continuum\": \"linear\",\n"
                  << "  \"parameters\": {\n"
                  << "    \"teff\": \"25000,2000\",\n"
                  << "    \"logg\": \"5.0:6.0\",\n"
                  << "    \"vrad\": \"0,10\"\n"
                  << "    \"vrad_scatter\": \"-5:5\",\n"
                  << "  },\n"
                  << "  \"multiplicity\": [1, 5]\n"
                  << "}\n";
        return 0;
    }
    
    // Load configuration
    MockDataConfig config;
    
    if (result.count("config")) {
        try {
            config = load_config_from_json(result["config"].as<std::string>());
            std::cout << "Loaded configuration from " << result["config"].as<std::string>() << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Error loading config file: " << e.what() << std::endl;
            return 1;
        }
    } else {
        // Parse from command line
        config.num_parameter_sets = result["num-sets"].as<int>();
        config.wave_start = result["wave-start"].as<double>();
        config.wave_end = result["wave-end"].as<double>();
        config.wave_sampling = result["wave-sampling"].as<double>();
        config.res_offset = result["res-offset"].as<double>();
        config.res_slope = result["res-slope"].as<double>();
        
        // Parse noise
        std::string noise_str = result["noise"].as<std::string>();
        if (noise_str.find(':') != std::string::npos) {
            auto pos = noise_str.find(':');
            config.noise_min = std::stod(noise_str.substr(0, pos)) / 100.0;
            config.noise_max = std::stod(noise_str.substr(pos + 1)) / 100.0;
        } else {
            config.noise_min = config.noise_max = std::stod(noise_str) / 100.0;
        }
        
        // Parse continuum
        std::string cont = result["continuum"].as<std::string>();
        if (cont == "linear") config.continuum_type = ContinuumType::RandomLinear;
        else if (cont == "quadratic") config.continuum_type = ContinuumType::RandomQuadratic;
        else if (cont == "polynomial") config.continuum_type = ContinuumType::RandomPolynomial;
        else config.continuum_type = ContinuumType::None;
        
        // Parse multiplicity
        std::string mult_str = result["multiplicity"].as<std::string>();
        if (mult_str.find(':') != std::string::npos) {
            auto pos = mult_str.find(':');
            config.multiplicity_min = std::stoi(mult_str.substr(0, pos));
            config.multiplicity_max = std::stoi(mult_str.substr(pos + 1));
        } else {
            config.multiplicity_min = config.multiplicity_max = std::stoi(mult_str);
        }
        
        // Parse parameters
        config.teff = ParameterConfig::from_string(result["teff"].as<std::string>());
        config.logg = ParameterConfig::from_string(result["logg"].as<std::string>());
        config.z = ParameterConfig::from_string(result["z"].as<std::string>());
        config.he = ParameterConfig::from_string(result["he"].as<std::string>());
        config.xi = ParameterConfig::from_string(result["xi"].as<std::string>());
        config.vrad = ParameterConfig::from_string(result["vrad"].as<std::string>());
        config.vrad_scatter = ParameterConfig::from_string(result["vrad-scatter"].as<std::string>());
        config.vsini = ParameterConfig::from_string(result["vsini"].as<std::string>());
        config.zeta = ParameterConfig::from_string(result["zeta"].as<std::string>());
        
        config.grid_path = result["grid"].as<std::string>();
        config.output_dir = result["output"].as<std::string>();
        config.num_threads = result["threads"].as<int>();

        config.use_nyquist_sampling = result["nyquist"].as<bool>();

        config.initial_guesses["c1_vrad"] = parse_initial_guess(result["init-vrad"].as<std::string>());
        config.initial_guesses["c1_vsini"] = parse_initial_guess(result["init-vsini"].as<std::string>());
        config.initial_guesses["c1_zeta"] = parse_initial_guess(result["init-zeta"].as<std::string>());
        config.initial_guesses["c1_teff"] = parse_initial_guess(result["init-teff"].as<std::string>());
        config.initial_guesses["c1_logg"] = parse_initial_guess(result["init-logg"].as<std::string>());
        config.initial_guesses["c1_xi"] = parse_initial_guess(result["init-xi"].as<std::string>());
        config.initial_guesses["c1_z"] = parse_initial_guess(result["init-z"].as<std::string>());
        config.initial_guesses["c1_HE"] = parse_initial_guess(result["init-he"].as<std::string>());
    }
    
    config.verbose = result["verbose"].as<bool>();
    // An explicit --seed overrides whatever the config file said.
    if (result.count("seed")) config.seed = result["seed"].as<unsigned>();

    // Print configuration summary
    std::cout << "\n=== Mock Data Generation Configuration ===\n";
    std::cout << "Parameter sets: " << config.num_parameter_sets << "\n";
    std::cout << "Wavelength range: " << config.wave_start << " - " << config.wave_end 
              << " Å (sampling: " << config.wave_sampling << " Å)\n";
    std::cout << "Resolution: R = " << config.res_offset << " + " << config.res_slope << " * λ\n";
    std::cout << "Noise: " << (config.noise_min * 100) << "% - " << (config.noise_max * 100) << "%\n";
    std::cout << "Multiplicity: " << config.multiplicity_min << " - " << config.multiplicity_max << " obs/set\n";
    std::cout << "\nParameter distributions:\n";
    std::cout << "  Teff: " << config.teff.to_string() << "\n";
    std::cout << "  log g: " << config.logg.to_string() << "\n";
    std::cout << "  [Z]: " << config.z.to_string() << "\n";
    std::cout << "  [He]: " << config.he.to_string() << "\n";
    std::cout << "  ξ: " << config.xi.to_string() << "\n";
    std::cout << "  vrad: " << config.vrad.to_string() << "\n";
    std::cout << "  vrad_scatter: " << config.vrad_scatter.to_string() << "\n";
    std::cout << "  vsini: " << config.vsini.to_string() << "\n";
    std::cout << "  ζ: " << config.zeta.to_string() << "\n";

    std::cout << "\nGrid: " << config.grid_path << "\n";
    std::cout << "Output: " << config.output_dir << "\n";
    std::cout << "Threads: " << config.num_threads << "\n";
    std::cout << "Seed: " << config.seed
              << (config.seed ? " (reproducible)" : " (random)") << "\n";
    std::cout << "==========================================\n\n";
    
    // Load base paths from global config
    std::vector<std::string> base_paths;
    try {
        auto global_cfg = load_json("global_settings.json");
        expand_env(global_cfg);
        base_paths = global_cfg["basePaths"].get<std::vector<std::string>>();
    } catch(...) {
        base_paths = {"./", "../"};
    }
    
    // Create output directory
    fs::create_directories(config.output_dir);
    
    // Generate parameter sets
    std::cout << "Generating parameter sets...\n";
    std::random_device rd;
    std::mt19937 rng(config.seed != 0 ? config.seed : rd());

    std::vector<StellarParams> param_sets;
    std::vector<int> multiplicities;
    
    for(int i = 0; i < config.num_parameter_sets; ++i) {
        StellarParams params;
        params.teff = config.teff.sample(rng);
        params.logg = config.logg.sample(rng);
        params.z = config.z.sample(rng);
        params.he = config.he.sample(rng);
        params.xi = config.xi.sample(rng);
        params.vrad = config.vrad.sample(rng);
        params.vsini = config.vsini.sample(rng);
        params.zeta = config.zeta.sample(rng);
        
        param_sets.push_back(params);
        
        // Sample multiplicity
        std::uniform_int_distribution<> mult_dist(
            config.multiplicity_min, config.multiplicity_max
        );
        multiplicities.push_back(mult_dist(rng));
    }
    
    // Sort parameter sets by stellar parameters for efficient caching
    std::cout << "Sorting parameter sets for cache efficiency...\n";
    std::vector<size_t> indices(param_sets.size());
    std::iota(indices.begin(), indices.end(), 0);
    
    std::sort(indices.begin(), indices.end(), 
        [&param_sets](size_t a, size_t b) {
            const auto& pa = param_sets[a];
            const auto& pb = param_sets[b];
            if(pa.teff != pb.teff) return pa.teff < pb.teff;
            if(pa.logg != pb.logg) return pa.logg < pb.logg;
            if(pa.z != pb.z) return pa.z < pb.z;
            if(pa.he != pb.he) return pa.he < pb.he;
            return pa.xi < pb.xi;
        });
    
    // Reorder based on sorted indices
    std::vector<StellarParams> sorted_params;
    std::vector<int> sorted_mults;
    for(auto idx : indices) {
        sorted_params.push_back(param_sets[idx]);
        sorted_mults.push_back(multiplicities[idx]);
    }
    param_sets = sorted_params;
    multiplicities = sorted_mults;
    
    // Load grid
    std::cout << "Loading model grid from " << config.grid_path << "...\n";
    ModelGrid* grid = nullptr;
    try {
        grid = new ModelGrid(base_paths, config.grid_path);
        std::cout << "Grid loaded successfully.\n";
    } catch(const std::exception& e) {
        std::cerr << "Error loading grid: " << e.what() << std::endl;
        return 1;
    }
    
    // Calculate total spectra
    int total_spectra = 0;
    for(auto m : multiplicities) total_spectra += m;
    
    GenerationProgress progress;
    progress.total = total_spectra;
    progress.verbose = config.verbose;
    
    std::cout << "\nGenerating " << total_spectra << " spectra from " 
              << config.num_parameter_sets << " parameter sets...\n";
    
    // Multi-threaded generation
    if(config.num_threads > 1) {
        std::vector<std::thread> threads;
        int sets_per_thread = param_sets.size() / config.num_threads;
        
        for(int t = 0; t < config.num_threads; ++t) {
            int start_idx = t * sets_per_thread;
            int end_idx = (t == config.num_threads - 1) ? param_sets.size() : (t + 1) * sets_per_thread;
            
            threads.emplace_back(generation_worker,
                               std::ref(config), grid, 
                               std::ref(param_sets), std::ref(multiplicities),
                               std::ref(progress), t, start_idx, end_idx);
        }
        
        for(auto& t : threads) {
            t.join();
        }
    } else {
        // Single-threaded
        generation_worker(config, grid, param_sets, multiplicities, 
                         progress, 0, 0, param_sets.size());
    }
    
    std::cout << "\n\n=== Generation Complete ===\n";
    std::cout << "Generated " << total_spectra << " spectra in " 
              << config.num_parameter_sets << " directories\n";
    std::cout << "Output location: " << fs::absolute(config.output_dir) << "\n";
    std::cout << "Each directory contains:\n";
    std::cout << "  - spectrum_*.txt files (mock observations)\n";
    std::cout << "  - metadata.json (true parameters)\n";
    std::cout << "  - input.json (ready for fitting)\n";
    
    delete grid;
    return 0;
}