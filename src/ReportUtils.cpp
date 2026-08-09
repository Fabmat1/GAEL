#include "specfit/ReportUtils.hpp"
#include "specfit/ContinuumUtils.hpp"
#include "matplotlibcpp.h"               // <-- header-only Python wrapper
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <iostream>     
#include <cmath>
#include <limits>   // std::numeric_limits<double>::quiet_NaN()


namespace plt = matplotlibcpp;           // shorthand
namespace fs  = std::filesystem;
using   specfit::Vector;

/* ===================================================================== */
/*            H e l p e r s   f o r   t a b l e   f o r m a t t i n g     */
/* ===================================================================== */
namespace specfit {

static std::string latex_name(const std::string& tag)
{
    static const std::map<std::string,std::string> lut = {
        {"vrad",  "Radial velocity $\\varv_{\\mathrm{rad}}$"},
        {"vsini", "Projected rotational velocity $\\varv\\sin(i)$"},
        {"zeta",  "Macroturbulence $\\zeta$"},
        {"teff",  "Effective temperature $T_{\\mathrm{eff}}$"},
        {"logg",  "Surface gravity $\\log g$"},
        {"xi",    "Microturbulence $\\xi$"},
        {"z",     "Metallicity $z$"},
        {"he",    "He abundance $\\log n(\\mathrm{He})$"}
    };
    auto it = lut.find(tag);
    return (it==lut.end())?tag:it->second;
}

static std::string unit_for(const std::string& tag)
{
    if (tag=="vrad" || tag=="vsini" || tag=="zeta" || tag=="xi")
        return "\\,\\mathrm{km\\,s}^{-1}";
    if (tag=="teff")
        return "\\,\\mathrm{K}";
    if (tag=="z")
        return "\\,\\mathrm{dex}";
    if (tag=="lambda")
        return "\\,\\AA";
    return "";
}

static std::string fmt_number(double v, const std::string& tag,
                              int p_fixed  = 1,
                              int p_float  = 2)
{
    std::ostringstream s;
    if (tag=="teff")          s<<std::fixed<<std::setprecision(0)<<v;
    else if (tag=="logg")     s<<std::fixed<<std::setprecision(3)<<v;
    else                      s<<std::fixed<<std::setprecision(p_fixed)<<v;
    return s.str();
}


/* ===================================================================== */
/*                       M u l t i P a n e l  P l o t t e r              */
/* ===================================================================== */
static std::vector<double> to_std(const Vector& v,
                                  const std::vector<int>* mask=nullptr)
{
    std::vector<double> out;
    out.reserve(v.size());
    if (!mask) {
        for (int i = 0; i < v.size(); ++i) out.push_back(v[i]);
    } else {
        for (int i = 0; i < v.size(); ++i)
            if ((*mask)[i]) out.push_back(v[i]);
    }
    return out;
}

MultiPanelPlotter::MultiPanelPlotter(double xr, bool grey)
    : xrange_(xr), grey_(grey) {}

/* --------------------------------------------------------------------- */
/*  Draw the observed spectrum, model and ignored parts.
 *  Ignored data are shown with a **grey dotted line** that replaces the
 *  corresponding solid-black segment.  One point on each side of every
 *  excluded block is added so that the curve remains continuous.
 */
// ---------------------------------------------------------------------------
void MultiPanelPlotter::simple_plot(const std::string& pdf,
                                    const Spectrum&    spec,
                                    const Vector&      mdl) const
{
    const std::size_t N = spec.lambda.size();
    const double      NaN = std::numeric_limits<double>::quiet_NaN();

    /* --- build the three curves ---------------------------------------------------- */
    std::vector<double> lam (N);
    std::vector<double> f_ok (N, NaN);   // kept points  → solid black
    std::vector<double> f_bad(N, NaN);   // ignored pts  → grey, dotted

    for (std::size_t i = 0; i < N; ++i)
    {
        lam[i] = spec.lambda[i];

        const bool ignored = !spec.ignoreflag[i];
        const bool neigh_ignored =
            (i && !spec.ignoreflag[i - 1]) ||
            (i + 1 < N && !spec.ignoreflag[i + 1]);

        if (ignored) {                          // really ignored
            f_bad[i] = spec.flux[i];
        } else {                                // good point
            f_ok [i] = spec.flux[i];
            if (neigh_ignored)                  // guard point
                f_bad[i] = spec.flux[i];
        }
    }

    /* --- draw ---------------------------------------------------------------------- */
    plt::figure_size(1600, 600);
    plt::plot(lam, f_ok , "k");                                 // good data
    plt::plot(lam, f_bad, {{"color", "0.6"}, {"linestyle", ":"}}); // ignored
    plt::plot(to_std(spec.lambda), to_std(mdl), "r");           // model

    plt::tight_layout();
    plt::save(pdf);
    plt::clf();
}

/* ────────────────────────────────────────────────────────────────── */
/*  location helper – tries several fall-backs                       */
/* ────────────────────────────────────────────────────────────────── */
namespace {

std::string python_helper_path()
{
    namespace fs = std::filesystem;

    /* user override -------------------------------------------------- */
    if (const char* e = std::getenv("GAEL_PY_HELPER");
        e && fs::exists(e))
        return e;

#ifdef GAEL_INSTALL_DATADIR
    {
        fs::path p = fs::path(GAEL_INSTALL_DATADIR)
                     / "GAEL_multiplot.py";
        if (fs::exists(p)) return p.string();
    }
#endif

    /* same directory as executable ---------------------------------- */
#ifndef _WIN32
    char exe[4096]{};
    ssize_t n = ::readlink("/proc/self/exe", exe, sizeof(exe)-1);
    if (n>0) {
        fs::path p = fs::path(exe).parent_path() / "GAEL_multiplot.py";
        if (fs::exists(p)) return p.string();
    }
#else
    char exe[MAX_PATH];
    if (::GetModuleFileNameA(NULL, exe, MAX_PATH)) {
        fs::path p = fs::path(exe).parent_path() / "GAEL_multiplot.py";
        if (fs::exists(p)) return p.string();
    }
#endif

    /* last resort: rely on PATH ------------------------------------- */
    return "GAEL_multiplot.py";
}

} // unnamed namespace


/* ------------------------------------------------------------------ */
void MultiPanelPlotter::plot(const std::string& pdf_path,
                             const Spectrum&    spec,
                             const Vector&      model,
                             const Vector&      continuum) const
{
    /* ---------- 1. write the CSV next to the PDF ------------------- */
    namespace fs  = std::filesystem;
    fs::path pdf  = pdf_path;
    fs::path csv  = pdf.parent_path() / (pdf.stem().string() + "_plotdata.csv");

    std::ofstream f(csv);
    f << "lambda,flux,sigma,model,spline,ignore\n";
    const std::size_t N = spec.lambda.size();
    for (std::size_t i = 0; i < N; ++i)
        f << std::setprecision(10)
          << spec.lambda[i]       << ','
          << spec.flux[i]         << ','
          << spec.sigma[i]        << ','
          << model[i]             << ','
          << continuum[i]         << ','
          << int(spec.ignoreflag[i]) << '\n';
    f.close();

    /* ---------- 2. call the Python helper -------------------------- */
    std::ostringstream cmd;
    cmd << "python3 \""            << python_helper_path() << "\" "
        << '"' << csv.string() << "\" "
        << '"' << pdf_path     << "\" "
        << xrange_;


    std::cout << "[MultiPanelPlotter] running: " << cmd.str() << '\n';
    int rc = std::system(cmd.str().c_str());
    if (rc != 0)
        std::cerr << "[MultiPanelPlotter] Python helper failed (" << rc
                  << ")\n";
}

/* ===================================================================== */
/*            High-level result generation (unchanged except includes)   */
/* ===================================================================== */
void generate_results(const std::string&              out_dir,
                      const UnifiedFitWorkflow&       wf,
                      const std::vector<DataSet>&     datasets,
                      const SharedModel&              model,
                      double                          xrange,
                      bool                            grey,
                      const std::vector<std::string>& untied_params,
                      bool                            make_plots,        // NEW
                      bool                            make_pdf)          // NEW
{
    fs::create_directories(out_dir);
    MultiPanelPlotter P(xrange, grey);

    const auto& all_p   = wf.get_parameters();
    const auto& all_err = wf.get_uncertainties();
    const auto& free_m  = wf.get_free_mask();

    /* ---------------------------  (1)  collect lines for PDF  --------- */
    std::vector<std::string> tex_lines;
    
    /*  --- Grid names ---------------------------------------------------*/
    for (const auto& g : model.grids)
        tex_lines.push_back("Grid & " + fs::path(g.axes().empty()
                              ? "(unknown)" : g.axes()[0].name).string() + R"(\\)");
    
    /*  --- fixed instrumental resolution (per spectrum) ---------------- */
    for (std::size_t d = 0; d < datasets.size(); ++d) {
        std::ostringstream lo, sl;
        lo << datasets[d].resOffset;
        sl << datasets[d].resSlope;
        tex_lines.push_back("Resolution offset $R = \\lambda/\\Delta\\lambda$ (fixed) & $" +
                            lo.str() + "$\\\\");
        tex_lines.push_back("Resolution slope $1/\\Delta\\lambda$ (fixed) & $" +
                            sl.str() + "$\\\\");
    }
    
    /* --- now all stellar parameters ----------------------------------- */
    const char* tags[8] = {"vrad","vsini","zeta","teff",
                           "logg","xi","z","he"};
    const int   n_comp  = model.params.size();
    const int   n_spec  = datasets.size();
    
    std::vector<std::tuple<std::string,double,double>> csv_entries;   // for later
    
    std::size_t idx = 0;
    auto is_untied = [&](const std::string& t){
        return std::find(untied_params.begin(), untied_params.end(), t) != untied_params.end();
    };
    
    for (int c = 0; c < n_comp; ++c)
    {
        for (int k = 0; k < 8; ++k)
        {
            const std::string tag = tags[k];
            const bool untied = is_untied(tag);
    
            const int   n_iter = untied ? n_spec : 1;
            for (int d = 0; d < n_iter; ++d, ++idx)
            {
                const bool fixed = !free_m[idx];
                const double val = all_p[idx];
                const double err = (idx < all_err.size()) ? all_err[idx] : 0.0;
    
                /* ----------- full parameter name for CSV ---------------- */
                std::string csv_name = "c" + std::to_string(c+1) + "_" + tag;
                if (untied) csv_name += "_d" + std::to_string(d+1);
                csv_entries.emplace_back(csv_name, val, err);
    
                /* ----------- skip spline parameters in PDF -------------- */
                std::string disp = latex_name(tag);
                if (n_comp > 1) disp += " (c" + std::to_string(c+1) + ")";
                if (untied)     disp += " (spec " + std::to_string(d+1) + ")";
                if (fixed)      disp += " (fixed)";
    
                /* numerical formatting ----------------------------------- */
                const std::string unit = unit_for(tag);
                std::string num;
                if (!fixed && err>0.0)
                    num = fmt_number(val, tag) + " \\pm " + fmt_number(err, tag);
                else
                    num = fmt_number(val, tag);
    
                tex_lines.push_back(disp + " & $" + num + unit + "$" + R"(\\)" + "\n");
            }
        }
    }
    
    /* ---------------------------  (2) continuum anchors for CSV only ---*/
    for (std::size_t d = 0; d < datasets.size(); ++d)
    {
        int cont_start = n_comp * 8;
        for (std::size_t j = 0; j < d; ++j)
            cont_start += datasets[j].cont_y.size();
    
        const int na = datasets[d].cont_y.size();
        for (int a = 0; a < na; ++a, ++idx)
        {
            std::string nm = fs::path(datasets[d].name).stem().string()
                             + "_cont" + std::to_string(a);
            const double val = all_p[idx];
            const double err = (idx < all_err.size()) ? all_err[idx] : 0.0;
            csv_entries.emplace_back(nm, val, err);
        }
    }
    
    /* ---------------------------  (3) write LaTeX ----------------------*/
    const std::string tex_path = out_dir + "/fit_report.tex";
    {
        std::ofstream tex(tex_path);
        tex << R"(\documentclass{standalone}
        \usepackage{amsmath,txfonts,color}
        \begin{document}
        \renewcommand{\arraystretch}{1.2}
        \begin{tabular}{lr}
        \hline\hline
        Value & 68\% confidence interval\\
        \hline
        )";
        for (auto& l : tex_lines) tex << l << '\n';
        tex << R"(\hline\hline)" << '\n';
        tex << R"(\multicolumn{2}{l}{\textbf{Spectrum identifiers}} \\)" << '\n';;
        tex << R"(\hline )" << '\n';;
        for (std::size_t d = 0; d < datasets.size(); ++d)
        {
            tex << R"(spec )" << (d+1) << R"( & \verb|)"
                << fs::path(datasets[d].name).filename().string() << R"(|\\)" << '\n';
        }
        tex << R"(\hline
    \end{tabular}
    \end{document})";
    }
    
    /* compile (quiet) --------------------------------------------------- */
    if (make_pdf)
    {
        std::string cmd = "pdflatex -interaction=batchmode -halt-on-error "
                          "-output-directory=" + out_dir + " " + tex_path +
                          " > /dev/null 2>&1";
        std::system(cmd.c_str());
    }
    
    /* ---------------------------  (4) CSV ------------------------------*/
    {
        std::ofstream csv(out_dir + "/fit_parameters.csv");
        csv << "parameter,value,error\n";
        csv << "final_chi2," << wf.get_final_chi2() << ",0.0\n"; 
        for (auto& t : csv_entries)
        {
            csv << std::get<0>(t) << ','
                << std::setprecision(10) << std::get<1>(t) << ','
                << std::setprecision(10) << std::get<2>(t) << '\n';
        }
        /* continuum λ anchors (error = 0) ------------------------------ */
        for (std::size_t d = 0; d < datasets.size(); ++d)
        {
            const auto& ds = datasets[d];
            const int na   = ds.cont_x.size();
            std::string stem = fs::path(ds.name).stem().string();
            for (int a = 0; a < na; ++a)
            {
                csv << stem << "_contX" << a << ','
                    << std::setprecision(10) << ds.cont_x[a] << ",0\n";
            }
        }
    }

    /* (C) spectrum summary plots */
    if (make_plots){
        for (std::size_t d = 0; d < datasets.size(); ++d) {
            Vector mdl = wf.get_model_for_dataset(d);    // continuum-free synthetic
            /* recompute correct continuum offset --------------------------- */
            int total_cont = 0;
            for (const auto& ds : datasets) total_cont += ds.cont_y.size();
            const int cont_block_start =
                static_cast<int>(all_p.size()) - total_cont;
            
            int cs = 0;
            for (std::size_t j = 0; j < d; ++j) cs += datasets[j].cont_y.size();
            cs += cont_block_start;
            Eigen::Map<const Vector> cy(all_p.data() + cs, datasets[d].cont_y.size());
            Vector cont = AkimaSpline(datasets[d].cont_x, cy)(datasets[d].obs.lambda);

            std::string pdf = out_dir + "/" +
                              fs::path(datasets[d].name).stem().string() + ".pdf";

            P.plot(pdf,
                   datasets[d].obs,         /* whole Spectrum ------------------- */
                   mdl,
                   cont);                   /* keep / ignore now inside Spectrum */
        }
    }
}

} // namespace specfit