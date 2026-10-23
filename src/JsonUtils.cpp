#include "specfit/JsonUtils.hpp"
#include <fstream>
#include <regex>
#include <cstdlib>

namespace specfit {

nlohmann::json load_json(const std::string& path)
{
    std::ifstream f(path);
    nlohmann::json j;
    f >> j;
    return j;
}

static std::string expand(const std::string& input)
{
    static const std::regex re(R"(\$\{([^}]+)\})");
    std::string out = input;
    std::smatch m;
    while (std::regex_search(out, m, re)) {
        std::string var = m[1];
        const char* env = std::getenv(var.c_str());
#ifdef _WIN32
        /*  The shipped basePaths name ${HOME} and ${USER}, which Windows
         *  spells differently; without this they expanded to nothing and a
         *  grid in the user's home folder was never found.                 */
        if (!env && var == "HOME") env = std::getenv("USERPROFILE");
        if (!env && var == "USER") env = std::getenv("USERNAME");
#endif
        out.replace(m.position(0), m.length(0), env ? env : "");
    }
    return out;
}

void expand_env(nlohmann::json& j)
{
    if (j.is_string()) {
        j = expand(j.get<std::string>());
    } else if (j.is_array() || j.is_object()) {
        for (auto& el : j) expand_env(el);
    }
}

} // namespace specfit