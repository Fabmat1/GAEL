#include "specfit/Platform.hpp"

#include <cstdint>
#include <string>
#include <system_error>
#include <vector>

#if defined(__APPLE__)
#  include <mach-o/dyld.h>
#elif defined(_WIN32)
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <windows.h>
#endif

namespace specfit {

namespace fs = std::filesystem;

fs::path executable_path()
{
    std::error_code ec;
#if defined(__APPLE__)
    /*  The first call reports the size it needs; the path it returns can go
     *  through symlinks, hence the canonical() below.                       */
    std::uint32_t size = 0;
    ::_NSGetExecutablePath(nullptr, &size);
    std::vector<char> buf(size + 1, '\0');
    if (::_NSGetExecutablePath(buf.data(), &size) != 0) return {};
    fs::path p = fs::canonical(fs::path(buf.data()), ec);
    return ec ? fs::path(buf.data()) : p;
#elif defined(_WIN32)
    /*  Wide characters, so a user name with umlauts in the install path
     *  survives; the buffer grows until the name fits.                      */
    std::vector<wchar_t> buf(260);
    for (;;) {
        const DWORD n = ::GetModuleFileNameW(
            nullptr, buf.data(), static_cast<DWORD>(buf.size()));
        if (n == 0) return {};
        if (n < buf.size()) return fs::path(std::wstring(buf.data(), n));
        buf.resize(buf.size() * 2);
    }
#else
    fs::path p = fs::canonical("/proc/self/exe", ec);
    return ec ? fs::path{} : p;
#endif
}

} // namespace specfit
