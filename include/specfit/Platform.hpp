#pragma once
#include <filesystem>

namespace specfit {

/*  The running executable, absolute, or an empty path when the OS would not
 *  say.  Linux reads /proc/self/exe, which macOS and Windows do not have --
 *  and the CLI looks for global_settings.json next to itself, so without
 *  this a Mac or Windows build found it only when started from its own
 *  directory.                                                              */
std::filesystem::path executable_path();

} // namespace specfit
