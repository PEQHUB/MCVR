#include "core/build/build_info.hpp"
#include <sstream>

namespace build_info {

std::string summary() {
    std::ostringstream out;
    out << "MCVR commit=" << kRepoCommit
        << " branch=" << kBranch
        << " dirty=" << (kDirty ? "true" : "false")
        << " abi=" << kTextureLoaderAbiVersion
        << " cache=" << kCacheSchemaVersion
        << " built=" << kBuildTimestamp;
    return out.str();
}

} // namespace build_info
