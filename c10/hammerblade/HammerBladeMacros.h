#pragma once

#include <c10/hammerblade/impl/hammerblade_cmake_macros.h>

#ifndef _MSC_VER
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-attributes"
#endif

// This one is being used by libc10_hammerblade.so
#ifdef C10_HAMMERBLADE_BUILD_MAIN_LIB
#define C10_HAMMERBLADE_API C10_HAMMERBLADE_EXPORT
#else
#define C10_HAMMERBLADE_API C10_HAMMERBLADE_IMPORT
#endif

namespace c10 {
namespace hammerblade {

} // namespace hammerblade
} // namespace c10
