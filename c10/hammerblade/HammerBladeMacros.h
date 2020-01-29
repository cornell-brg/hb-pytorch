#pragma once

#include <c10/hammerblade/impl/hammerblade_cmake_macros.h>

#ifndef _MSC_VER
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-attributes"
#endif

// See c10/macros/Export.h for a detailed explanation of what the function
// of these macros are.

#ifdef _WIN32
#if defined(C10_HAMMERBLADE_BUILD_SHARED_LIBS)
#define C10_HAMMERBLADE_EXPORT __declspec(dllexport)
#define C10_HAMMERBLADE_IMPORT __declspec(dllimport)
#else
#define C10_HAMMERBLADE_EXPORT
#define C10_HAMMERBLADE_IMPORT
#endif
#else // _WIN32
#if defined(__GNUC__)
#define C10_HAMMERBLADE_EXPORT __attribute__((__visibility__("default")))
#else // defined(__GNUC__)
#define C10_HAMMERBLADE_EXPORT
#endif // defined(__GNUC__)
#define C10_HAMMERBLADE_IMPORT C10_HAMMERBLADE_EXPORT
#endif // _WIN32

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
