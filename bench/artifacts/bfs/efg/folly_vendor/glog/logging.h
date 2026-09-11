#pragma once
// Minimal, self-contained stand-in for the small slice of glog's public
// macro API that folly's (vendored, unmodified) EliasFanoCoding.h /
// Instructions.h / Select64.h use: CHECK_EQ/NE/LE/LT/GE/GT, CHECK, DCHECK
// (+ its comparison variants), VLOG. glog itself is not header-only (it
// needs gflags plus a compiled library) and this project has no sudo / no
// system-package-install path (see bench/ARTIFACT_GUIDE.md), so building
// real glog was out of scope for a login-node integration pass.
//
// This header reproduces glog's OBSERVABLE macro behavior exactly:
//   - CHECK*  always aborts with a message on failure (release or debug).
//   - DCHECK* only checks when NDEBUG is not defined (glog's own contract);
//     the guarded expression is not evaluated at all when NDEBUG is set,
//     matching glog's "compiled out in release" semantics.
//   - VLOG(n) is a no-op sink (functionally equivalent to running glog at
//     verbosity 0 -- discards the message, evaluates nothing observable).
// None of this touches Elias-Fano encode/decode logic: every algorithmic
// line in this vendor tree comes from the real, unmodified upstream folly
// source (see ../../STATUS.md "compressed representation" section).
#include <cstdio>
#include <cstdlib>
#include <sstream>

namespace efg_glog_shim {

struct CheckFatal {
    std::ostringstream oss;
    bool triggered;
    const char* file;
    int line;
    const char* expr;
    CheckFatal(bool ok, const char* file_, int line_, const char* expr_)
        : triggered(!ok), file(file_), line(line_), expr(expr_) {}
    ~CheckFatal() {
        if (triggered) {
            fprintf(stderr, "%s:%d: Check failed: %s %s\n", file, line, expr, oss.str().c_str());
            std::fflush(stderr);
            std::abort();
        }
    }
    std::ostringstream& stream() { return oss; }
};

struct LogSink {
    std::ostringstream oss;
    ~LogSink() {}  // discarded: VLOG/LOG are no-ops in this shim
    std::ostringstream& stream() { return oss; }
};

}  // namespace efg_glog_shim

#define CHECK(cond) \
    ::efg_glog_shim::CheckFatal(!!(cond), __FILE__, __LINE__, #cond).stream()
#define CHECK_EQ(a, b) ::efg_glog_shim::CheckFatal((a) == (b), __FILE__, __LINE__, #a " == " #b).stream()
#define CHECK_NE(a, b) ::efg_glog_shim::CheckFatal((a) != (b), __FILE__, __LINE__, #a " != " #b).stream()
#define CHECK_LE(a, b) ::efg_glog_shim::CheckFatal((a) <= (b), __FILE__, __LINE__, #a " <= " #b).stream()
#define CHECK_LT(a, b) ::efg_glog_shim::CheckFatal((a) < (b), __FILE__, __LINE__, #a " < " #b).stream()
#define CHECK_GE(a, b) ::efg_glog_shim::CheckFatal((a) >= (b), __FILE__, __LINE__, #a " >= " #b).stream()
#define CHECK_GT(a, b) ::efg_glog_shim::CheckFatal((a) > (b), __FILE__, __LINE__, #a " > " #b).stream()

#ifdef NDEBUG
#define EFG_GLOG_DCHECK_ACTIVE 0
#else
#define EFG_GLOG_DCHECK_ACTIVE 1
#endif

#define DCHECK(cond) \
    ::efg_glog_shim::CheckFatal(!EFG_GLOG_DCHECK_ACTIVE || !!(cond), __FILE__, __LINE__, #cond).stream()
#define DCHECK_EQ(a, b) \
    ::efg_glog_shim::CheckFatal(!EFG_GLOG_DCHECK_ACTIVE || ((a) == (b)), __FILE__, __LINE__, #a " == " #b).stream()
#define DCHECK_NE(a, b) \
    ::efg_glog_shim::CheckFatal(!EFG_GLOG_DCHECK_ACTIVE || ((a) != (b)), __FILE__, __LINE__, #a " != " #b).stream()
#define DCHECK_LE(a, b) \
    ::efg_glog_shim::CheckFatal(!EFG_GLOG_DCHECK_ACTIVE || ((a) <= (b)), __FILE__, __LINE__, #a " <= " #b).stream()
#define DCHECK_LT(a, b) \
    ::efg_glog_shim::CheckFatal(!EFG_GLOG_DCHECK_ACTIVE || ((a) < (b)), __FILE__, __LINE__, #a " < " #b).stream()
#define DCHECK_GE(a, b) \
    ::efg_glog_shim::CheckFatal(!EFG_GLOG_DCHECK_ACTIVE || ((a) >= (b)), __FILE__, __LINE__, #a " >= " #b).stream()
#define DCHECK_GT(a, b) \
    ::efg_glog_shim::CheckFatal(!EFG_GLOG_DCHECK_ACTIVE || ((a) > (b)), __FILE__, __LINE__, #a " > " #b).stream()

#define VLOG(n) ::efg_glog_shim::LogSink().stream()
#define LOG(sev) ::efg_glog_shim::LogSink().stream()
