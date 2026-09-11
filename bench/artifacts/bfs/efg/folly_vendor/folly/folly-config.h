/*
 * Hand-written stand-in for folly's CMake-GENERATED folly-config.h
 * (normally produced from CMake/folly-config.h.cmake by folly's own build).
 * This project vendors only the narrow EliasFanoCoding/Select64/Instructions
 * header closure needed by bench/artifacts/bfs/efg's adapter (see
 * ../../STATUS.md) rather than building all of folly, so there is no CMake
 * configure step to generate this file -- these are the feature flags that
 * are true for a modern x86_64 Linux + glibc + GCC toolchain (this machine),
 * left conservatively OFF for anything we don't use (compression libs,
 * gflags/glog, libunwind/dwarf symbolization) since the narrow header
 * closure here never exercises those paths.
 */
#pragma once

#if !defined(FOLLY_MOBILE)
#define FOLLY_MOBILE 0
#endif

#define FOLLY_HAVE_PTHREAD 1
#define FOLLY_HAVE_PTHREAD_ATFORK 1

/* no gflags/glog vendored -- see ../../../glog/logging.h shim */
/* #define FOLLY_HAVE_LIBGFLAGS 1 */
/* #define FOLLY_HAVE_LIBGLOG 1 */

/* #define FOLLY_USE_JEMALLOC 1 */

#define FOLLY_HAVE_ACCEPT4 1
#define FOLLY_HAVE_GETRANDOM 1
#define FOLLY_HAVE_PREADV 1
#define FOLLY_HAVE_PWRITEV 1
#define FOLLY_HAVE_CLOCK_GETTIME 1
#define FOLLY_HAVE_PIPE2 1
#define FOLLY_HAVE_SENDMMSG 1
#define FOLLY_HAVE_RECVMMSG 1
/* #define FOLLY_HAVE_OPENSSL_ASN1_TIME_DIFF 1 */

#define FOLLY_HAVE_IFUNC 1
#define FOLLY_HAVE_STD__IS_TRIVIALLY_COPYABLE 1
#define FOLLY_HAVE_UNALIGNED_ACCESS 1
#define FOLLY_HAVE_VLA 1
#define FOLLY_HAVE_WEAK_SYMBOLS 1
#define FOLLY_HAVE_LINUX_VDSO 1
#define FOLLY_HAVE_MALLOC_USABLE_SIZE 1
#define FOLLY_HAVE_INT128_T 1
#define FOLLY_HAVE_WCHAR_SUPPORT 1
/* #define FOLLY_HAVE_EXTRANDOM_SFMT19937 1 */
/* #define HAVE_VSNPRINTF_ERRORS 1 */

/* symbolization / unwind libs not vendored -- not used by this closure */
/* #define FOLLY_HAVE_LIBUNWIND 1 */
/* #define FOLLY_HAVE_DWARF 1 */
/* #define FOLLY_HAVE_ELF 1 */
#define FOLLY_HAVE_SWAPCONTEXT 1
/* #define FOLLY_HAVE_BACKTRACE 1 */
/* #define FOLLY_USE_SYMBOLIZER 1 */
#define FOLLY_DEMANGLE_MAX_SYMBOL_SIZE 1024

/* #define FOLLY_HAVE_SHADOW_LOCAL_WARNINGS 1 */

/* compression libs not vendored -- not used by this closure */
/* #define FOLLY_HAVE_LIBLZ4 1 */
/* #define FOLLY_HAVE_LIBLZMA 1 */
/* #define FOLLY_HAVE_LIBSNAPPY 1 */
/* #define FOLLY_HAVE_LIBZ 1 */
/* #define FOLLY_HAVE_LIBZSTD 1 */
/* #define FOLLY_HAVE_LIBBZ2 1 */

#define FOLLY_LIBRARY_SANITIZE_ADDRESS 0

/* #define FOLLY_SUPPORT_SHARED_LIBRARY 1 */

#define FOLLY_HAVE_LIBRT 1
