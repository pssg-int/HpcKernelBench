/*
 * HPC-KernelBench / hicma-x integration: build-only compatibility shim.
 *
 * hcore's compute kernels (hcore/src/compute/hcore_{s,d,z}gemm.c and
 * hcore_dgemm_fast.c) call LAPACKE_{s,d,z}ge_trans() when built with
 * -DLAPACKE_UTILS, guarded by "#ifdef LAPACKE_UTILS #include
 * <lapacke_utils.h> #endif" in each file. On this machine that resolves to
 * Cray LibSci's lapacke_utils.h, which DOES declare and export these three
 * functions (confirmed via `nm -D libsci_gnu.so` and this build's own
 * successful link) -- but hcore.h's own non-MKL/non-ARMPL "#else" branch
 * ALSO does "#include <lapacke.h>" (unqualified, angle-bracket), which
 * resolves via this project's own include_directories() list to DPLASMA's
 * bundled 2015 LAPACKE snapshot (dplasma/src/include/lapacke.h) rather than
 * the real vendor header, because that directory appears earlier in the
 * search path. That vendored snapshot predates the *ge_trans extensions
 * and, worse, shares the exact same "_LAPACKE_H_" include guard with the
 * real vendor lapacke.h -- so whichever copy is opened first wins and
 * silently blocks the other for the rest of the translation unit. Neither
 * "make our -I win" nor "rename one guard" turned out to be a clean fix:
 * the two LAPACKE snapshots differ in a few function signatures elsewhere
 * (LAPACKE_zuncsd2by1, LAPACKE_ztprfb, ...), so letting both bodies execute
 * produces genuine conflicting-redeclaration errors, and reordering the
 * search path only moves the same guard collision to a different file.
 *
 * The actual gap is much smaller than "which whole LAPACKE header wins":
 * hcore only ever calls three functions this vendored snapshot happens to
 * lack. Rather than resolve the guard collision at the header level, this
 * file force-declares exactly those three (via -include on the `hcore`
 * CMake target only, see build.sh) so the compiler has a prototype
 * regardless of which lapacke.h ends up included; the real, exported
 * symbols are resolved at link time from Cray LibSci (BLAS_LIBRARIES),
 * which already provides them. This is a build-system fix -- no numerics,
 * no HCORE kernel source, and no submodule content touched at all (this
 * file lives outside every submodule and is wired in purely via a compiler
 * flag in build.sh).
 *
 * Signatures match Cray LibSci's lapacke_utils.h exactly (int matrix_layout,
 * plain `int` for lapack_int since this build is LP64, not ILP64, so
 * API_SUFFIX() is the identity and the exported symbol names below are
 * unsuffixed) -- confirmed against
 * /opt/cray/pe/libsci/26.03.0/GNU/12/x86_64/include/lapacke_utils.h and
 * `nm -D .../libsci_gnu.so | grep ge_trans`.
 */
#ifndef HPC_KERNELBENCH_HICMA_X_LAPACKE_GE_TRANS_COMPAT_H
#define HPC_KERNELBENCH_HICMA_X_LAPACKE_GE_TRANS_COMPAT_H

#ifdef __cplusplus
extern "C" {
#endif

void LAPACKE_sge_trans(int matrix_layout, int m, int n,
                        const float *in, int ldin,
                        float *out, int ldout);
void LAPACKE_dge_trans(int matrix_layout, int m, int n,
                        const double *in, int ldin,
                        double *out, int ldout);
void LAPACKE_zge_trans(int matrix_layout, int m, int n,
                        const double _Complex *in, int ldin,
                        double _Complex *out, int ldout);

#ifdef __cplusplus
}
#endif

#endif /* HPC_KERNELBENCH_HICMA_X_LAPACKE_GE_TRANS_COMPAT_H */
