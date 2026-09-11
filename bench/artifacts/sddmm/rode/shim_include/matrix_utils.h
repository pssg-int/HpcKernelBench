// Compile-time stand-in for source/utils/matrix_utils.h, used ONLY when
// compiling source/RoDe_SDDMM/RoDeSddmm.cu.
//
// Why this exists (ARTIFACT_GUIDE.md rule 3: build-system fix, not a kernel
// patch -- no file inside source/ is touched):
//
// RoDeSddmm.cu has `#include "matrix_utils.h"` at file scope. The real
// header (source/utils/matrix_utils.h) pulls in "absl/random/random.h"
// (abseil-cpp), which is an uninitialized git submodule in this clone
// (source/third_party/abseil-cpp/ is empty) and unavailable as a NERSC
// module (confirmed via `module spider abseil`, "unable to find" -- see
// bench/artifacts/spmm/rode/STATUS.md, which hit the identical dependency
// for RoDe_SpMM's SparseMatrix class and made the same call).
//
// Verified by direct grep of RoDeSddmm.cu: it contains ZERO references to
// any symbol matrix_utils.h declares (SPC::SparseMatrix, SPC::CudaMatrix,
// SPC::CudaSparseMatrix, absl::BitGen, MakeSparseMatrixRandomUniform, ...).
// The include is dead code -- RoDeSDDMM_n32/RoDeSDDMM_n128 and the
// SDDMMKernel4Block/SDDMMKernel4Residue __global__ kernels they launch take
// only raw pointers/ints (see RoDeSddmm.h) and use only Load/Store/Barrier/
// Value2Index/TypeUtils from common_utils.h (the OTHER, real header this
// file leaves untouched -- included and used normally).
//
// This header is placed on the include path BEFORE source/utils/ in
// build.sh's -I order, so the quoted `#include "matrix_utils.h"` resolves
// here instead of to the real file -- an include-path substitution, not a
// change to any tracked file, no source.patch needed. Its content is
// intentionally empty: nothing in RoDeSddmm.cu's actual compiled kernel
// code depends on it.
#ifndef MATRIX_UTILS_H_
#define MATRIX_UTILS_H_
#endif
