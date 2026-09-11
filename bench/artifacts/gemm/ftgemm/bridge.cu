// bridge.cu (HPC-KernelBench, NOT part of the FT-GEMM artifact).
//
// #includes the artifact's own kernel headers verbatim (see build.sh) and
// exposes a plain extern "C" launcher for exactly ONE kernel pair: the
// "large" tile config (ms=ns=64, ks=8), fault-tolerance ON
// (`ft_sgemm_large`, source/kernel/ft_sgemm/include_code_gen/
// ft_sgemm_large.cuh -- the paper's own headline contribution: online
// Algorithm-Based Fault Tolerance fused directly into the SGEMM main loop,
// self-contained -- no external checksum-vector arguments, the ABFT
// row/column checksums are computed on-the-fly in shared memory inside the
// kernel itself, confirmed by reading the file and by how source/kernel/
// ft_sgemm/sgemm.cu's own main() calls it with the SAME 8-argument
// signature as the plain kernel) and the plain fault-tolerance-OFF
// counterpart (`sgemm_large`, .../include_code_gen/sgemm_large.cuh) for
// disclosure/comparison (see STATUS.md; only the FT-on one is registered
// as this directory's IMPL_NAME, per the task's "disclose which" framing
// -- the FT-on kernel is the paper's actual point).
//
// "large" was chosen (over small/medium/tall/wide/huge) because its tile
// requirement (M,N multiple of 64; K multiple of 8) is the loosest of the
// five configs and is satisfied by ALL THREE of this domain's --smoke gemm
// shapes (256x256x256, 384x256x512, and even the 64x64x64-per-batch-item
// smoke-gemm-batched-b4-64 case) -- broader gate coverage than any other
// tile config, not an arbitrary pick.
//
// No kernel arithmetic is touched: this file only #includes the artifact's
// own .cuh files (unmodified) and adds a NEW host-side extern "C" wrapper
// that computes the SAME grid/block dims source/kernel/ft_sgemm/sgemm.cu's
// own main() uses for kernel_number==3 (plain) / 13 (FT-on), then launches
// the kernel exactly once.
//
// LAYOUT (verified empirically before being wired into adapter.py -- see
// STATUS.md's probe transcript): reading the kernel body's own comments
// and stride arithmetic (`A += ks * M` per K-step -> A is COLUMN-MAJOR
// (M,K), i.e. bit-identical to a row-major (K,M) array's flat memory;
// `B += ks * N` per K-step -> B is ROW-MAJOR (K,N), i.e. bit-identical to
// this domain's own row-major (K,N) numpy B operand, no transform needed;
// `C += (by*ns+...) * M` -> C is written COLUMN-MAJOR (M,N)). adapter.py
// therefore feeds `A.T` (C-contiguous) as the "A" argument, `B` as-is (C-
// contiguous) as the "B" argument, and reads the raw output back as a
// (N,M) row-major buffer, transposing once more to get a plain (M,N)
// row-major result -- exactly the pattern verified against dense.py's
// reference_gemm on a small random shape (see STATUS.md).
//
// PRECISION: `float` throughout (SGEMM) -- PRECISIONS = ["fp32"].
//
// SHAPE ALIGNMENT CONSTRAINT (inherited, not introduced): as with every
// other hand-tiled kernel in this benchmark (turbofno's cgemm, moonpoly's
// CUTLASS dispatch), `sgemm_large`/`ft_sgemm_large` have no tail/boundary
// guard for M/N not a multiple of 64 or K not a multiple of 8 -- the
// float4-vectorized loads would read/write out of bounds. adapter.py
// raises NotImplementedError otherwise.

#include "source/kernel/ft_sgemm/include_code_gen/sgemm_large.cuh"
#include "source/kernel/ft_sgemm/include_code_gen/ft_sgemm_large.cuh"

extern "C" {

// One plain (fault-tolerance OFF) SGEMM call. A: (K,M) row-major host
// buffer (== (M,K) col-major, see file docstring), B: (K,N) row-major host
// buffer, C: (N,M) row-major OUTPUT host buffer (== (M,N) col-major).
// Device alloc/H2D/launch/D2H all happen here (one call = one full kernel
// invocation, matching every sibling gemm adapter's per-run() device
// round-trip discipline; see adapter.py).
void ftgemm_sgemm_large_run(const float *A_colmajor, const float *B_rowmajor,
                             float *C_colmajor, int M, int N, int K,
                             float alpha, float beta) {
    float *dA, *dB, *dC;
    cudaMalloc(&dA, (size_t)M * K * sizeof(float));
    cudaMalloc(&dB, (size_t)K * N * sizeof(float));
    cudaMalloc(&dC, (size_t)M * N * sizeof(float));
    cudaMemcpy(dA, A_colmajor, (size_t)M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B_rowmajor, (size_t)K * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dC, C_colmajor, (size_t)M * N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockDim(64);
    dim3 gridDim((M + 63) / 64, (N + 63) / 64);
    sgemm_large<<<gridDim, blockDim>>>(M, N, K, dA, dB, dC, alpha, beta);
    cudaDeviceSynchronize();

    cudaMemcpy(C_colmajor, dC, (size_t)M * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
}

// Same contract, wrapping the fused-ABFT (fault-tolerance ON) kernel --
// the paper's own headline contribution.
void ftgemm_ft_sgemm_large_run(const float *A_colmajor, const float *B_rowmajor,
                                float *C_colmajor, int M, int N, int K,
                                float alpha, float beta) {
    float *dA, *dB, *dC;
    cudaMalloc(&dA, (size_t)M * K * sizeof(float));
    cudaMalloc(&dB, (size_t)K * N * sizeof(float));
    cudaMalloc(&dC, (size_t)M * N * sizeof(float));
    cudaMemcpy(dA, A_colmajor, (size_t)M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B_rowmajor, (size_t)K * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dC, C_colmajor, (size_t)M * N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockDim(64);
    dim3 gridDim((M + 63) / 64, (N + 63) / 64);
    ft_sgemm_large<<<gridDim, blockDim>>>(M, N, K, dA, dB, dC, alpha, beta);
    cudaDeviceSynchronize();

    cudaMemcpy(C_colmajor, dC, (size_t)M * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
}

}  // extern "C"
