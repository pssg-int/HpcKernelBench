// Thin ctypes-callable bridge around TileSpMV's own tiling format
// construction (source/src/csr2tile.h: Tile_create) and its own GPU kernels
// (source/src/tilespmv_cuda.h: stir_spmv_cuda_kernel_v5/_v6).
//
// NOT part of the artifact. It exists because TileSpMV only ships a
// monolithic driver (main.cu -> call_tilespmv_cuda()) that loads a .mtx
// file itself and, inside ONE function, does format construction, 200
// warmup launches, 400 more "cache warm" launches, then BENCH_REPEAT=1000
// more launches wrapped in a single gettimeofday() pair -- exactly the
// "own end-to-end binary reading .mtx and timing internally" case the
// integration brief calls out. This file splits that into:
//
//   tilespmv_prepare()  -- called ONCE, timed as preprocessing by
//                           adapter.py. Does everything call_tilespmv_cuda()
//                           does BEFORE its first timed kernel launch:
//                             1. Tile_create()   (source/csr2tile.h, the
//                                paper's tiled-format construction,
//                                unmodified)
//                             2. tilespmv_cpu()   (source/tilespmv_cpu.h)
//                                -- ALSO unmodified; despite its name this
//                                does not just run a CPU SpMV, it derives
//                                ptroffset1/ptroffset2, rowblkblock and the
//                                blkcoostylerowidx* row-balance arrays the
//                                GPU kernels need. Its own CPU-side
//                                y/y_golden self-check is harmless busywork
//                                neutralized by passing the SAME buffer for
//                                both (see below) rather than touching its
//                                logic.
//                             3. every cudaMalloc/cudaMemcpy
//                                call_tilespmv_cuda() does for the tile
//                                matrix and x -- lifted verbatim from that
//                                function's H2D section.
//                             4. ONE untimed launch of
//                                stir_spmv_cuda_kernel_v5 -- see "why v5"
//                                below.
//
//   tilespmv_run()      -- called once per harness iteration. Re-zeros d_y
//                          and launches stir_spmv_cuda_kernel_v6 ONCE (the
//                          artifact's actual steady-state SpMV kernel) --
//                          see "why not CSR5" below for why nothing else is
//                          needed. Launched on the caller-supplied stream
//                          (0 = default/legacy stream, matching every other
//                          bridge in this repo and CudaEventTimer's
//                          recording stream).
//
// No kernel/format-conversion code is modified: every struct, malloc,
// memcpy and kernel launch below is copied unchanged from
// tilespmv_cuda.h's call_tilespmv_cuda(), just reorganized across
// prepare/run instead of being bundled with the artifact's own
// warmup+bench-loop+gettimeofday timing.
//
// -------------------------------------------------------------------------
// Why v5 (stir_spmv_cuda_kernel_v5) has to run once in prepare(), not run():
//
// The artifact's own driver calls v5 exactly ONCE (tilespmv_cuda.h:1045),
// before its WARMUP_NUM/BENCH_REPEAT loops of v6. v5 computes a full SpMV
// pass (its output y is thrown away -- the driver memsets d_y and re-runs
// v6 immediately afterward) but, as a SIDE EFFECT, populates three
// structural lookup tables consumed by every later v6 call:
// d_coodeferoffset, d_deferbuf_coooff, d_deferbuf_dxoff (tilespmv_cuda.h
// lines 146-149, 356-366 inside v5; read back at lines 69-104 inside v6).
// These tables store *offsets/indices* into the tile's own CSR/x arrays for
// a per-warp "prefetch schedule" -- derived purely from the tile structure
// (which nonzero groups a warp defers into shared memory), never from the
// numeric VALUE of x. Concretely: the values written are indices such as
// `s_ptroffset1_local[...] + lane_id` and `x_offset` (a column-tile base
// offset), not anything computed FROM d_x's contents. Confirmed by reading
// v6's own consumption of these tables: it uses them only to relocate
// WHERE to read d_x/d_Blockcsr_Val at kernel-launch time, always re-reading
// current device memory through them -- so the schedule stays valid for
// however many times v6 is later called with the SAME tile format, matching
// exactly how the artifact's own driver runs v5 once and v6 thousands of
// times against it. Running v5 once during prepare() and v6 alone during
// run() is therefore not a shortcut around the artifact's structure, it IS
// the artifact's own structure.
//
// -------------------------------------------------------------------------
// Why NOT CSR5 -- a genuine finding, corrected after direct measurement:
//
// TileSpMV's format selection (csr2tile.h) tags any 16x16 tile with
// nnztmp <= COO_NNZ_TH=12 nonzeros as Format=1 ("very sparse" -> in-tile
// COO), storing its entries in the tile's own Blockcoo_Val/coo_*Idx arrays
// -- handled by stir_spmv_cuda_kernel_v6's own `case 1` branch, exactly
// like every other per-tile format. Separately, source/src/external/
// CSR5_cuda/ vendors a full copy of Weifeng Liu's CSR5 SpMV, and
// csr2tile.h ALSO copies those exact same Format=1 entries (see
// convert_step3/convert_step4's `case 1` block: it writes into
// Blockcoo_Val/coo_*Idx AND, at the same offset_new = new_coocount[tile_id],
// into new_coo_value/new_coo_rowidx/new_coo_colidx, which convert_step
// later assembles into matrix->deferredcoo_{ptr,colidx,val}) -- a verbatim
// SECOND copy of the identical nonzeros. call_tilespmv_cuda()'s WARMUP_NUM
// loop and its first (untimed) BENCH_REPEAT loop call `A.spmv(alpha, d_y)`
// after v6 (tilespmv_cuda.h lines 1079-1081, 1104-1105), which -- since
// CSR5's kernel accumulates via `d_y[...] +=` -- ADDS this duplicate
// contribution on top of what v6 already computed via its own Format=1
// handling. An earlier version of this bridge assumed the artifact's
// SECOND, actually-measured loop (lines 1112-1137, where the equivalent
// call is commented out: "// if (coototal != 0...) // err = A.spmv(...)")
// was silently DROPPING necessary work, and added the CSR5 call to every
// run(). Direct measurement disproved that: on a synthetic matrix
// engineered so every nonzero lands in its own Format=1 tile (coototal ==
// nnz), the CSR5-including version returned y = 2*y_ref EXACTLY (verified
// element-by-element against a NumPy reference) -- i.e. genuine double
// counting, not a missing correction. The artifact's own measured/reported
// loop is the one that's actually correct: v6 alone already accounts for
// 100% of the matrix (confirmed separately: a coototal==0 case, e.g. this
// spec's "smoke-banded" workload, matches an fp64 NumPy reference to
// max_scaled_err ~1e-16 with v6 alone). matrix->deferredcoo_*/CSR5 is
// therefore dead weight for correctness in this artifact as published --
// this bridge does not build or call it at all. Reported in STATUS.md.

// CUDA-version compat: must precede tilespmv_cuda.h (-> CSR5_cuda headers,
// still #included transitively by tilespmv_cuda.h even though this bridge
// never instantiates/calls anonymouslibHandle -- see build.sh).
#include "shfl_compat.h"

// encode.h/format.h have no include guards and are already pulled in by
// csr2tile.h -- do not #include them a second time here (nvcc errors on
// the resulting redefinitions of Tile_matrix/encode/decode/...).
#include "common.h"
#include "utils.h"
#include "csr2tile.h"
#include "tilespmv_cpu.h"
#include "tilespmv_cuda.h"

#include <cstring>
#include <cstdlib>

struct TileSpmvHandle {
    Tile_matrix matrixA{};
    int rowA = 0, colA = 0;
    MAT_PTR_TYPE nnzA = 0;

    // host-side preprocessing derived by tilespmv_cpu()
    int *ptroffset1 = nullptr;
    int *ptroffset2 = nullptr;
    int rowblkblock = 0;
    unsigned int *blkcoostylerowidx = nullptr;
    int *blkcoostylerowidx_colstart = nullptr;
    int *blkcoostylerowidx_colstop = nullptr;
    MAT_VAL_TYPE *y_scratch = nullptr;  // aliased as both y/y_golden -- see file docstring

    // device buffers -- names match call_tilespmv_cuda()'s d_* locals 1:1
    // (minus the deferredcoo_*/CSR5 buffers, which this bridge never
    // builds -- see "why NOT CSR5" above).
    MAT_PTR_TYPE *d_tile_ptr = nullptr;
    int *d_tile_columnidx = nullptr;
    char *d_Format = nullptr;
    int *d_blknnz = nullptr;
    unsigned char *d_blknnznnz = nullptr;
    unsigned char *d_csr_compressedIdx = nullptr;
    MAT_VAL_TYPE *d_Blockcsr_Val = nullptr;
    unsigned char *d_Blockcsr_Ptr = nullptr;
    unsigned char *d_coo_compressed_Idx = nullptr;
    MAT_VAL_TYPE *d_Blockcoo_Val = nullptr;
    unsigned char *d_ell_compressedIdx = nullptr;
    MAT_VAL_TYPE *d_Blockell_Val = nullptr;
    unsigned char *d_hybIdx = nullptr;
    char *d_tilewidth = nullptr;
    MAT_VAL_TYPE *d_Blockhyb_Val = nullptr;
    MAT_VAL_TYPE *d_Blockdense_Val = nullptr;
    int *d_dnsrowptr = nullptr;
    MAT_VAL_TYPE *d_Blockdenserow_Val = nullptr;
    char *d_denserowid = nullptr;
    int *d_dnscolptr = nullptr;
    MAT_VAL_TYPE *d_Blockdensecol_Val = nullptr;
    char *d_densecolid = nullptr;
    unsigned int *d_blkcoostylerowidx = nullptr;
    int *d_blkcoostylerowidx_colstart = nullptr;
    int *d_blkcoostylerowidx_colstop = nullptr;
    int *d_ptroffset1 = nullptr;
    int *d_ptroffset2 = nullptr;
    MAT_VAL_TYPE *d_x = nullptr;
    MAT_VAL_TYPE *d_y = nullptr;
    int *d_coodeferoffset = nullptr;
    int *d_deferbuf_coooff = nullptr;
    int *d_deferbuf_dxoff = nullptr;

    int num_blocks = 0;
    int num_threads = 0;
};

extern "C" {

// Builds the tile format (Tile_create, csr2tile.h -- unmodified), the
// row-balance bookkeeping (tilespmv_cpu, tilespmv_cpu.h -- unmodified),
// every device buffer call_tilespmv_cuda() would allocate/H2D-copy for the
// tile matrix and x, and runs the ONE untimed v5 launch that builds v6's
// per-warp prefetch schedule. All of this is the artifact's own
// preprocessing -- called once, timed once, by adapter.py.
void *tilespmv_prepare(int rowA, int colA, int nnzA,
                        int *csrRowPtrA, int *csrColIdxA, double *csrValA,
                        double *x_in) {
    TileSpmvHandle *h = new TileSpmvHandle();
    h->rowA = rowA;
    h->colA = colA;
    h->nnzA = nnzA;

    // 1. tiled format construction -- Tile_create only reads csrRowPtrA/
    //    csrColIdxA/csrValA (verified: no assignment into any of the three
    //    input arrays anywhere in csr2tile.h), so the caller's buffers can
    //    be passed straight through.
    Tile_create(&h->matrixA, rowA, colA, (MAT_PTR_TYPE)nnzA,
                csrRowPtrA, csrColIdxA, csrValA);

    int tilenum = h->matrixA.tilenum;
    h->ptroffset1 = (int *)calloc(tilenum, sizeof(int));
    h->ptroffset2 = (int *)calloc(tilenum, sizeof(int));
    h->y_scratch = (MAT_VAL_TYPE *)malloc(sizeof(MAT_VAL_TYPE) * rowA);

    // 2. row-balance bookkeeping. y_golden == y_scratch (same buffer) so
    //    the function's internal self-check trivially reads back what it
    //    just wrote (errcount always 0) -- neutralizes busywork we don't
    //    need without touching tilespmv_cpu.h's logic; the real
    //    correctness gate is the harness's own, against reference_spmv().
    tilespmv_cpu(&h->matrixA, h->ptroffset1, h->ptroffset2,
                 &h->rowblkblock, &h->blkcoostylerowidx,
                 &h->blkcoostylerowidx_colstart, &h->blkcoostylerowidx_colstop,
                 rowA, colA, (MAT_PTR_TYPE)nnzA,
                 csrRowPtrA, csrColIdxA, csrValA,
                 x_in, h->y_scratch, h->y_scratch);

    Tile_matrix *m = &h->matrixA;
    int csr_csize = m->csrsize % 2 == 0 ? m->csrsize / 2 : m->csrsize / 2 + 1;
    int ell_csize = m->ellsize % 2 == 0 ? m->ellsize / 2 : m->ellsize / 2 + 1;
    int hyb_size = m->hybellsize % 2 == 0 ? m->hybellsize / 2 : m->hybellsize / 2 + 1;

    // 3. device buffers -- verbatim from call_tilespmv_cuda()'s H2D section
    //    (minus deferredcoo_*/CSR5, never built -- see file docstring).
    cudaMalloc((void **)&h->d_tile_ptr, (m->tilem + 1) * sizeof(MAT_PTR_TYPE));
    cudaMalloc((void **)&h->d_tile_columnidx, tilenum * sizeof(int));
    cudaMalloc((void **)&h->d_Format, tilenum * sizeof(char));
    cudaMalloc((void **)&h->d_blknnz, (tilenum + 1) * sizeof(int));
    cudaMalloc((void **)&h->d_blknnznnz, (tilenum + 1) * sizeof(unsigned char));
    cudaMemcpy(h->d_tile_ptr, m->tile_ptr, (m->tilem + 1) * sizeof(MAT_PTR_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(h->d_tile_columnidx, m->tile_columnidx, tilenum * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(h->d_Format, m->Format, tilenum * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(h->d_blknnz, m->blknnz, (tilenum + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(h->d_blknnznnz, m->blknnznnz, (tilenum + 1) * sizeof(unsigned char), cudaMemcpyHostToDevice);

    cudaMalloc((void **)&h->d_csr_compressedIdx, csr_csize * sizeof(unsigned char));
    cudaMalloc((void **)&h->d_Blockcsr_Val, m->csrsize * sizeof(MAT_VAL_TYPE));
    cudaMalloc((void **)&h->d_Blockcsr_Ptr, m->csrptrlen * sizeof(unsigned char));
    cudaMemcpy(h->d_csr_compressedIdx, m->csr_compressedIdx, csr_csize * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(h->d_Blockcsr_Val, m->Blockcsr_Val, m->csrsize * sizeof(MAT_VAL_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(h->d_Blockcsr_Ptr, m->Blockcsr_Ptr, m->csrptrlen * sizeof(unsigned char), cudaMemcpyHostToDevice);

    cudaMalloc((void **)&h->d_coo_compressed_Idx, m->coosize * sizeof(unsigned char));
    cudaMalloc((void **)&h->d_Blockcoo_Val, m->coosize * sizeof(MAT_VAL_TYPE));
    cudaMemcpy(h->d_coo_compressed_Idx, m->coo_compressed_Idx, m->coosize * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(h->d_Blockcoo_Val, m->Blockcoo_Val, m->coosize * sizeof(MAT_VAL_TYPE), cudaMemcpyHostToDevice);

    cudaMalloc((void **)&h->d_ell_compressedIdx, ell_csize * sizeof(unsigned char));
    cudaMalloc((void **)&h->d_Blockell_Val, m->ellsize * sizeof(MAT_VAL_TYPE));
    cudaMemcpy(h->d_ell_compressedIdx, m->ell_compressedIdx, ell_csize * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(h->d_Blockell_Val, m->Blockell_Val, m->ellsize * sizeof(MAT_VAL_TYPE), cudaMemcpyHostToDevice);

    cudaMalloc((void **)&h->d_hybIdx, (hyb_size + m->hybcoosize) * sizeof(unsigned char));
    cudaMalloc((void **)&h->d_tilewidth, tilenum * sizeof(char));
    cudaMalloc((void **)&h->d_Blockhyb_Val, (m->hybellsize + m->hybcoosize) * sizeof(MAT_VAL_TYPE));
    cudaMemcpy(h->d_hybIdx, m->hybIdx, (hyb_size + m->hybcoosize) * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(h->d_tilewidth, m->tilewidth, tilenum * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(h->d_Blockhyb_Val, m->Blockhyb_Val, (m->hybellsize + m->hybcoosize) * sizeof(MAT_VAL_TYPE), cudaMemcpyHostToDevice);

    cudaMalloc((void **)&h->d_Blockdense_Val, m->dnssize * sizeof(MAT_VAL_TYPE));
    cudaMemcpy(h->d_Blockdense_Val, m->Blockdense_Val, m->dnssize * sizeof(MAT_VAL_TYPE), cudaMemcpyHostToDevice);

    cudaMalloc((void **)&h->d_dnsrowptr, (tilenum + 1) * sizeof(int));
    cudaMalloc((void **)&h->d_Blockdenserow_Val, m->dnsrowsize * sizeof(MAT_VAL_TYPE));
    cudaMalloc((void **)&h->d_denserowid, m->dnsrowptr[tilenum] * sizeof(char));
    cudaMemcpy(h->d_dnsrowptr, m->dnsrowptr, (tilenum + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(h->d_Blockdenserow_Val, m->Blockdenserow_Val, m->dnsrowsize * sizeof(MAT_VAL_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(h->d_denserowid, m->denserowid, m->dnsrowptr[tilenum] * sizeof(char), cudaMemcpyHostToDevice);

    cudaMalloc((void **)&h->d_dnscolptr, (tilenum + 1) * sizeof(int));
    cudaMalloc((void **)&h->d_Blockdensecol_Val, m->dnscolsize * sizeof(MAT_VAL_TYPE));
    cudaMalloc((void **)&h->d_densecolid, m->dnscolptr[tilenum] * sizeof(char));
    cudaMemcpy(h->d_dnscolptr, m->dnscolptr, (tilenum + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(h->d_Blockdensecol_Val, m->Blockdensecol_Val, m->dnscolsize * sizeof(MAT_VAL_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(h->d_densecolid, m->densecolid, m->dnscolptr[tilenum] * sizeof(char), cudaMemcpyHostToDevice);

    cudaMalloc((void **)&h->d_blkcoostylerowidx, h->rowblkblock * sizeof(unsigned int));
    cudaMalloc((void **)&h->d_blkcoostylerowidx_colstart, h->rowblkblock * sizeof(int));
    cudaMalloc((void **)&h->d_blkcoostylerowidx_colstop, h->rowblkblock * sizeof(int));
    cudaMemcpy(h->d_blkcoostylerowidx, h->blkcoostylerowidx, h->rowblkblock * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(h->d_blkcoostylerowidx_colstart, h->blkcoostylerowidx_colstart, h->rowblkblock * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(h->d_blkcoostylerowidx_colstop, h->blkcoostylerowidx_colstop, h->rowblkblock * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc((void **)&h->d_ptroffset1, tilenum * sizeof(int));
    cudaMalloc((void **)&h->d_ptroffset2, tilenum * sizeof(int));
    cudaMemcpy(h->d_ptroffset1, h->ptroffset1, tilenum * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(h->d_ptroffset2, h->ptroffset2, tilenum * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc((void **)&h->d_x, colA * sizeof(MAT_VAL_TYPE));
    cudaMalloc((void **)&h->d_y, rowA * sizeof(MAT_VAL_TYPE));
    cudaMemcpy(h->d_x, x_in, colA * sizeof(MAT_VAL_TYPE), cudaMemcpyHostToDevice);

    // 4. one untimed v5 launch -- builds the per-warp prefetch schedule
    //    (d_coodeferoffset/d_deferbuf_coooff/d_deferbuf_dxoff) that every
    //    v6 call in run() will read; see "why v5" in the file docstring.
    cudaMalloc((void **)&h->d_coodeferoffset, h->rowblkblock * sizeof(int));
    cudaMemset(h->d_coodeferoffset, 0, h->rowblkblock * sizeof(int));
    cudaMalloc((void **)&h->d_deferbuf_coooff, (size_t)h->rowblkblock * PREFETCH_SMEM_TH * COO_NNZ_TH * sizeof(int));
    cudaMemset(h->d_deferbuf_coooff, 0, (size_t)h->rowblkblock * PREFETCH_SMEM_TH * COO_NNZ_TH * sizeof(int));
    cudaMalloc((void **)&h->d_deferbuf_dxoff, (size_t)h->rowblkblock * PREFETCH_SMEM_TH * COO_NNZ_TH * sizeof(int));
    cudaMemset(h->d_deferbuf_dxoff, 0, (size_t)h->rowblkblock * PREFETCH_SMEM_TH * COO_NNZ_TH * sizeof(int));

    h->num_threads = WARP_PER_BLOCK * WARP_SIZE;
    h->num_blocks = (int)ceil((double)h->rowblkblock / (double)(h->num_threads / WARP_SIZE));

    stir_spmv_cuda_kernel_v5<<<h->num_blocks, h->num_threads>>>(
        m->tilem, m->tilen, rowA, colA,
        h->d_tile_ptr, h->d_tile_columnidx, h->d_Format, h->d_blknnz, h->d_blknnznnz,
        h->d_csr_compressedIdx, h->d_Blockcsr_Val, h->d_Blockcsr_Ptr,
        h->d_coo_compressed_Idx, h->d_Blockcoo_Val,
        h->d_tilewidth, h->d_ell_compressedIdx, h->d_Blockell_Val,
        h->d_hybIdx, h->d_Blockhyb_Val,
        h->d_Blockdense_Val,
        h->d_dnsrowptr, h->d_Blockdenserow_Val, h->d_denserowid,
        h->d_dnscolptr, h->d_Blockdensecol_Val, h->d_densecolid,
        h->d_ptroffset1, h->d_ptroffset2,
        h->rowblkblock, h->d_blkcoostylerowidx, h->d_blkcoostylerowidx_colstart, h->d_blkcoostylerowidx_colstop,
        h->d_x, h->d_y, 7, h->d_coodeferoffset, h->d_deferbuf_coooff, h->d_deferbuf_dxoff);
    cudaDeviceSynchronize();

    return h;
}

// ONE y=Ax call: re-zero d_y, launch the artifact's own steady-state kernel
// (stir_spmv_cuda_kernel_v6) exactly once -- v6 alone accounts for every
// nonzero (see "why NOT CSR5" in the file docstring). Launched on `stream`
// (pass 0 / null for the default stream, same convention as every other
// bridge in this repo).
void tilespmv_run(void *handle, cudaStream_t stream) {
    TileSpmvHandle *h = reinterpret_cast<TileSpmvHandle *>(handle);
    Tile_matrix *m = &h->matrixA;

    cudaMemsetAsync(h->d_y, 0, h->rowA * sizeof(MAT_VAL_TYPE), stream);

    stir_spmv_cuda_kernel_v6<<<h->num_blocks, h->num_threads, 0, stream>>>(
        m->tilem, m->tilen, h->rowA, h->colA, h->nnzA,
        h->d_tile_ptr, h->d_tile_columnidx, h->d_Format, h->d_blknnz, h->d_blknnznnz,
        h->d_csr_compressedIdx, h->d_Blockcsr_Val, h->d_Blockcsr_Ptr,
        h->d_coo_compressed_Idx, h->d_Blockcoo_Val,
        h->d_tilewidth, h->d_ell_compressedIdx, h->d_Blockell_Val,
        h->d_hybIdx, h->d_Blockhyb_Val,
        h->d_Blockdense_Val,
        h->d_dnsrowptr, h->d_Blockdenserow_Val, h->d_denserowid,
        h->d_dnscolptr, h->d_Blockdensecol_Val, h->d_densecolid,
        h->d_ptroffset1, h->d_ptroffset2,
        h->rowblkblock, h->d_blkcoostylerowidx, h->d_blkcoostylerowidx_colstart, h->d_blkcoostylerowidx_colstop,
        h->d_x, h->d_y, 7, h->d_coodeferoffset, h->d_deferbuf_coooff, h->d_deferbuf_dxoff);
}

// D2H copy of y (rowA elements); a plain cudaMemcpy on the default stream
// blocks until any prior kernel on that stream completes, so this also
// serves as the sync point between run() and the correctness/gate check.
void tilespmv_copy_y(void *handle, double *host_y_out) {
    TileSpmvHandle *h = reinterpret_cast<TileSpmvHandle *>(handle);
    cudaMemcpy(host_y_out, h->d_y, h->rowA * sizeof(MAT_VAL_TYPE), cudaMemcpyDeviceToHost);
}

void tilespmv_free(void *handle) {
    TileSpmvHandle *h = reinterpret_cast<TileSpmvHandle *>(handle);
    if (!h) return;

    Tile_destroy(&h->matrixA);
    free(h->ptroffset1);
    free(h->ptroffset2);
    free(h->blkcoostylerowidx);
    free(h->blkcoostylerowidx_colstart);
    free(h->blkcoostylerowidx_colstop);
    free(h->y_scratch);

    cudaFree(h->d_tile_ptr);
    cudaFree(h->d_tile_columnidx);
    cudaFree(h->d_Format);
    cudaFree(h->d_blknnz);
    cudaFree(h->d_blknnznnz);
    cudaFree(h->d_csr_compressedIdx);
    cudaFree(h->d_Blockcsr_Val);
    cudaFree(h->d_Blockcsr_Ptr);
    cudaFree(h->d_coo_compressed_Idx);
    cudaFree(h->d_Blockcoo_Val);
    cudaFree(h->d_ell_compressedIdx);
    cudaFree(h->d_Blockell_Val);
    cudaFree(h->d_hybIdx);
    cudaFree(h->d_tilewidth);
    cudaFree(h->d_Blockhyb_Val);
    cudaFree(h->d_Blockdense_Val);
    cudaFree(h->d_dnsrowptr);
    cudaFree(h->d_Blockdenserow_Val);
    cudaFree(h->d_denserowid);
    cudaFree(h->d_dnscolptr);
    cudaFree(h->d_Blockdensecol_Val);
    cudaFree(h->d_densecolid);
    cudaFree(h->d_blkcoostylerowidx);
    cudaFree(h->d_blkcoostylerowidx_colstart);
    cudaFree(h->d_blkcoostylerowidx_colstop);
    cudaFree(h->d_ptroffset1);
    cudaFree(h->d_ptroffset2);
    cudaFree(h->d_x);
    cudaFree(h->d_y);
    cudaFree(h->d_coodeferoffset);
    cudaFree(h->d_deferbuf_coooff);
    cudaFree(h->d_deferbuf_dxoff);

    delete h;
}

}  // extern "C"
