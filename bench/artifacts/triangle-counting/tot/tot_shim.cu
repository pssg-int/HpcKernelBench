// Thin extern "C" shim around ToT's own header-only pipeline (tot.h).
//
// NOT part of the artifact -- same role as bench/artifacts/spmv/sspmv/
// csr_shim.cpp: a minimal wrapper that instantiates the artifact's own
// templated functions with concrete types and exposes a ctypes-callable
// boundary, touching zero kernel code (ARTIFACT_GUIDE.md rule 3).
//
// Boundary chosen to match ToT's own driver (source/apps/tot.cu) exactly:
//   tot_prepare() = read CSR -> convert_csr_to_coo -> convert_undirected
//                   (symmetrize) -> extract_upper_triangular (orientation,
//                   row<col fixed order -- NOT this project's domain-level
//                   degree-ascending order, see adapter.py docstring) ->
//                   convert_coo2bmp (algorithm-native format). ALL of this
//                   is tot.cu's own preprocessing, run once, matching the
//                   spec's "orientation + format conversion NOT timed,
//                   reported once" requirement.
//   tot_count()   = tot::count_triangles_on_tensors(bmp, bmp, bmp) ONLY --
//                   the tensor-core masked-SpGEMM kernel itself, tot.cu's
//                   own "[Counting Triangles]" timed region. Since
//                   extract_upper_triangular was applied (equivalent to
//                   tot.cu's `-e 1`), the raw returned count is EXACT --
//                   no /6 division needed (see tot.cu: `count = config.extract
//                   ? count : count / 6;`).
#include "tot.h"

#include <cstdint>
#include <cstdio>

using CsrMat = tot::CsrMatrix<int, float, tot::device_memory>;
using CooMat = tot::CooMatrix<int, float, tot::device_memory>;
using BmpMat = tot::BitmapCOO<int, float, tot::bmp64_t, 4, tot::device_memory>;

struct TotHandle {
    BmpMat bmp;
};

extern "C" {

// n: number of vertices; nnz: number of directed entries in the input CSR
// (may be a directed or already-symmetric pattern -- convert_undirected
// below makes it symmetric+dedup+self-loop-free regardless, matching
// tot.cu's own unconditional call). row_ptr has n+1 entries, col_idx has
// nnz entries, both int32, host memory. Returns an opaque handle (caller
// must free with tot_free), or nullptr on failure (message on stderr).
void* tot_prepare(int n, long long nnz, const int* row_ptr, const int* col_idx) {
    try {
        CsrMat csr_d;
        csr_d.resize(n, n, (int)nnz);
        thrust::host_vector<int> h_rowptr(row_ptr, row_ptr + n + 1);
        thrust::host_vector<int> h_colidx(col_idx, col_idx + nnz);
        thrust::host_vector<float> h_vals((size_t)nnz, 1.0f);
        csr_d.row_pointers = h_rowptr;
        csr_d.column_indices = h_colidx;
        csr_d.values = h_vals;

        CooMat coo_d;
        tot::convert_csr_to_coo(coo_d, csr_d);
        csr_d.free();

        tot::convert_undirected(coo_d);       // symmetrize + dedup + drop self-loops
        tot::extract_upper_triangular(coo_d); // orientation: source < target

        TotHandle* h = new TotHandle();
        tot::convert_coo2bmp(coo_d, h->bmp);  // algorithm-native bitmap format
        coo_d.free();
        cudaDeviceSynchronize();
        return (void*)h;
    } catch (const std::exception& e) {
        fprintf(stderr, "tot_prepare failed: %s\n", e.what());
        return nullptr;
    }
}

// The kernel-only call: exact triangle count via tensor-core masked SpGEMM.
long long tot_count(void* handle) {
    TotHandle* h = (TotHandle*)handle;
    size_t count = tot::count_triangles_on_tensors(h->bmp, h->bmp, h->bmp);
    return (long long)count;
}

void tot_free(void* handle) {
    delete (TotHandle*)handle;
}

}  // extern "C"
