// gunrock_shim.cu -- NOT part of the artifact. Exposes Gunrock (essentials
// v2.1.0)'s single-source BFS through a small extern "C" API so the harness
// can call it in-process with ctypes, splitting along the same lines as the
// blest/efg shims:
//
//   gr_prepare()      CSR (host) -> Gunrock device CSR + graph view, plus the
//                     distance buffer and device context. Preprocessing; done
//                     once per graph.
//   gr_run()          gunrock::bfs::run(G, source, distances, ...) only, then a
//                     context synchronize (Gunrock works on its own non-blocking
//                     stream, so the caller's CUDA events would not wait for it).
//                     *internal_ms = the value bfs::run returns: Gunrock's own
//                     enactor timer around the BFS loop (what Gunrock's example
//                     driver prints as "GPU Elapsed Time").
//   gr_copy_levels()  device distances -> host (untimed; only the gate needs it)
//   gr_free()
//
// Types: vertex_t = edge_t = int (Gunrock's example BFS driver uses the same),
// so graphs need nnz < 2^31; the adapter rejects larger ones before calling.
// Unreached vertices keep Gunrock's sentinel, std::numeric_limits<int>::max().
#include <gunrock/algorithms/bfs.hxx>

#include <cstdio>
#include <exception>
#include <memory>
#include <utility>

using namespace gunrock;
using namespace memory;

namespace {
using vertex_t = int;
using edge_t = int;
using weight_t = float;
using csr_t = format::csr_t<memory_space_t::device, vertex_t, edge_t, weight_t>;
using graph_type = decltype(graph::build<memory_space_t::device>(
    std::declval<graph::graph_properties_t>(), std::declval<csr_t&>()));

struct Handle {
  csr_t csr;  // owns the device arrays the graph view points into
  graph_type G;
  thrust::device_vector<vertex_t> distances;
  thrust::device_vector<vertex_t> predecessors;
  std::shared_ptr<gcuda::multi_context_t> context;
  vertex_t n = 0;
};
}  // namespace

extern "C" {

void* gr_prepare(int n, long long nnz, const int* row_ptr, const int* col_idx,
                 int directed) {
  try {
    auto* h = new Handle();
    h->n = n;
    h->csr.number_of_rows = n;
    h->csr.number_of_columns = n;
    h->csr.number_of_nonzeros = static_cast<edge_t>(nnz);
    h->csr.row_offsets.resize(n + 1);
    h->csr.column_indices.resize(nnz);
    h->csr.nonzero_values.resize(nnz);
    thrust::copy(row_ptr, row_ptr + n + 1, h->csr.row_offsets.begin());
    thrust::copy(col_idx, col_idx + nnz, h->csr.column_indices.begin());
    thrust::fill(h->csr.nonzero_values.begin(), h->csr.nonzero_values.end(), 1.0f);

    graph::graph_properties_t props;
    props.directed = directed != 0;
    props.weighted = false;
    props.symmetric = directed == 0;
    h->G = graph::build<memory_space_t::device>(props, h->csr);

    h->distances.resize(n);
    h->predecessors.resize(n);
    h->context = std::make_shared<gcuda::multi_context_t>(0);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
      std::fprintf(stderr, "gr_prepare: %s\n", cudaGetErrorString(err));
      delete h;
      return nullptr;
    }
    return h;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "gr_prepare: %s\n", e.what());
    return nullptr;
  }
}

int gr_run(void* handle, int source, float* internal_ms) {
  try {
    auto* h = static_cast<Handle*>(handle);
    vertex_t src = source;
    float ms = gunrock::bfs::run(h->G, src, h->distances.data().get(),
                                 h->predecessors.data().get(), h->context);
    h->context->get_context(0)->synchronize();
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
      std::fprintf(stderr, "gr_run: %s\n", cudaGetErrorString(err));
      return 1;
    }
    if (internal_ms) *internal_ms = ms;
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "gr_run: %s\n", e.what());
    return 1;
  }
}

int gr_copy_levels(void* handle, int* out) {
  auto* h = static_cast<Handle*>(handle);
  cudaError_t err = cudaMemcpy(out, h->distances.data().get(),
                               sizeof(vertex_t) * h->n, cudaMemcpyDeviceToHost);
  return err == cudaSuccess ? 0 : 1;
}

void gr_free(void* handle) { delete static_cast<Handle*>(handle); }

}  // extern "C"
