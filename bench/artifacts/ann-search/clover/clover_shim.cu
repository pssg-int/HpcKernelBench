// clover_shim.cu -- NOT part of the artifact (bench/artifacts/ann-search/clover/,
// this directory only). Links directly against CLOVER's own header-only
// "hubs" method (include/bitonic-hubs.cuh) and exposes 3 extern "C" entry
// points that split EXACTLY along bitonic_hubs::C_and_Q's own two phases:
//
//   clover_prepare() -- hub selection (Randomly_Select_Hubs) -> per-hub
//                        distance computation (Calculate_Distances) ->
//                        lower-bound matrix construction (Construct_D) ->
//                        bucket sort by assigned hub (BucketSort) -> sorted
//                        lower-bound matrix (fused_transform_sort_D). This
//                        IS CLOVER's own spatio-graph INDEX (the hub graph +
//                        lower-bound matrix, this paper's actual
//                        contribution) -- all of it, verbatim from
//                        bitonic_hubs::C_and_Q (include/bitonic-hubs.cuh,
//                        lines ~552-620), just split out of that one
//                        function into this one. Zero lines of source/ were
//                        modified; this file only #includes it.
//   clover_query()   -- bitonic_hubs::Query<ROUNDS><<<>>> ONLY: the k-NN
//                        search kernel, dispatched on ROUNDS=ceil(k/32)
//                        exactly as C_and_Q's own switch statement does.
//   clover_free()    -- releases every device buffer prepare() allocated.
//
// Finding worth recording (see STATUS.md): bitonic_hubs::Query's first
// parameter `Qps` (intended as a caller-supplied query-point-index array,
// per C_and_Q's own signature `C_and_Q(n, data, q, queries, k, ...)`) is
// NEVER dereferenced anywhere in the kernel body -- confirmed by grepping
// bitonic-hubs.cuh for every occurrence of the identifier `Qps`. The kernel
// instead derives which point is being queried purely from
// `arr_idx[blockIdx.x * queries_per_block + threadIdx.y]` and is launched
// with a grid sized by `n` (Points_num), not by the caller's `q`. In other
// words: bitonic_hubs::C_and_Q's own public "hubs" method ALWAYS computes
// k-NN for every one of the n base points against itself (the paper's own
// "all-points-as-queries" convention, survey.md's own finding) and has NO
// working code path for a caller-chosen, held-out query subset -- `q` and
// `queries` are accepted parameters that do nothing. clover_query() below
// reproduces this faithfully (it does not silently "fix" it): it always
// returns n*k results; the adapter (adapter.py) slices out the rows for
// whichever query subset the workload actually asked for.

#include "bitonic-hubs.cuh"

using namespace bitonic_hubs;

extern "C" {

struct CloverHandle {
    unsigned int n;
    float *d_data;
    idx_t *dH;
    idx_t *dH_psum;
    idx_t *dH_assignments;
    float *arr_x, *arr_y, *arr_z;
    idx_t *arr_idx;
    idx_t *iD;
    float *dD;
    int *d_hubsScanned, *d_pointsScanned;
};

// n: point count. h_data: host array of n*3 float32 (row-major x,y,z).
// Builds CLOVER's hub graph + lower-bound matrix -- this IS the artifact's
// index construction; the harness times this call as preprocessing.
void *clover_prepare(unsigned int n, const float *h_data) {
    auto *h = new CloverHandle();
    h->n = n;

    idx_t constexpr block_size = 1024;

    CUDA_CALL(cudaMalloc((void **)&h->d_data, sizeof(float) * dim * n));
    CUDA_CALL(cudaMemcpy(h->d_data, h_data, sizeof(float) * dim * n, cudaMemcpyHostToDevice));

    idx_t *dH_psum_copy, *d_psum_placeholder;
    CUDA_CALL(cudaMalloc((void **)&h->dH, sizeof(idx_t) * H));
    CUDA_CALL(cudaMalloc((void **)&h->dH_psum, sizeof(idx_t) * (H + 1)));
    CUDA_CALL(cudaMalloc((void **)&dH_psum_copy, sizeof(idx_t) * (H + 1)));
    CUDA_CALL(cudaMalloc((void **)&d_psum_placeholder, sizeof(idx_t) * (H + 1)));
    CUDA_CALL(cudaMalloc((void **)&h->dH_assignments, sizeof(idx_t) * n));
    cudaMemset(h->dH_psum, 0, sizeof(idx_t) * (1 + H));
    cudaMemset(dH_psum_copy, 0, sizeof(idx_t) * (1 + H));
    cudaMemset(d_psum_placeholder, 0, sizeof(idx_t) * (1 + H));
    cudaMemset(h->dH_assignments, 0, sizeof(idx_t) * n);

    float *distances;
    idx_t constexpr batch_size = 10000;
    idx_t batch_number = (n + batch_size - 1) / batch_size;
    CUDA_CALL(cudaMalloc((void **)&distances, sizeof(float) * H * batch_size));

    CUDA_CALL(cudaMalloc((void **)&h->arr_x, sizeof(float) * n));
    CUDA_CALL(cudaMalloc((void **)&h->arr_y, sizeof(float) * n));
    CUDA_CALL(cudaMalloc((void **)&h->arr_z, sizeof(float) * n));
    CUDA_CALL(cudaMalloc((void **)&h->arr_idx, sizeof(idx_t) * n));

    float *D;
    CUDA_CALL(cudaMalloc((void **)&D, sizeof(float) * H * H));
    CUDA_CALL(cudaMalloc((void **)&h->iD, sizeof(idx_t) * H * H));
    CUDA_CALL(cudaMalloc((void **)&h->dD, sizeof(float) * H * H));

    std::size_t num_blocks = (H + block_size - 1) / block_size;
    Randomly_Select_Hubs<<<num_blocks, block_size>>>(n, h->dH);
    CHECK_ERROR("Randomly_Select_Hubs.");

    num_blocks = (batch_size + block_size - 1) / block_size;
    set_max_float<<<(H * H + block_size - 1) / block_size, block_size>>>(D, H * H);

    for (idx_t batch_id = 0; batch_id < batch_number; batch_id++) {
        Calculate_Distances<<<num_blocks, block_size>>>(
            batch_id, batch_size, n, h->dH, distances, h->d_data, h->dH_psum, h->dH_assignments);
        Construct_D<<<H, block_size>>>(distances, h->dH_assignments, batch_id, batch_size, n, D);
    }
    cudaFree(distances);

    fused_prefix_sum_copy<<<1, dim3(warp_size, warp_size, 1)>>>(h->dH_psum, dH_psum_copy);
    cudaMemcpy(d_psum_placeholder, dH_psum_copy, (H + 1) * sizeof(idx_t), cudaMemcpyDeviceToDevice);
    cudaMemcpy(dH_psum_copy + 1, d_psum_placeholder, H * sizeof(idx_t), cudaMemcpyDeviceToDevice);
    cudaMemcpy(h->dH_psum + 1, d_psum_placeholder, H * sizeof(idx_t), cudaMemcpyDeviceToDevice);
    cudaMemset(h->dH_psum, 0, sizeof(idx_t));
    cudaMemset(dH_psum_copy, 0, sizeof(idx_t));
    CHECK_ERROR("Fused_prefix_sum_copy.");
    cudaFree(d_psum_placeholder);

    num_blocks = (n + block_size - 1) / block_size;
    BucketSort<<<num_blocks, block_size>>>(
        n, h->arr_x, h->arr_y, h->arr_z, h->arr_idx, h->d_data, h->dH_assignments, dH_psum_copy);
    CHECK_ERROR("BucketSort.");
    cudaFree(dH_psum_copy);

    fused_transform_sort_D<float, (H + block_size - 1) / block_size>
        <<<H, dim3{warp_size, block_size / warp_size, 1}>>>(D, h->iD, h->dD);
    CHECK_ERROR("Sort_D.");
    cudaFree(D);

    CUDA_CALL(cudaMalloc((void **)&h->d_hubsScanned, sizeof(int)));
    CUDA_CALL(cudaMalloc((void **)&h->d_pointsScanned, sizeof(int)));

    cudaDeviceSynchronize();
    return h;
}

// Always computes k-NN for ALL n points (self-query; see the header
// comment's "Finding worth recording" -- CLOVER's Qps/q arguments are
// accepted but never used by the kernel). h_results_knn must have room for
// n*k idx_t (unsigned int) entries; row `qp` holds point qp's k nearest
// neighbor indices (original, un-permuted point ids), ascending distance.
void clover_query(void *handle, unsigned int k, idx_t *h_results_knn) {
    auto *h = static_cast<CloverHandle *>(handle);
    idx_t n = h->n;

    idx_t *d_results_knn;
    float *d_results_distances;
    CUDA_CALL(cudaMalloc((void **)&d_results_knn, sizeof(idx_t) * k * n));
    CUDA_CALL(cudaMalloc((void **)&d_results_distances, sizeof(float) * k * n));

    std::size_t constexpr queries_per_block = 128 / warp_size;
    std::size_t num_blocks = util::CEIL_DIV(n, queries_per_block);
    dim3 block_dim{warp_size, queries_per_block, 1};

    switch (util::CEIL_DIV(k, warp_size)) {
        case 1: Query<1><<<num_blocks, block_dim>>>(nullptr, d_results_knn, d_results_distances, k, n, h->d_data, h->dH, h->arr_idx, h->arr_x, h->arr_y, h->arr_z, h->iD, h->dD, h->dH_psum, h->dH_assignments, h->d_hubsScanned, h->d_pointsScanned); break;
        case 2: Query<2><<<num_blocks, block_dim>>>(nullptr, d_results_knn, d_results_distances, k, n, h->d_data, h->dH, h->arr_idx, h->arr_x, h->arr_y, h->arr_z, h->iD, h->dD, h->dH_psum, h->dH_assignments, h->d_hubsScanned, h->d_pointsScanned); break;
        case 3: Query<3><<<num_blocks, block_dim>>>(nullptr, d_results_knn, d_results_distances, k, n, h->d_data, h->dH, h->arr_idx, h->arr_x, h->arr_y, h->arr_z, h->iD, h->dD, h->dH_psum, h->dH_assignments, h->d_hubsScanned, h->d_pointsScanned); break;
        case 4: Query<4><<<num_blocks, block_dim>>>(nullptr, d_results_knn, d_results_distances, k, n, h->d_data, h->dH, h->arr_idx, h->arr_x, h->arr_y, h->arr_z, h->iD, h->dD, h->dH_psum, h->dH_assignments, h->d_hubsScanned, h->d_pointsScanned); break;
        case 5: Query<5><<<num_blocks, block_dim>>>(nullptr, d_results_knn, d_results_distances, k, n, h->d_data, h->dH, h->arr_idx, h->arr_x, h->arr_y, h->arr_z, h->iD, h->dD, h->dH_psum, h->dH_assignments, h->d_hubsScanned, h->d_pointsScanned); break;
        case 6: Query<6><<<num_blocks, block_dim>>>(nullptr, d_results_knn, d_results_distances, k, n, h->d_data, h->dH, h->arr_idx, h->arr_x, h->arr_y, h->arr_z, h->iD, h->dD, h->dH_psum, h->dH_assignments, h->d_hubsScanned, h->d_pointsScanned); break;
        case 7: Query<7><<<num_blocks, block_dim>>>(nullptr, d_results_knn, d_results_distances, k, n, h->d_data, h->dH, h->arr_idx, h->arr_x, h->arr_y, h->arr_z, h->iD, h->dD, h->dH_psum, h->dH_assignments, h->d_hubsScanned, h->d_pointsScanned); break;
        case 8: Query<8><<<num_blocks, block_dim>>>(nullptr, d_results_knn, d_results_distances, k, n, h->d_data, h->dH, h->arr_idx, h->arr_x, h->arr_y, h->arr_z, h->iD, h->dD, h->dH_psum, h->dH_assignments, h->d_hubsScanned, h->d_pointsScanned); break;
        default:
            printf("clover_shim: k=%u exceeds CLOVER's max supported (8*32=256)\n", k);
            std::exit(EXIT_FAILURE);
    }
    CHECK_ERROR("Running scan kernel.");
    CUDA_CALL(cudaMemcpy(h_results_knn, d_results_knn, sizeof(idx_t) * k * n, cudaMemcpyDeviceToHost));
    cudaFree(d_results_knn);
    cudaFree(d_results_distances);
}

void clover_free(void *handle) {
    auto *h = static_cast<CloverHandle *>(handle);
    cudaFree(h->d_data);
    cudaFree(h->dH);
    cudaFree(h->dH_psum);
    cudaFree(h->dH_assignments);
    cudaFree(h->arr_x);
    cudaFree(h->arr_y);
    cudaFree(h->arr_z);
    cudaFree(h->arr_idx);
    cudaFree(h->iD);
    cudaFree(h->dD);
    cudaFree(h->d_hubsScanned);
    cudaFree(h->d_pointsScanned);
    delete h;
}

}  // extern "C"
