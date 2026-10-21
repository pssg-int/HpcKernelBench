// AmgX (NVIDIA's algebraic multigrid library) as an AMG-preconditioned CG
// solver -- the baseline BootCMatchGX (TPDS'23) reports its speedup over.
// Built into bridge.so against source/'s libamgxsh.so and driven in-process
// through ctypes by ./adapter.py. Follows AmgX's own C-API example
// (source/examples/amgx_capi.c): config from a JSON string, simple
// single-GPU resources, CSR upload, setup, solve, download.
// Changed vs the example: b and x0 = 0 come from the caller; output is
// captured instead of printed; setup and solve are timed separately.
#include <amgx_c.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static char g_err[2048] = "";
static std::string g_log;  // last AmgX output, kept for error messages
static bool g_init = false;

static void capture(const char *msg, int len) {
  g_log.append(msg, len);
  if (g_log.size() > 8192) g_log.erase(0, g_log.size() - 8192);
}

#define CK(x)                                                                    \
  do {                                                                           \
    AMGX_RC rc_ = (x);                                                           \
    if (rc_ != AMGX_RC_OK) {                                                     \
      char m_[256];                                                              \
      AMGX_get_error_string(rc_, m_, sizeof m_);                                 \
      std::string tail_ = g_log.size() > 600 ? g_log.substr(g_log.size() - 600) : g_log; \
      snprintf(g_err, sizeof g_err, "%s: %s | amgx output: %s", #x, m_, tail_.c_str()); \
      return 1;                                                                  \
    }                                                                            \
  } while (0)

struct Handle {
  int n = 0, prec = 0;
  AMGX_Mode mode;
  AMGX_config_handle cfg = nullptr;
  AMGX_resources_handle rsrc = nullptr;
  AMGX_matrix_handle A = nullptr;
  AMGX_vector_handle b = nullptr, x = nullptr;
  AMGX_solver_handle solver = nullptr;
  int iters = 0, status = -1;
  double setup_ms = 0, solve_ms = 0;
};

static int build(Handle *h, int n, int nnz, const int *rp, const int *ci, const double *val,
                 const double *b, const char *config) {
  if (!g_init) {
    CK(AMGX_initialize());
    CK(AMGX_register_print_callback(capture));
    g_init = true;
  }
  CK(AMGX_config_create(&h->cfg, config));
  CK(AMGX_resources_create_simple(&h->rsrc, h->cfg));
  CK(AMGX_matrix_create(&h->A, h->rsrc, h->mode));
  CK(AMGX_vector_create(&h->b, h->rsrc, h->mode));
  CK(AMGX_vector_create(&h->x, h->rsrc, h->mode));
  if (h->prec == 0) {
    CK(AMGX_matrix_upload_all(h->A, n, nnz, 1, 1, rp, ci, val, nullptr));
    CK(AMGX_vector_upload(h->b, n, 1, b));
  } else {
    std::vector<float> fv(val, val + nnz), fb(b, b + n);
    CK(AMGX_matrix_upload_all(h->A, n, nnz, 1, 1, rp, ci, fv.data(), nullptr));
    CK(AMGX_vector_upload(h->b, n, 1, fb.data()));
  }
  CK(AMGX_vector_set_zero(h->x, n, 1));
  return 0;
}

extern "C" {

const char *kbamgx_last_error() { return g_err; }
const char *kbamgx_last_log() { return g_log.c_str(); }

// prec: 0 = fp64 (AMGX_mode_dDDI), 1 = fp32 (AMGX_mode_dFFI).
// config: AmgX JSON config text. CSR 0-based, full (both triangles).
void *kbamgx_prepare(int n, int nnz, const int *rowptr, const int *colind, const double *val,
                     const double *b, int prec, const char *config) {
  g_err[0] = 0;
  Handle *h = new Handle();
  h->n = n;
  h->prec = prec;
  h->mode = prec == 0 ? AMGX_mode_dDDI : AMGX_mode_dFFI;
  if (build(h, n, nnz, rowptr, colind, val, b, config)) {
    delete h;  // leaks AmgX objects created before the failure; prepare failed anyway
    return nullptr;
  }
  return h;
}

static int run_impl(Handle *h) {
  if (h->solver) {
    CK(AMGX_solver_destroy(h->solver));
    h->solver = nullptr;
  }
  // fresh AMG hierarchy every call (setup is part of time-to-solution)
  CK(AMGX_solver_create(&h->solver, h->rsrc, h->mode, h->cfg));
  cudaDeviceSynchronize();
  auto t0 = std::chrono::steady_clock::now();
  CK(AMGX_solver_setup(h->solver, h->A));
  cudaDeviceSynchronize();
  auto t1 = std::chrono::steady_clock::now();
  CK(AMGX_vector_set_zero(h->x, h->n, 1));
  CK(AMGX_solver_solve_with_0_initial_guess(h->solver, h->b, h->x));
  cudaDeviceSynchronize();
  auto t2 = std::chrono::steady_clock::now();
  h->setup_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  h->solve_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
  CK(AMGX_solver_get_iterations_number(h->solver, &h->iters));
  AMGX_SOLVE_STATUS st;
  CK(AMGX_solver_get_status(h->solver, &st));
  h->status = (int)st;
  return 0;
}

int kbamgx_run(void *p) { return run_impl(static_cast<Handle *>(p)); }

int kbamgx_copy_x(void *p, double *out) {
  Handle *h = static_cast<Handle *>(p);
  if (h->prec == 0) {
    CK(AMGX_vector_download(h->x, out));
  } else {
    std::vector<float> fx(h->n);
    CK(AMGX_vector_download(h->x, fx.data()));
    for (int i = 0; i < h->n; ++i) out[i] = fx[i];
  }
  return 0;
}

int kbamgx_iters(void *p) { return static_cast<Handle *>(p)->iters; }
int kbamgx_status(void *p) { return static_cast<Handle *>(p)->status; }
double kbamgx_setup_ms(void *p) { return static_cast<Handle *>(p)->setup_ms; }
double kbamgx_solve_ms(void *p) { return static_cast<Handle *>(p)->solve_ms; }

void kbamgx_free(void *p) {
  Handle *h = static_cast<Handle *>(p);
  if (h->solver) AMGX_solver_destroy(h->solver);
  if (h->A) AMGX_matrix_destroy(h->A);
  if (h->b) AMGX_vector_destroy(h->b);
  if (h->x) AMGX_vector_destroy(h->x);
  if (h->rsrc) AMGX_resources_destroy(h->rsrc);
  if (h->cfg) AMGX_config_destroy(h->cfg);
  delete h;
}

}  // extern "C"
