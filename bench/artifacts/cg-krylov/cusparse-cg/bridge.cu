// cuSPARSE/cuBLAS conjugate gradient -- the vendor-library baseline for the
// cg-krylov track. Built into bridge.so and driven in-process through ctypes
// by ./adapter.py (cusparse-cg, plain CG) and ../cusparse-ilu0-cg/adapter.py
// (ILU(0)-preconditioned CG). Both share this one file.
//
// Port of NVIDIA's own CUDA samples, which is what "cuSPARSE CG" means in
// the papers this track compares (SPCG's driver is itself derived from the
// second one):
//   https://github.com/NVIDIA/cuda-samples  @ v12.9
//   Samples/4_CUDA_Libraries/conjugateGradient/main.cpp         (plain CG)
//   Samples/4_CUDA_Libraries/conjugateGradientPrecond/main.cpp  (ILU(0) PCG)
// Kept from the samples: cusparseSpMV (CSR, ALG_DEFAULT) for A*p, cuBLAS
// dot/axpy/scal/copy for the vector ops, host pointer mode (each dot product
// returns its scalar to the host), csrilu02 + two cusparseSpSV triangular
// solves (L unit-lower, U non-unit-upper on the same factored arrays).
// Changed: templated on fp64/fp32; x0 = 0 and b supplied by the caller;
// stop rule is the spec's RELATIVE residual ||r_k|| / ||b|| < rtol (x0 = 0,
// so ||r_0|| = ||b||), or exactly `maxiter` steps in fixed-iteration mode;
// ILU(0) zero pivots are reported as an error instead of being ignored;
// setup (ILU(0) analysis + factorization) and solve are timed separately
// with CUDA events.
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static char g_err[1024] = "";

#define CK_CUDA(x)                                                              \
  do {                                                                          \
    cudaError_t e_ = (x);                                                       \
    if (e_ != cudaSuccess) {                                                    \
      snprintf(g_err, sizeof g_err, "%s:%d %s: %s", __FILE__, __LINE__, #x,     \
               cudaGetErrorString(e_));                                         \
      return 1;                                                                 \
    }                                                                           \
  } while (0)
#define CK_SP(x)                                                                \
  do {                                                                          \
    cusparseStatus_t s_ = (x);                                                  \
    if (s_ != CUSPARSE_STATUS_SUCCESS) {                                        \
      snprintf(g_err, sizeof g_err, "%s:%d %s: %s", __FILE__, __LINE__, #x,     \
               cusparseGetErrorString(s_));                                     \
      return 1;                                                                 \
    }                                                                           \
  } while (0)
#define CK_BL(x)                                                                \
  do {                                                                          \
    cublasStatus_t s_ = (x);                                                    \
    if (s_ != CUBLAS_STATUS_SUCCESS) {                                          \
      snprintf(g_err, sizeof g_err, "%s:%d %s: cublas status %d", __FILE__,     \
               __LINE__, #x, (int)s_);                                          \
      return 1;                                                                 \
    }                                                                           \
  } while (0)

// cuBLAS overloads so the solver body is written once
static cublasStatus_t dot(cublasHandle_t h, int n, const double *x, const double *y, double *r) {
  return cublasDdot(h, n, x, 1, y, 1, r);
}
static cublasStatus_t dot(cublasHandle_t h, int n, const float *x, const float *y, float *r) {
  return cublasSdot(h, n, x, 1, y, 1, r);
}
static cublasStatus_t axpy(cublasHandle_t h, int n, const double *a, const double *x, double *y) {
  return cublasDaxpy(h, n, a, x, 1, y, 1);
}
static cublasStatus_t axpy(cublasHandle_t h, int n, const float *a, const float *x, float *y) {
  return cublasSaxpy(h, n, a, x, 1, y, 1);
}
static cublasStatus_t scal(cublasHandle_t h, int n, const double *a, double *x) {
  return cublasDscal(h, n, a, x, 1);
}
static cublasStatus_t scal(cublasHandle_t h, int n, const float *a, float *x) {
  return cublasSscal(h, n, a, x, 1);
}
static cublasStatus_t copy(cublasHandle_t h, int n, const double *x, double *y) {
  return cublasDcopy(h, n, x, 1, y, 1);
}
static cublasStatus_t copy(cublasHandle_t h, int n, const float *x, float *y) {
  return cublasScopy(h, n, x, 1, y, 1);
}
static cusparseStatus_t ilu0_buffer(cusparseHandle_t h, int n, int nnz, cusparseMatDescr_t d,
                                    double *v, const int *rp, const int *ci, csrilu02Info_t i, int *sz) {
  return cusparseDcsrilu02_bufferSize(h, n, nnz, d, v, rp, ci, i, sz);
}
static cusparseStatus_t ilu0_buffer(cusparseHandle_t h, int n, int nnz, cusparseMatDescr_t d,
                                    float *v, const int *rp, const int *ci, csrilu02Info_t i, int *sz) {
  return cusparseScsrilu02_bufferSize(h, n, nnz, d, v, rp, ci, i, sz);
}
static cusparseStatus_t ilu0_analysis(cusparseHandle_t h, int n, int nnz, cusparseMatDescr_t d,
                                      const double *v, const int *rp, const int *ci, csrilu02Info_t i,
                                      void *buf) {
  return cusparseDcsrilu02_analysis(h, n, nnz, d, v, rp, ci, i, CUSPARSE_SOLVE_POLICY_USE_LEVEL, buf);
}
static cusparseStatus_t ilu0_analysis(cusparseHandle_t h, int n, int nnz, cusparseMatDescr_t d,
                                      const float *v, const int *rp, const int *ci, csrilu02Info_t i,
                                      void *buf) {
  return cusparseScsrilu02_analysis(h, n, nnz, d, v, rp, ci, i, CUSPARSE_SOLVE_POLICY_USE_LEVEL, buf);
}
static cusparseStatus_t ilu0_factor(cusparseHandle_t h, int n, int nnz, cusparseMatDescr_t d,
                                    double *v, const int *rp, const int *ci, csrilu02Info_t i, void *buf) {
  return cusparseDcsrilu02(h, n, nnz, d, v, rp, ci, i, CUSPARSE_SOLVE_POLICY_USE_LEVEL, buf);
}
static cusparseStatus_t ilu0_factor(cusparseHandle_t h, int n, int nnz, cusparseMatDescr_t d,
                                    float *v, const int *rp, const int *ci, csrilu02Info_t i, void *buf) {
  return cusparseScsrilu02(h, n, nnz, d, v, rp, ci, i, CUSPARSE_SOLVE_POLICY_USE_LEVEL, buf);
}

struct Base {
  virtual ~Base() {}
  virtual int setup() = 0;
  virtual int solve(int maxiter, double rtol, int fixed) = 0;
  virtual int copy_x(double *out) = 0;
  int iters = 0;
  double relres_recurrence = NAN;
  float setup_ms = 0.f, solve_ms = 0.f;
};

template <typename T>
struct Solver : Base {
  int n = 0, nnz = 0, precond = 0;
  cudaDataType dt;
  cublasHandle_t cub = nullptr;
  cusparseHandle_t sp = nullptr;
  int *d_rp = nullptr, *d_ci = nullptr;
  T *d_val = nullptr, *d_ilu = nullptr;
  T *d_b = nullptr, *d_x = nullptr, *d_r = nullptr, *d_p = nullptr, *d_q = nullptr;
  T *d_y = nullptr, *d_z = nullptr;
  cusparseSpMatDescr_t matA = nullptr, matL = nullptr, matU = nullptr;
  cusparseDnVecDescr_t vp = nullptr, vq = nullptr, vr = nullptr, vy = nullptr, vz = nullptr;
  cusparseSpSVDescr_t svL = nullptr, svU = nullptr;
  void *spmv_buf = nullptr, *svL_buf = nullptr, *svU_buf = nullptr, *ilu_buf = nullptr;
  cusparseMatDescr_t ilu_descr = nullptr;
  csrilu02Info_t ilu_info = nullptr;
  cudaEvent_t e0 = nullptr, e1 = nullptr;
  double bnorm = 0.0;

  int init(int n_, int nnz_, const int *rp, const int *ci, const double *val,
           const double *b, int precond_) {
    n = n_; nnz = nnz_; precond = precond_;
    dt = sizeof(T) == 8 ? CUDA_R_64F : CUDA_R_32F;
    std::vector<T> hv(val, val + nnz), hb(b, b + n);
    double s = 0.0;
    for (int i = 0; i < n; ++i) s += b[i] * b[i];
    bnorm = std::sqrt(s);

    CK_BL(cublasCreate(&cub));
    CK_SP(cusparseCreate(&sp));
    CK_CUDA(cudaEventCreate(&e0));
    CK_CUDA(cudaEventCreate(&e1));
    CK_CUDA(cudaMalloc(&d_rp, sizeof(int) * (n + 1)));
    CK_CUDA(cudaMalloc(&d_ci, sizeof(int) * nnz));
    CK_CUDA(cudaMalloc(&d_val, sizeof(T) * nnz));
    T **vecs[] = {&d_b, &d_x, &d_r, &d_p, &d_q, &d_y, &d_z};
    for (T **v : vecs) CK_CUDA(cudaMalloc(v, sizeof(T) * n));
    CK_CUDA(cudaMemcpy(d_rp, rp, sizeof(int) * (n + 1), cudaMemcpyHostToDevice));
    CK_CUDA(cudaMemcpy(d_ci, ci, sizeof(int) * nnz, cudaMemcpyHostToDevice));
    CK_CUDA(cudaMemcpy(d_val, hv.data(), sizeof(T) * nnz, cudaMemcpyHostToDevice));
    CK_CUDA(cudaMemcpy(d_b, hb.data(), sizeof(T) * n, cudaMemcpyHostToDevice));
    CK_CUDA(cudaMemset(d_x, 0, sizeof(T) * n));

    CK_SP(cusparseCreateCsr(&matA, n, n, nnz, d_rp, d_ci, d_val, CUSPARSE_INDEX_32I,
                            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, dt));
    CK_SP(cusparseCreateDnVec(&vp, n, d_p, dt));
    CK_SP(cusparseCreateDnVec(&vq, n, d_q, dt));
    CK_SP(cusparseCreateDnVec(&vr, n, d_r, dt));
    CK_SP(cusparseCreateDnVec(&vy, n, d_y, dt));
    CK_SP(cusparseCreateDnVec(&vz, n, d_z, dt));
    T one = 1, zero = 0;
    size_t sz = 0;
    CK_SP(cusparseSpMV_bufferSize(sp, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, matA, vp, &zero,
                                  vq, dt, CUSPARSE_SPMV_ALG_DEFAULT, &sz));
    CK_CUDA(cudaMalloc(&spmv_buf, sz ? sz : 1));

    if (precond) {
      CK_CUDA(cudaMalloc(&d_ilu, sizeof(T) * nnz));
      CK_SP(cusparseCreateMatDescr(&ilu_descr));
      CK_SP(cusparseSetMatType(ilu_descr, CUSPARSE_MATRIX_TYPE_GENERAL));
      CK_SP(cusparseSetMatIndexBase(ilu_descr, CUSPARSE_INDEX_BASE_ZERO));
      CK_SP(cusparseCreateCsrilu02Info(&ilu_info));
      int isz = 0;
      CK_SP(ilu0_buffer(sp, n, nnz, ilu_descr, d_val, d_rp, d_ci, ilu_info, &isz));
      CK_CUDA(cudaMalloc(&ilu_buf, isz ? isz : 1));

      cusparseFillMode_t lo = CUSPARSE_FILL_MODE_LOWER, up = CUSPARSE_FILL_MODE_UPPER;
      cusparseDiagType_t unit = CUSPARSE_DIAG_TYPE_UNIT, nonunit = CUSPARSE_DIAG_TYPE_NON_UNIT;
      CK_SP(cusparseCreateCsr(&matL, n, n, nnz, d_rp, d_ci, d_ilu, CUSPARSE_INDEX_32I,
                              CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, dt));
      CK_SP(cusparseSpMatSetAttribute(matL, CUSPARSE_SPMAT_FILL_MODE, &lo, sizeof lo));
      CK_SP(cusparseSpMatSetAttribute(matL, CUSPARSE_SPMAT_DIAG_TYPE, &unit, sizeof unit));
      CK_SP(cusparseCreateCsr(&matU, n, n, nnz, d_rp, d_ci, d_ilu, CUSPARSE_INDEX_32I,
                              CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, dt));
      CK_SP(cusparseSpMatSetAttribute(matU, CUSPARSE_SPMAT_FILL_MODE, &up, sizeof up));
      CK_SP(cusparseSpMatSetAttribute(matU, CUSPARSE_SPMAT_DIAG_TYPE, &nonunit, sizeof nonunit));
      CK_SP(cusparseSpSV_createDescr(&svL));
      CK_SP(cusparseSpSV_createDescr(&svU));
      CK_SP(cusparseSpSV_bufferSize(sp, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, matL, vr, vy, dt,
                                    CUSPARSE_SPSV_ALG_DEFAULT, svL, &sz));
      CK_CUDA(cudaMalloc(&svL_buf, sz ? sz : 1));
      CK_SP(cusparseSpSV_bufferSize(sp, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, matU, vy, vz, dt,
                                    CUSPARSE_SPSV_ALG_DEFAULT, svU, &sz));
      CK_CUDA(cudaMalloc(&svU_buf, sz ? sz : 1));
    }
    CK_CUDA(cudaDeviceSynchronize());
    return 0;
  }

  // ILU(0) analysis + factorization from the ORIGINAL values, so every call
  // is a fresh factorization; then the triangular-solve analyses.
  int setup() override {
    setup_ms = 0.f;
    if (!precond) return 0;
    T one = 1;
    CK_CUDA(cudaEventRecord(e0));
    CK_CUDA(cudaMemcpy(d_ilu, d_val, sizeof(T) * nnz, cudaMemcpyDeviceToDevice));
    CK_SP(ilu0_analysis(sp, n, nnz, ilu_descr, d_ilu, d_rp, d_ci, ilu_info, ilu_buf));
    int pivot = -1;
    cusparseStatus_t st = cusparseXcsrilu02_zeroPivot(sp, ilu_info, &pivot);
    if (st == CUSPARSE_STATUS_ZERO_PIVOT) {
      snprintf(g_err, sizeof g_err, "ILU(0) analysis: structural zero pivot at row %d", pivot);
      return 1;
    }
    CK_SP(ilu0_factor(sp, n, nnz, ilu_descr, d_ilu, d_rp, d_ci, ilu_info, ilu_buf));
    st = cusparseXcsrilu02_zeroPivot(sp, ilu_info, &pivot);
    if (st == CUSPARSE_STATUS_ZERO_PIVOT) {
      snprintf(g_err, sizeof g_err, "ILU(0) factorization: numerical zero pivot at row %d", pivot);
      return 1;
    }
    CK_SP(cusparseSpSV_analysis(sp, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, matL, vr, vy, dt,
                                CUSPARSE_SPSV_ALG_DEFAULT, svL, svL_buf));
    CK_SP(cusparseSpSV_analysis(sp, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, matU, vy, vz, dt,
                                CUSPARSE_SPSV_ALG_DEFAULT, svU, svU_buf));
    CK_CUDA(cudaEventRecord(e1));
    CK_CUDA(cudaEventSynchronize(e1));
    CK_CUDA(cudaEventElapsedTime(&setup_ms, e0, e1));
    return 0;
  }

  int apply_precond() {  // z = U^-1 L^-1 r
    T one = 1;
    CK_SP(cusparseSpSV_solve(sp, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, matL, vr, vy, dt,
                             CUSPARSE_SPSV_ALG_DEFAULT, svL));
    CK_SP(cusparseSpSV_solve(sp, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, matU, vy, vz, dt,
                             CUSPARSE_SPSV_ALG_DEFAULT, svU));
    return 0;
  }

  int spmv_pq() {  // q = A p
    T one = 1, zero = 0;
    CK_SP(cusparseSpMV(sp, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, matA, vp, &zero, vq, dt,
                       CUSPARSE_SPMV_ALG_DEFAULT, spmv_buf));
    return 0;
  }

  // fixed != 0: exactly `maxiter` steps (stops early only on an exact zero
  // residual, which would otherwise divide by zero). fixed == 0: stop when
  // ||r|| / ||b|| < rtol or after maxiter steps.
  int solve(int maxiter, double rtol, int fixed) override {
    iters = 0;
    relres_recurrence = NAN;
    solve_ms = 0.f;
    const double target = rtol * bnorm;
    CK_CUDA(cudaEventRecord(e0));
    CK_CUDA(cudaMemset(d_x, 0, sizeof(T) * n));
    CK_BL(copy(cub, n, d_b, d_r));  // r = b - A*0
    T rr;
    CK_BL(dot(cub, n, d_r, d_r, &rr));
    double rnorm = std::sqrt((double)rr);
    if (precond) {
      T rho = 0, rho_old = 0;
      while (iters < maxiter && rr != (T)0 && (fixed || rnorm >= target)) {
        if (apply_precond()) return 1;
        rho_old = rho;
        CK_BL(dot(cub, n, d_r, d_z, &rho));
        if (iters == 0) {
          CK_BL(copy(cub, n, d_z, d_p));
        } else {
          T beta = rho / rho_old, one = 1;
          CK_BL(scal(cub, n, &beta, d_p));
          CK_BL(axpy(cub, n, &one, d_z, d_p));
        }
        if (spmv_pq()) return 1;
        T pq;
        CK_BL(dot(cub, n, d_p, d_q, &pq));
        T alpha = rho / pq, nalpha = -alpha;
        CK_BL(axpy(cub, n, &alpha, d_p, d_x));
        CK_BL(axpy(cub, n, &nalpha, d_q, d_r));
        CK_BL(dot(cub, n, d_r, d_r, &rr));
        rnorm = std::sqrt((double)rr);
        ++iters;
      }
    } else {
      T r0 = 0;
      while (iters < maxiter && rr != (T)0 && (fixed || rnorm >= target)) {
        if (iters == 0) {
          CK_BL(copy(cub, n, d_r, d_p));
        } else {
          T beta = rr / r0, one = 1;
          CK_BL(scal(cub, n, &beta, d_p));
          CK_BL(axpy(cub, n, &one, d_r, d_p));
        }
        if (spmv_pq()) return 1;
        T pq;
        CK_BL(dot(cub, n, d_p, d_q, &pq));
        T alpha = rr / pq, nalpha = -alpha;
        CK_BL(axpy(cub, n, &alpha, d_p, d_x));
        CK_BL(axpy(cub, n, &nalpha, d_q, d_r));
        r0 = rr;
        CK_BL(dot(cub, n, d_r, d_r, &rr));
        rnorm = std::sqrt((double)rr);
        ++iters;
      }
    }
    CK_CUDA(cudaEventRecord(e1));
    CK_CUDA(cudaEventSynchronize(e1));
    CK_CUDA(cudaEventElapsedTime(&solve_ms, e0, e1));
    relres_recurrence = bnorm > 0 ? rnorm / bnorm : NAN;
    return 0;
  }

  int copy_x(double *out) override {
    std::vector<T> hx(n);
    CK_CUDA(cudaMemcpy(hx.data(), d_x, sizeof(T) * n, cudaMemcpyDeviceToHost));
    for (int i = 0; i < n; ++i) out[i] = (double)hx[i];
    return 0;
  }

  ~Solver() override {
    if (svL) cusparseSpSV_destroyDescr(svL);
    if (svU) cusparseSpSV_destroyDescr(svU);
    if (matA) cusparseDestroySpMat(matA);
    if (matL) cusparseDestroySpMat(matL);
    if (matU) cusparseDestroySpMat(matU);
    cusparseDnVecDescr_t vs[] = {vp, vq, vr, vy, vz};
    for (auto v : vs) if (v) cusparseDestroyDnVec(v);
    if (ilu_info) cusparseDestroyCsrilu02Info(ilu_info);
    if (ilu_descr) cusparseDestroyMatDescr(ilu_descr);
    void *ptrs[] = {d_rp, d_ci, d_val, d_ilu, d_b, d_x, d_r, d_p, d_q, d_y, d_z,
                    spmv_buf, svL_buf, svU_buf, ilu_buf};
    for (void *p : ptrs) if (p) cudaFree(p);
    if (e0) cudaEventDestroy(e0);
    if (e1) cudaEventDestroy(e1);
    if (sp) cusparseDestroy(sp);
    if (cub) cublasDestroy(cub);
  }
};

extern "C" {

const char *kbcg_last_error() { return g_err; }

// prec: 0 = fp64, 1 = fp32. precond: 0 = none, 1 = ILU(0).
// CSR must be 0-based, square, sorted column indices, diagonal stored.
void *kbcg_prepare(int n, int nnz, const int *rowptr, const int *colind, const double *val,
                   const double *b, int prec, int precond) {
  g_err[0] = 0;
  Base *s = nullptr;
  int rc;
  if (prec == 0) {
    auto *t = new Solver<double>();
    rc = t->init(n, nnz, rowptr, colind, val, b, precond);
    s = t;
  } else {
    auto *t = new Solver<float>();
    rc = t->init(n, nnz, rowptr, colind, val, b, precond);
    s = t;
  }
  if (rc) {
    delete s;
    return nullptr;
  }
  return s;
}

int kbcg_setup(void *h) { return static_cast<Base *>(h)->setup(); }
int kbcg_solve(void *h, int maxiter, double rtol, int fixed) {
  return static_cast<Base *>(h)->solve(maxiter, rtol, fixed);
}
int kbcg_copy_x(void *h, double *out) { return static_cast<Base *>(h)->copy_x(out); }
int kbcg_iters(void *h) { return static_cast<Base *>(h)->iters; }
double kbcg_relres_recurrence(void *h) { return static_cast<Base *>(h)->relres_recurrence; }
float kbcg_setup_ms(void *h) { return static_cast<Base *>(h)->setup_ms; }
float kbcg_solve_ms(void *h) { return static_cast<Base *>(h)->solve_ms; }
void kbcg_free(void *h) { delete static_cast<Base *>(h); }

}  // extern "C"
