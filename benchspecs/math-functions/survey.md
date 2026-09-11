# Survey: math-functions

SINGLETON track — 1 paper (`journals/tpds/ShibataP20`, SLEEF). Surveyed via
arXiv fulltext (ar5iv mirror of `arxiv.org/abs/2001.09258`, since the plain
`arxiv.org/abs` page and `arxiv.org/html` returned abstract-only /404) plus
the artifact repo (`github.com/shibatch/sleef`) directory listing (`docs/`,
`src/libm`, `src/libm-tester`).

## SLEEF (journals/tpds/ShibataP20)

- **Functions benchmarked**: trigonometric (sin, cos, tan), inverse
  trigonometric (asin, acos, atan), exponential (exp), logarithmic (log),
  power (pow), floating-point remainder (fmod).
- **Accuracy classes**: SLEEF ships a **1.0-ULP** and a **3.5-ULP** accuracy
  variant for most functions — the paper's own central speed/accuracy
  trade-off axis.
- **Argument ranges swept** (the paper's own domain buckets, chosen to stress
  its custom Payne-Hanek argument reduction): sin/cos/tan over `[0, 2π]`
  ("small" domain) and `[0, 1e100]` ("large" domain, forces full Payne-Hanek
  reduction); asin/acos over `[-1,1]`; atan/exp over `[-700,700]`; log over
  `[0, 1e300]`; pow over `[-30,30] x [-30,30]`; fmod with numerator in
  `[1,100]` and denominator swept as `r*d` for `r` in `[1, 1e25]`.
- **Vector width / ISA**: primary reported configuration is 256-bit AVX2
  (double precision); comparison baselines were forced to the same 256-bit
  width (`Vector-libm`'s `VECTOR_LENGTH` set to 4) for apples-to-apples
  throughput. SLEEF itself additionally supports SSE2, AVX-512, ARM NEON,
  and SVE (repo `src/arch`, `docs/x86.xhtml`/`aarch64.xhtml`), but the
  fetched paper excerpt only gives quantitative numbers for the AVX2 config.
- **Timing protocol**: a tight loop calling the function `1e10` times in a
  single thread, compiler optimization disabled for the loop body only
  (libraries themselves built at default optimization), timed with
  `clock_gettime`. **Single measurement per configuration — no repetition,
  no reported spread.**
- **Precision / correctness**: not detailed in the fetched excerpt beyond the
  1.0-ULP / 3.5-ULP accuracy-class labels themselves; SLEEF's repo ships a
  dedicated `src/libm-tester` correctness harness (presumably MPFR-based
  ULP measurement, standard for this class of library) not read in this pass.
- **Metric**: reciprocal throughput in **nanoseconds per call** (Table II);
  hardware Intel Core i7-6700 @ 3.40GHz, Turbo Boost disabled ("10ns
  corresponds to 34 clock cycles" — i.e., ~3.4GHz).
- **Baselines**: Intel SVML (`-fimf-max-error=1.0` for the 1-ULP comparison,
  `=4.0` for the 4-ULP comparison), FDLIBM 5.3, Vector-libm. Built with
  gcc-7.3.0, `-O3 -mavx2 -mfma`. Paper's own conclusion: SLEEF's reciprocal
  throughput is "comparable to SVML in all cases."

Source: ar5iv fulltext of arXiv:2001.09258; repo `shibatch/sleef` root +
`docs/` + `src/` listings (GitHub API, no file contents beyond the directory
listing read).

## Divergences

Not applicable — singleton track, one paper. The only internal tension is
between the paper's own single-measurement protocol (no repetition/spread at
all) and this benchmark suite's general fairness requirement of a
median-with-min/max statistic; the spec resolves this by keeping SLEEF's own
argument ranges/functions/ISA target but requiring repeated trials, which is
a strict strengthening, not a contradiction, of SLEEF's own practice.
