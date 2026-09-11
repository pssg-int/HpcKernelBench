// Minimal stand-in for YACCLAB's own include/register.h.
//
// Real register.h wires each algorithm class into YACCLAB's own CLI-driven
// algorithm/kernel name registry (LabelingMapSingleton / KernelMapSingleton)
// so their `main.cc` benchmark driver can look algorithms up by name --
// that registry is exactly the "paper's own benchmark script" machinery
// ARTIFACT_GUIDE rule 1 says to wrap AROUND, not reproduce. This
// integration instantiates `BUF`/`BKE` directly (see bridge_yacclab_buf.cu),
// so the registry itself is unused -- these macros only need to expand to
// syntactically valid (harmless) statements at the exact call sites
// labeling_allegretti_2019_BUF.cu/_BKE.cu use, verbatim, unmodified.
#ifndef YACCLAB_SHIM_REGISTER_H_
#define YACCLAB_SHIM_REGISTER_H_

// Call site: `REGISTER_LABELING(BUF);` (caller supplies the trailing ';').
#define REGISTER_LABELING(x) namespace { static const int x##_registered_unused_ = 0; }

// Call site: `REGISTER_KERNELS(BUF, InitLabeling, Compression, Merge, FinalLabeling)`
// (no trailing ';' at the call site -- the macro must supply its own).
#define REGISTER_KERNELS(x, ...) namespace { static const int x##_kernels_registered_unused_ = 0; }

// Call site: several consecutive `BLOCKSIZE_KERNEL(Kernel, grid, block, shmem, args...)`
// invocations with no separating ';' -- each expansion must be a complete
// statement (a real kernel launch, so PerformLabelingBlocksize -- unused by
// this integration but must still compile -- stays behaviorally correct if
// anyone ever does call it).
#define BLOCKSIZE_KERNEL(kernel, grid, block, shmem, ...) kernel<<<grid, block, shmem>>>(__VA_ARGS__);

#endif  // YACCLAB_SHIM_REGISTER_H_
