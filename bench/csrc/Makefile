# Build the baseline CUDA kernels into a shared library loaded by ctypes.
#
# Compilation is safe on a login node; RUNNING the resulting kernels for timing
# is not (shared, contended GPU) — use a compute-node allocation for that.

NVCC    ?= nvcc
# A100 (sm_80) and H100 (sm_90); extend as needed.
ARCHS   ?= -gencode arch=compute_80,code=sm_80
CXXFLAGS = -O3 -std=c++17 -Xcompiler -fPIC -lineinfo
TARGET   = libkernelbench.so

all: $(TARGET)

$(TARGET): kernels.cu
	$(NVCC) $(CXXFLAGS) $(ARCHS) -shared -o $@ $<

# Syntax/codegen check without linking a library (fast, no GPU needed).
check: kernels.cu
	$(NVCC) $(CXXFLAGS) $(ARCHS) -c -o /dev/null $<
	@echo "kernels.cu compiles cleanly"

clean:
	rm -f $(TARGET) *.o

.PHONY: all check clean
