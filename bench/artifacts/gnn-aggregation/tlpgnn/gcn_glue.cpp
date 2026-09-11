// Pybind glue for TLPGNN's gcn/naive_kernel.cu (the kernel
// source/gcn/test_kernel.py itself JIT-compiles inline via
// torch.utils.cpp_extension.load_inline -- see that file's `cpp_source`
// string). This file is a byte-for-byte copy of that inline C++ string,
// written out to a real .cpp file so build.sh can drive
// torch.utils.cpp_extension.load() (file-based, ahead-of-time) instead of
// load_inline() (which would otherwise have to run at import/available()
// time -- ARTIFACT_GUIDE.md rule 9 forbids compiling inside available()).
// naive_kernel.cu itself (source/gcn/naive_kernel.cu) is used UNMODIFIED;
// only this pybind wrapper -- which the artifact's own script also
// generates verbatim, just inline rather than as a file -- lives here,
// outside source/ (ARTIFACT_GUIDE.md rule 3: files we write live outside
// source/). NOTE: load_inline() auto-prepends `#include <torch/extension.h>`
// to its cpp_source before compiling; load() (file-based) does not, so it is
// added explicitly here -- the only line not a verbatim copy of the inline
// string, a build-system necessity, not a kernel-code change.
#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> gcn_conv_cuda_forward(
        torch::Tensor features,
        torch::Tensor col_starts,
        torch::Tensor rows);

std::vector<torch::Tensor> gcn_conv_cuda_backward(
        torch::Tensor features,
        torch::Tensor grad,
        torch::Tensor indegs,
        torch::Tensor row_starts,
        torch::Tensor cols);

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> gcn_conv_forward(
        torch::Tensor features,
        torch::Tensor col_starts,
        torch::Tensor rows)
{
    CHECK_INPUT(features);
    CHECK_INPUT(col_starts);
    CHECK_INPUT(rows);

    return gcn_conv_cuda_forward(features, col_starts, rows);
}

std::vector<torch::Tensor> gcn_conv_backward(
        torch::Tensor features,
        torch::Tensor grad,
        torch::Tensor indegs,
        torch::Tensor row_starts,
        torch::Tensor cols)
{
    CHECK_INPUT(features);
    CHECK_INPUT(grad);
    CHECK_INPUT(indegs);
    CHECK_INPUT(row_starts);
    CHECK_INPUT(cols);

    return gcn_conv_cuda_backward(features, grad, indegs, row_starts, cols);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &gcn_conv_forward, "GCN conv forward (CUDA)");
    m.def("backward", &gcn_conv_backward, "GCN conv backward (CUDA)");
}
