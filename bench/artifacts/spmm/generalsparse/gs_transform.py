"""
Text-surgery that turns ONE of GeneralSparse's own generated `kernel_file.cu`
programs (produced by gs_emit, this directory's own driver -- see its header
comment) into a small shared library this adapter can call once per harness
repetition with fresh operands, instead of running GeneralSparse's own
`main()` (which mallocs+H2D-copies everything, launches the kernel exactly
once, times it with `gettimeofday` wall-clock, and exits -- ARTIFACT_GUIDE.md
rule 1: wrap the kernel, not the paper's benchmark script).

Why this is possible at all: code_generator::generate_kernel_calling_code()
(source/code_generator.cc:486-535) emits, for `kernel_repeat_number == 1`,
EXACTLY one statement of the form
    kernel_<id><<<grid_dim, block_dim>>>(d_arg1, ..., d_argN, d_y_arr, d_x_arr, K);
with a fixed, greppable argument-name convention (every device pointer is
named `d_<metadata-item-name>`, always declared on its own preceding line as
`TYPE* d_<name>;` by generate_matrix_format_read_code(), one line before its
cudaMalloc/cudaMemcpy pair). That determinism -- confirmed empirically against
a real generated file, source/data_source/<id>/kernel_file.cu, while writing
this module -- is what makes a general (not hand-tuned per matrix) text split
possible.

The split:
  - KERNEL BLOCK (the `__global__ void kernel_<id>(...) { ... }` definition):
    copied byte-for-byte, never touched (rule 1/3: this is the actual paper
    kernel, and the only thing we must not alter).
  - SETUP (host reads of the metadata arrays this specific box-division
    strategy needs + their cudaMalloc/cudaMemcpy to device -- i.e. GeneralSparse's
    own real preprocessing/format-conversion output, ARTIFACT_GUIDE.md rule 2):
    copied verbatim EXCEPT the standalone `TYPE* d_name;` declaration lines
    are deleted (that pointer becomes a file-scope static instead, so a
    later, separate function can still see it) -- this is the only structural
    change, and it changes storage duration, not arithmetic.
  - kb_init(): the SETUP text, wrapped in `extern "C" void kb_init()`, plus
    the grid/block/K assignment (also textually lifted, decl->assignment).
  - kb_run(void* c_ptr, void* b_ptr): the ONE kernel-launch statement
    (textually identical whichever of its two occurrences in the file --
    GeneralSparse's own generated main() launches the same statement twice,
    once as an unused "warm-up" whose result is never compared against
    anything real -- see STATUS.md's "double-accumulation" finding -- and
    once as the "timed" call; we only need the statement text once), with
    `d_y_arr`/`d_x_arr` rebound to the two pointers passed in from Python
    (real, harness-generated operands) instead of GeneralSparse's own
    generated host arrays, which this module never emits at all (no
    `y_arr`/`x_arr` malloc/fill/cudaMalloc/cudaMemcpy text is copied --
    cut off before it, see `_CUT_MARKER`).

The result is byte-identical GeneralSparse kernel arithmetic, invoked with
harness-controlled operands and timed by the harness's own per-iteration
CUDA-event timer -- exactly the rode/insum/mp-spmm adapters' established
ctypes-plus-torch-tensor-pointers pattern in this same track.
"""

from __future__ import annotations

import re

_CUT_MARKER = re.compile(r'\n\w[\w ]*\*\s*y_arr\s*=')
_KERNEL_DECL_RE = re.compile(r'__global__ void (kernel_\d+)\(([^)]*)\)\s*\n\{')
_GRID_RE = re.compile(r'dim3 grid_dim\(([^;]*)\);')
_BLOCK_RE = re.compile(r'dim3 block_dim\(([^;]*)\);')
_K_RE = re.compile(r'unsigned int K = (\d+);')
_CALL_RE = re.compile(r'(kernel_(\d+)<<<grid_dim, block_dim>>>\(([^;]*)\));')
_DECL_RE = re.compile(r'^(\w[\w ]*)\*\s*(d_[A-Za-z0-9_]+);$', re.MULTILINE)


class GSTransformError(RuntimeError):
    pass


def build_shim(kernel_file_text: str, kernel_lib_include_dir: str) -> str:
    """Return the full text of shim.cu for one generated kernel_file.cu."""
    text = kernel_file_text

    m = _KERNEL_DECL_RE.search(text)
    if not m:
        raise GSTransformError("could not find __global__ kernel declaration")
    kernel_name = m.group(1)
    kernel_params = m.group(2)

    main_idx = text.find("int main(int argc, char** argv)")
    if main_idx < 0:
        raise GSTransformError("could not find generated main()")
    kernel_block = text[:main_idx].strip()
    # the original file's own '#include "kernel_lib.hpp"' (relative, found via
    # the compile dir) is replaced by our absolute one below -- drop it here
    # to avoid a duplicate #include.
    kernel_block = re.sub(r'^#include\s*"kernel_lib\.hpp"\s*\n*', "",
                          kernel_block, count=1)

    body_start = text.index("{", main_idx) + 1
    body_end = text.rindex("return 0;")
    body = text[body_start:body_end]

    cut = _CUT_MARKER.search(body)
    if not cut:
        raise GSTransformError("could not find y_arr cut marker in generated main()")
    setup = body[:cut.start()]

    gm, bm, km = _GRID_RE.search(body), _BLOCK_RE.search(body), _K_RE.search(body)
    if not (gm and bm and km):
        raise GSTransformError("could not find grid_dim/block_dim/K in generated main()")

    cm = _CALL_RE.search(body)
    if not cm:
        raise GSTransformError("could not find kernel launch statement")
    call_stmt = cm.group(1)
    if cm.group(2) != kernel_name.split("_")[1]:
        raise GSTransformError("kernel launch id does not match kernel declaration id")

    # last two pointer params before K in the kernel signature are y_arr/x_arr
    # (C, B) -- see generate_kernel_calling_code: "... << 'd_y_arr, d_x_arr, K'".
    param_list = [p.strip() for p in kernel_params.split(",")]
    if len(param_list) < 3:
        raise GSTransformError(f"unexpected kernel param list: {kernel_params!r}")
    y_decl, x_decl = param_list[-3], param_list[-2]
    y_type = y_decl.rsplit("*", 1)[0].strip()
    x_type = x_decl.rsplit("*", 1)[0].strip()

    decls = _DECL_RE.findall(setup)
    if not decls:
        raise GSTransformError("no device-pointer declarations found in setup")
    setup_no_decls = _DECL_RE.sub("", setup)

    globals_block = "\n".join(f"static {t}* {n} = nullptr;" for t, n in decls)
    globals_block += "\nstatic dim3 grid_dim;\nstatic dim3 block_dim;\nstatic unsigned int K = 0;\n"

    free_block = "\n".join(f"    if ({n}) cudaFree({n});" for _, n in decls)

    init_tail = (
        f"grid_dim = dim3({gm.group(1)});\n"
        f"block_dim = dim3({bm.group(1)});\n"
        f"K = {km.group(1)};\n"
    )

    shim = f"""// AUTO-GENERATED by gs_transform.py from GeneralSparse's own
// data_source/<id>/kernel_file.cu (see that module's header comment for the
// exact split rule). NOT hand-written kernel code: the __global__ block
// below is copied byte-for-byte from GeneralSparse's own code_builder /
// code_generator output for this matrix + operator composition.
#include "{kernel_lib_include_dir}/kernel_lib.hpp"

{kernel_block}

// ---- promoted from kernel_file.cu's generated main()'s locals (rule: only
// storage duration changed, not arithmetic -- see module docstring) ----
{globals_block}

extern "C" void kb_init()
{{
{setup_no_decls}
{init_tail}
}}

extern "C" void kb_run(void* c_ptr, void* b_ptr)
{{
    {y_type}* d_y_arr = ({y_type}*)c_ptr;
    {x_type}* d_x_arr = ({x_type}*)b_ptr;
    {call_stmt};
}}

extern "C" void kb_free()
{{
{free_block}
}}
"""
    return shim
