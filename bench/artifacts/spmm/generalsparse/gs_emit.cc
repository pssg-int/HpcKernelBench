// gs_emit.cc -- ours, NOT part of GeneralSparse's own source/ (ARTIFACT_GUIDE.md
// rule 3: files we write live outside source/, and this one is linked against
// source/'s already-built .o files by build.sh).
//
// GeneralSparse's own token_test.cc::main() hardcodes a benchmark SCRIPT: it
// loops over ~30 fixed operator compositions per matrix, and for EACH one
// calls code_generator::generate_final_program(10000) (writes kernel_file.cu
// with a 10000-iteration internal repeat loop) immediately followed by
// execute_binary() (nvcc-compiles that file to a.out, in unsupported
// arithmetic executes it via `system()`, wall-clock-times via gettimeofday
// around a 10000-rep loop, and greps the resulting a.out's stdout) -- i.e.
// codegen+compile+run+pick-best is baked into ONE C++ function with no CLI
// hook to run a single strategy or to get the generated kernel text without
// also compiling/running it (ARTIFACT_GUIDE.md rule 1: "wrap the kernel, not
// the paper's benchmark script").
//
// This program reproduces exactly ONE of those ~30 strategies --
// test_spmm_warp_bitmap's operator sequence, verbatim (fixed thread blocking
// in the column direction + a warp-bitmap reduction; a real, representative
// GeneralSparse composition, not a synthetic simplification) -- but stops at
// `generate_final_program(1)` (writes kernel_file.cu with NO internal repeat
// loop, so the file contains exactly one `kernel_<id><<<...>>>(...)` call)
// and never calls execute_binary(): no nvcc invocation, no a.out run
// happens in this program. adapter.py::prepare() reads the resulting
// kernel_file.cu text itself, restructures it (init/run split, see
// _gs_transform.py) and compiles/loads that restructured version -- so the
// FULL codegen search over other strategies is out of scope for this
// integration (documented in STATUS.md as a scope reduction, not a kernel
// change: the kernel this program emits is byte-for-byte what GeneralSparse's
// own code_builder/code_generator produce for this operator sequence).
//
// Usage: ./gs_emit <mtx_path> <N>
// Prints exactly one line to stdout: "<output_id>"
// (the data_source/<output_id>/ directory containing kernel_file.cu).

#include <iostream>
#include <memory>
#include "code_source_data.hpp"
#include <map>
#include "metadata_set.hpp"
#include "data_transform_step.hpp"
#include "operator.hpp"
#include "kernel_generator.h"
#include "code_generator.hpp"
#include <string>
#include <fstream>
#include "operator_executer.hpp"
#include "reduction_token.hpp"

using namespace std;

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        cerr << "usage: gs_emit <mtx_path> <N>" << endl;
        return 1;
    }
    string matrix_path = argv[1];
    int N = atoi(argv[2]);
    set_config("DENSE_MATRIX_SIZE", N);

    // ---- verbatim copy of test_spmm_warp_bitmap's operator sequence
    //      (token_test.cc:1253-1308), stopping before execute_binary() ----
    shared_ptr<meta_data_set> meta_dataset_ptr =
        create_init_metadata_set_from_file(matrix_path, "kb");

    shared_ptr<code_generator> code_generator_ptr(new code_generator(meta_dataset_ptr, 0));
    shared_ptr<operator_executer> operator_executer_ptr(new operator_executer());

    int sparse_coarsen_factor = 4;
    int coarsen_factor = 1;

    shared_ptr<basic_operator> fixed_interval_col_direction_thread_blocking_operator_ptr(
        new fixed_interval_col_direction_thread_blocking_operator(
            code_generator_ptr, 64, false, false, true, false,
            operator_executer_ptr->get_operator_context()));
    operator_executer_ptr->add_and_run(fixed_interval_col_direction_thread_blocking_operator_ptr);

    assert(meta_dataset_ptr->check());
    assert(logical_check(meta_dataset_ptr) == true);

    unsigned int grid_x = meta_dataset_ptr->get_element(GLOBAL_META, "origin_row_num", -1)
                              ->get_metadata_arr()->read_integer_from_arr(0);
    int y_size = min((int)get_config()["DENSE_MATRIX_SIZE"].as_float() / coarsen_factor, 32);
    int x_size = max(128 / y_size, 32);
    set_config("VECTOR_WIDTH", x_size);

    shared_ptr<thread_total_reduce_operator> thread_total_reduce_operator_ptr(
        new thread_total_reduce_operator(code_generator_ptr, true, sparse_coarsen_factor,
                                         coarsen_factor, operator_executer_ptr->get_operator_context()));
    operator_executer_ptr->add_and_run(thread_total_reduce_operator_ptr);

    shared_ptr<warp_bit_map_operator> warp_reduce_operator_ptr(
        new warp_bit_map_operator(code_generator_ptr, coarsen_factor, true, true,
                                  operator_executer_ptr->get_operator_context()));
    operator_executer_ptr->add_and_run(warp_reduce_operator_ptr);

    vector<unsigned int> block;
    block.push_back(x_size);
    block.push_back(y_size);

    shared_ptr<grid_block_operator> grid_block_operator_ptr(
        new grid_block_operator(code_generator_ptr, grid_x, block, coarsen_factor,
                                operator_executer_ptr->get_operator_context()));
    operator_executer_ptr->add_and_run(grid_block_operator_ptr);

    code_generator_ptr->compile();

    // repeat_num=1 (NOT the artifact's own 10000): we want the raw,
    // un-repeated kernel launch so adapter.py can lift it into a shared
    // library and let the harness's own per-iteration CUDA-event timer
    // repeat it, instead of GeneralSparse's own internal wall-clock-timed
    // repeat loop (rule: harness timing, not the artifact's benchmark loop).
    unsigned long output_id = code_generator_ptr->generate_final_program(1);

    cout << output_id << endl;
    return 0;
}
