//==================================================================
// kernelbench_driver.cu -- HPC-KernelBench integration driver for LOGAN.
//
// ADDITIVE ONLY: this file is new (not part of the upstream LOGAN repo,
// which is otherwise untouched -- `git -C source diff` against upstream is
// empty except for this one new file). It calls LOGAN's own public library
// entry point, extendSeedL() (declared in logan_functions.cuh, the exact
// same function demo.cu calls), directly -- no LOGAN source file is
// modified, per ARTIFACT_GUIDE.md rule 3.
//
// Why a new driver instead of the shipped `demo` binary: demo.cu computes
// scores into an `int* res` buffer and then immediately `free(res)`s it
// WITHOUT ever printing or returning the scores -- the shipped CLI has no
// way to observe a single alignment score at all (confirmed by reading
// demo.cu; only a single aggregate wall-clock number is printed). This
// driver is the finest available boundary that actually exposes per-pair
// scores: it reads a TSV file in LOGAN's OWN input format (identical to
// inputs_demo/example.txt) plus scoring/X-drop parameters, calls
// extendSeedL() once, and writes one score per line to an output file.
//==================================================================
#include "logan.cuh"
#include <fstream>
#include <sstream>
#include <iostream>

using namespace std;

int main(int argc, char **argv)
{
    if (argc != 10) {
        cerr << "usage: kernelbench_driver <input.tsv> <output.txt> <ksize> "
                "<xdrop> <match> <mismatch> <gap_ext> <gap_open> <ngpus>\n";
        return 1;
    }
    string input_path = argv[1];
    string output_path = argv[2];
    int ksize = atoi(argv[3]);
    int xdrop = atoi(argv[4]);
    int match = atoi(argv[5]);
    int mismatch = atoi(argv[6]);
    int gap_ext = atoi(argv[7]);
    int gap_open = atoi(argv[8]);
    int ngpus = atoi(argv[9]);

    ifstream input(input_path);
    if (!input) { cerr << "cannot open input " << input_path << "\n"; return 1; }
    vector<string> lines;
    string line;
    while (getline(input, line)) {
        if (!line.empty()) lines.push_back(line);
    }
    input.close();
    int N = (int)lines.size();
    if (N == 0) { cerr << "empty input\n"; return 1; }

    vector<int> posV(N), posH(N);
    vector<SeedL> seeds(N);
    vector<string> seqsV(N), seqsH(N);
    // Same ScoringSchemeL construction demo.cu uses. NOTE (see STATUS.md):
    // LOGAN's compiled GPU kernel (logan_functions.cu:computeAntidiag) uses
    // the compile-time MATCH/MISMATCH/GAP_EXT macros directly and NEVER
    // reads this struct's fields inside any kernel launch -- passed here
    // for fidelity to the upstream API and because extendSeedL() DOES read
    // gap_extend_score/gap_open_score for one host-side validity check
    // (must be < 0), but the actual match/mismatch/gap VALUES used by the
    // GPU computation are always the hardcoded macros regardless of what is
    // passed here.
    vector<ScoringSchemeL> penalties(N, ScoringSchemeL(match, mismatch, gap_ext, gap_open));

    for (int i = 0; i < N; i++) {
        stringstream ss(lines[i]);
        string tok;
        vector<string> f;
        while (getline(ss, tok, '\t')) f.push_back(tok);
        if (f.size() < 4) { cerr << "malformed line " << i << ": " << lines[i] << "\n"; return 1; }
        seqsV[i] = f[0];
        posV[i] = stoi(f[1]);
        seqsH[i] = f[2];
        posH[i] = stoi(f[3]);
        SeedL sseed(posH[i], posV[i], ksize);
        seeds[i] = sseed;
    }

    cudaFree(0); // init CUDA context, same as demo.cu

    vector<int> res(N, -123456789); // pre-filled with an obviously-wrong sentinel purely for
                                     // debugging visibility; extendSeedL() always overwrites
                                     // every res[i] itself (its own internal scoreLeft/
                                     // scoreRight buffers can still carry garbage for
                                     // edge-anchored seeds -- see STATUS.md's uninitialized-
                                     // buffer finding, which this sentinel does NOT prevent).
    extendSeedL(seeds, EXTEND_BOTHL, seqsH, seqsV, penalties, xdrop, ksize,
                res.data(), N, ngpus, 128);

    ofstream out(output_path);
    for (int i = 0; i < N; i++) out << res[i] << "\n";
    out.close();
    return 0;
}
