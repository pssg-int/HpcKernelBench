# Baseline selection under the revised rule (core-first)

Rule: centrality core > component (tangential excluded); regime matches > partial > mismatch; single-NVIDIA-GPU path required; year only as tiebreak; up to 5 per track. 'integrated' = an artifact dir already exists for that paper.

## amr-kernel  (1 papers, 1 eligible, 1 to integrate)
- [TODO] 2022 SC — A GPU-Accelerated AMR Solver for Gravitational Wave Propagation  (core, regime matches)

## attention-kernel  (9 papers, 8 eligible, 0 to integrate)
- [integrated] 2026 ASPLOS — PAT: Accelerating LLM Decoding via Prefix-Aware Attention with Resource Efficient Multi-Ti  (core, regime matches) → `pat`
- [integrated] 2026 PPoPP — FlashAttention-T: Towards Fully Tensorized Attention by Exploiting Tensor-Vector Paralleli  (core, regime matches) → `flashattention-t`
- [integrated] 2026 PPoPP — MetaAttention: A Unified and Performant Attention Framework across Hardware Backends  (core, regime matches) → `metaattention`
- [integrated] 2023 IPDPS — ByteTransformer: A High-Performance Transformer Boosted for Variable-Length Inputs  (core, regime matches) → `bytetransformer`
- [integrated] 2021 SC — E.T.: re-thinking self-attention for transformer models on GPUs  (core, regime matches) → `et`

## autotuner-multi-kernel  (11 papers, 7 eligible, 5 to integrate)
- [TODO] 2025 ASPLOS — Pruner: A Draft-then-Verify Exploration Mechanism to Accelerate Tensor Program Tuning  (core, regime matches)
- [TODO] 2025 CGO — CuAsmRL: Optimizing GPU SASS Schedules via Deep Reinforcement Learning  (core, regime matches)
- [TODO] 2024 PPoPP — A Holistic Approach to Automatic Mixed-Precision Code Generation and Tuning for Affine Pro  (core, regime matches)
- [TODO] 2023 ASPLOS — SparseTIR: Composable Abstractions for Sparse Compilation in Deep Learning  (core, regime matches)
- [TODO] 2020 ASPLOS — FlexTensor: An Automatic Schedule Exploration and Optimization Framework for Tensor Comput  (core, regime matches)

## batched-gemm  (1 papers, 1 eligible, 1 to integrate)
- [TODO] 2024 ICS — FASTEN: Fast GPU-accelerated Segmented Matrix Multiplication for Heterogenous Graph Neural  (core, regime partial)

## betweenness  (1 papers, 0 eligible, 0 to integrate)

## bfs  (7 papers, 4 eligible, 0 to integrate)
- [integrated] 2026 ICS — BLEST: Blazingly Efficient BFS using Tensor Cores  (core, regime matches) → `blest`
- [integrated] 2021 IPDPS — Speculative Parallel Reverse Cuthill-McKee Reordering on Multi- and Many-core Architecture  (core, regime matches) → `parallelbatchrcm`
- [integrated] 2023 IPDPS — Traversing Large Compressed Graphs on GPUs  (component, regime partial) → `efg`
- [integrated] 2022 IPDPS — Bit-GraphBLAS: Bit-Level Optimizations of Matrix-Centric Graph Processing on GPU  (component, regime partial) → `bit-graphblas`

## blas-level1-2  (4 papers, 0 eligible, 0 to integrate)

## cellular-automata  (1 papers, 1 eligible, 1 to integrate)
- [TODO] 2025 TPDS — CAT: Cellular Automata on Tensor Cores  (core, regime matches)

## cg-krylov  (7 papers, 5 eligible, 0 to integrate)
- [integrated] 2025 SC — Sparsified Preconditioned Conjugate Gradient Solver on GPUs  (core, regime matches) → `spcg`
- [integrated] 2024 SC — Mille-feuille: A Tile-Grained Mixed Precision Single-Kernel Conjugate Gradient Solver on G  (core, regime matches) → `millefeuille`
- [integrated] 2023 ICS — PERKS: a Locality-Optimized Execution Model for Iterative Memory-bound GPU Applications  (core, regime matches) → `perks`
- [integrated] 2023 TPDS — A Multi-GPU Aggregation-Based AMG Preconditioner for Iterative Linear Solvers  (core, regime matches) → `bootcmatchgx`
- [TODO] 2020 SC — Multi-node multi-GPU diffeomorphic image registration for large-scale imaging problems  (component, regime mismatch)

## cholesky  (8 papers, 2 eligible, 0 to integrate)
- [integrated] 2024 SC — Toward Capturing Genetic Epistasis From Multivariate Genome-Wide Association Studies Using  (core, regime matches) → `hicma-x`
- [integrated] 2022 TPDS — Accelerating Geostatistical Modeling and Prediction With Mixed-Precision Computations: A H  (core, regime matches) → `exageostat`

## climate-kernel  (2 papers, 2 eligible, 2 to integrate)
- [TODO] 2026 IPDPS — High-Order-Preserving Acceleration of a Shallow-Water Dynamical Core Using Tensor Units  (core, regime matches)
- [TODO] 2024 SC — Boosting Earth System Model Outputs And Saving PetaBytes in Their Storage Using Exascale C  (core, regime matches)

## community-detection  (4 papers, 2 eligible, 2 to integrate)
- [TODO] 2025 PPoPP — Swift Unfolding of Communities: GPU-Accelerated Louvain Algorithm  (core, regime matches)
- [TODO] 2023 TPDS — A Low-Memory Community Detection Algorithm With Hybrid Sparse Structure and Structural Inf  (core, regime matches)

## connected-components  (4 papers, 3 eligible, 0 to integrate)
- [integrated] 2026 IPDPS — GPU Algorithms for Biconnected Components on Large Graphs  (core, regime matches) → `fast-bcc`
- [integrated] 2023 SC — A GPU Algorithm for Detecting Strongly Connected Components  (core, regime matches) → `ecl-scc`
- [integrated] 2020 TPDS — Optimized Block-Based Algorithms to Label Connected Components on GPUs  (core, regime matches) → `yacclab`

## convolution  (9 papers, 5 eligible, 0 to integrate)
- [integrated] 2024 PPoPP — Tetris: Accelerating Sparse Convolution by Exploiting Memory Reuse on GPU  (core, regime matches) → `tetris`
- [integrated] 2024 ICS — Accelerated Auto-Tuning of GPU Kernels for Tensor Computations  (component, regime matches) → `ansor-af-ds`
- [integrated] 2023 ASPLOS — Hidet: Task-Mapping Programming Paradigm for Deep Learning Tensor Programs  (component, regime partial) → `hidet`
- [TODO] 2021 IPDPS — DSXplore: Optimizing Convolutional Neural Networks via Sliding-Channel Convolutions  (component, regime mismatch)
- [TODO] 2021 TPDS — Accelerating Binarized Neural Networks via Bit-Tensor-Cores in Turing GPUs  (component, regime mismatch)

## dnn-operator-fusion  (5 papers, 3 eligible, 3 to integrate)
- [TODO] 2026 ASPLOS — RedFuser: An Automatic Operator Fusion Framework for Cascaded Reductions on AI Accelerator  (core, regime matches)
- [TODO] 2026 PPoPP — Accelerating Sparse Transformer Inference on GPU  (core, regime matches)
- [TODO] 2022 SC — LightSeq2: Accelerated Training for Transformer-Based Models on GPUs  (core, regime matches)

## dp-dynamic-programming  (1 papers, 0 eligible, 0 to integrate)

## dynamic-graph-kernel  (1 papers, 0 eligible, 0 to integrate)

## eigensolver  (5 papers, 2 eligible, 2 to integrate)
- [TODO] 2025 PPoPP — Improving Tridiagonalization Performance on GPU Architectures  (core, regime matches)
- [TODO] 2025 SC — Rethinking Back Transformation in 2-stage Eigenvalue Decomposition on Heterogeneous Archit  (core, regime matches)

## embedding-ops  (4 papers, 4 eligible, 3 to integrate)
- [TODO] 2024 IPDPS — cuKE: An Efficient Code Generator for Score Function Computation in Knowledge Graph Embedd  (core, regime matches)
- [TODO] 2024 SC — RecFlex: Enabling Feature Heterogeneity-Aware Optimization for Deep Recommendation Models   (core, regime matches)
- [TODO] 2021 ICS — FULL-W2V: fully exploiting data reuse for W2V on GPU-accelerated systems  (core, regime matches)
- [TODO] 2022 SC — EL-Rec: Efficient Large-Scale Recommendation Model Training via Tensor-Train Embedding Tab  (core, regime mismatch)

## fft  (10 papers, 8 eligible, 0 to integrate)
- [integrated] 2025 PPoPP — TurboFFT: Co-Designed High-Performance and Fault-Tolerant Fast Fourier Transform on GPUs  (core, regime matches) → `turbofft`
- [integrated] 2025 SC — TurboFNO: High-Performance Fourier Neural Operator with Fused FFT-GEMM-iFFT on GPU  (core, regime partial) → `turbofno`
- [integrated] 2026 IPDPS — FFCz: Fast Fourier Correction for Spectrum-Preserving Lossy Compression of Scientific Data  (component, regime matches) → `ffcz`
- [integrated] 2026 IPDPS — cuHPX: GPU-Accelerated Differentiable Spherical Harmonic Transforms on HEALPix Grids  (component, regime matches) → `cuhpx`
- [integrated] 2026 TPDS — cuFalcon: An Adaptive Parallel GPU Implementation for High-Performance Falcon Acceleration  (component, regime matches) → `cufalcon`

## gemm  (35 papers, 9 eligible, 0 to integrate)
- [integrated] 2023 ICS — Anatomy of High-Performance GEMM with Online Fault Tolerance on GPUs  (core, regime matches) → `ftgemm`
- [integrated] 2026 CGO — Hexcute: A Compiler Framework for Automating Layout Synthesis in GPU Programs  (component, regime partial) → `hexcute`
- [integrated] 2025 SC — TurboFNO: High-Performance Fourier Neural Operator with Fused FFT-GEMM-iFFT on GPU  (component, regime partial) → `turbofno`
- [integrated] 2024 ASPLOS — Optimizing Dynamic-Shape Neural Networks on Accelerators via On-the-Fly Micro-Kernel Polym  (component, regime partial) → `moonpoly`
- [TODO] 2024 ICS — Accelerated Auto-Tuning of GPU Kernels for Tensor Computations  (component, regime partial)

## gemv  (5 papers, 3 eligible, 0 to integrate)
- [integrated] 2025 PPoPP — MARLIN: Mixed-Precision Auto-Regressive Parallel Inference on Large Language Models  (core, regime matches) → `marlin`
- [integrated] 2026 IPDPS — PackKV: Reducing KV Cache Memory Footprint through LLM-Aware Lossy Compression  (component, regime matches) → `packkv`
- [integrated] 2024 ASPLOS — GPU-based Private Information Retrieval for On-Device Machine Learning Inference  (component, regime mismatch) → `gpu-dpf`

## gnn-aggregation  (11 papers, 8 eligible, 0 to integrate)
- [integrated] 2023 ATC — TC-GNN: Bridging Sparse GNN Computation and Dense Tensor Cores on GPUs  (core, regime matches) → `tc-gnn`
- [integrated] 2022 HPDC — TLPGNN: A Lightweight Two-Level Parallelism Paradigm for Graph Neural Network Computation   (core, regime matches) → `tlpgnn`
- [integrated] 2022 TPDS — Accelerating Backward Aggregation in GCN Training With Execution Path Preparing on GPUs  (core, regime matches) → `eppgcn`
- [integrated] 2020 SC — FeatGraph: a flexible and efficient backend for graph neural network systems  (core, regime matches) → `featgraph`
- [integrated] 2020 SC — GE-SpMM: general-purpose sparse matrix-matrix multiplication on GPUs for graph neural netw  (core, regime matches) → `ge-spmm`
- [kept] `stragcn` — StraGCN: GPU-Accelerated Strassen's Sparse-Dense Matrix Multiplication for Graph (core, regime partial; outside the top-5 cutoff only)

## graph-partitioning  (2 papers, 1 eligible, 1 to integrate)
- [TODO] 2024 DAC — G-kway: Multilevel GPU-Accelerated k-way Graph Partitioner  (core, regime matches)

## graph-pattern-mining  (10 papers, 6 eligible, 0 to integrate)
- [integrated] 2025 PPoPP — GLumin: Fast Connectivity Check Based on LUTs For Efficient Graph Pattern Mining  (core, regime matches) → `glumin`
- [integrated] 2025 SC — Fringe-SGC: Counting Subgraphs with Fringe Vertices  (core, regime matches) → `fringe-sgc`
- [integrated] 2024 PPoPP — Exploiting Fine-Grained Redundancy in Set-Centric Graph Pattern Mining  (core, regime matches) → `graphfold`
- [integrated] 2023 SC — GraphSet: High Performance Graph Mining through Equivalent Set Transformations  (core, regime matches) → `graphset`
- [integrated] 2022 SC — STMatch: Accelerating Graph Pattern Matching on GPU with Stack-Based Loop Optimizations  (core, regime matches) → `stmatch`

## hash-table  (3 papers, 2 eligible, 0 to integrate)
- [TODO] 2023 ATC — Towards Iterative Relational Algebra on the GPU  (component, regime matches)
- [TODO] 2021 IPDPS — Distributed-Memory k-mer Counting on GPUs  (component, regime matches)

## kcore-kclique  (5 papers, 2 eligible, 2 to integrate)
- [TODO] 2026 PPoPP — Root-Down Exposure for Maximal Clique Enumeration on GPUs  (core, regime matches)
- [TODO] 2022 ICS — Parallel K-clique counting on GPUs  (core, regime matches)

## lossless-compression  (7 papers, 5 eligible, 0 to integrate)
- [integrated] 2026 ASPLOS — ZipServ: Fast and Memory-Efficient LLM Inference with Hardware-Aware Lossless Compression  (core, regime matches) → `zipserv`
- [integrated] 2025 SC — MANS: Efficient and Portable ANS Encoding for Multi-Byte Integer Data on CPUs and GPUs  (core, regime matches) → `mans`
- [integrated] 2025 SC — lsCOMP: Efficient Light Source Compression  (core, regime matches) → `lscomp`
- [integrated] 2023 ICS — GPULZ: Optimizing LZSS Lossless Compression for Multi-byte Data on Modern GPUs  (core, regime matches) → `gpulz`
- [integrated] 2022 IPDPS — Optimizing Huffman Decoding for Error-Bounded Lossy Compression on GPUs  (core, regime matches) → `opthuffdec`

## lossy-compression  (20 papers, 9 eligible, 0 to integrate)
- [integrated] 2025 IPDPS — Fast and Effective Lossy Compression on GPUs and CPUs with Guaranteed Error Bounds  (core, regime matches) → `pfpl`
- [integrated] 2025 SC — GPU Lossy Compression for HPC Can Be Versatile and Ultra-Fast  (core, regime matches) → `cuszp`
- [integrated] 2023 HPDC — FZ-GPU: A Fast and High-Ratio Lossy Compressor for Scientific Computing Applications on GP  (core, regime matches) → `fzgpu`
- [integrated] 2023 SC — cuSZp: An Ultra-fast GPU Error-bounded Lossy Compression Framework with Optimized End-to-E  (core, regime matches) → `cuszp-v1`
- [integrated] 2025 SC — lsCOMP: Efficient Light Source Compression  (core, regime mismatch) → `lscomp`

## lu  (3 papers, 1 eligible, 0 to integrate)
- [TODO] 2022 SC — Memory Optimizations in an Array Language  (component, regime matches)

## math-functions  (1 papers, 0 eligible, 0 to integrate)

## md-force  (1 papers, 0 eligible, 0 to integrate)

## mixed-precision-solver  (4 papers, 0 eligible, 0 to integrate)

## moe-kernel  (1 papers, 1 eligible, 0 to integrate)
- [TODO] 2026 CGO — Hexcute: A Compiler Framework for Automating Layout Synthesis in GPU Programs  (component, regime partial)

## morphology-image-kernel  (1 papers, 0 eligible, 0 to integrate)

## mttkrp  (4 papers, 1 eligible, 0 to integrate)
- [integrated] 2022 ICS — Efficient, out-of-memory sparse MTTKRP on massively parallel architectures  (core, regime matches) → `blco`

## multigrid  (5 papers, 2 eligible, 0 to integrate)
- [integrated] 2024 SC — AmgT: Algebraic Multigrid Solver on Tensor Cores  (core, regime matches) → `amgt`
- [integrated] 2023 TPDS — A Multi-GPU Aggregation-Based AMG Preconditioner for Iterative Linear Solvers  (core, regime matches) → `bootcmatchgx`

## nbody  (1 papers, 1 eligible, 1 to integrate)
- [TODO] 2026 ICS — Rethinking Collision Detection on GPU Ray Tracing Architecture  (core, regime matches)

## ntt  (1 papers, 1 eligible, 0 to integrate)
- [TODO] 2026 ASPLOS — Cheddar: A Swift Fully Homomorphic Encryption Library Designed for GPU Architectures  (component, regime matches)

## ann-search  (7 papers, 3 eligible, 0 to integrate)
- [integrated] 2025 ATC — PathWeaver: A High-Throughput Multi-GPU System for Graph-Based Approximate Nearest Neighbo  (core, regime matches) → `pathweaver`
- [integrated] 2025 ICS — CLOVER: A GPU-native, Spatio-graph-based Approach to Exact kNN  (core, regime matches) → `clover`
- [integrated] 2022 PPoPP — RTNN: accelerating neighbor search using hardware ray tracing  (core, regime matches) → `rtnn`

## astar-search  (1 papers, 1 eligible, 1 to integrate)
- [TODO] 2026 ICS — Parallel Bidirectional A* Search for GPU-Accelerated Pathfinding  (core, regime matches)

## bipartite-matching  (1 papers, 1 eligible, 1 to integrate)
- [TODO] 2025 IPDPS — A Bidirectional GPU Algorithm for Computing Maximum Matchings in Bipartite Graphs  (core, regime matches)

## content-defined-chunking  (1 papers, 0 eligible, 0 to integrate)

## erasure-coding  (1 papers, 0 eligible, 0 to integrate)

## graph-layout  (1 papers, 1 eligible, 1 to integrate)
- [TODO] 2024 SC — Rapid GPU-Based Pangenome Graph Layout  (core, regime matches)

## hamiltonian-simulation  (1 papers, 1 eligible, 1 to integrate)
- [TODO] 2026 ICS — Diagonal-Budgeted Trotterization for Efficient Quantum Hamiltonian Simulation  (core, regime matches)

## hausdorff-distance  (1 papers, 1 eligible, 1 to integrate)
- [TODO] 2026 ICS — X-HD: Fast Hausdorff Distance Computation with Ray Tracing  (core, regime matches)

## homotopy-continuation  (1 papers, 1 eligible, 1 to integrate)
- [TODO] 2025 IPDPS — Accelerating Homotopy Continuation with GPUs: Application to Trifocal Pose Estimation  (core, regime matches)

## json-parsing  (1 papers, 1 eligible, 1 to integrate)
- [TODO] 2026 ASPLOS — cuJSON: A Highly Parallel JSON Parser for GPUs  (core, regime matches)

## knn-graph-construction  (1 papers, 0 eligible, 0 to integrate)

## lca-bridge-finding  (1 papers, 1 eligible, 1 to integrate)
- [TODO] 2021 IPDPS — Euler Meets GPU: Practical Graph Algorithms with Theoretical Guarantees  (core, regime matches)

## matrix-function-evaluation  (1 papers, 1 eligible, 1 to integrate)
- [TODO] 2020 SC — A submatrix-based method for approximate matrix function evaluation in the quantum chemist  (core, regime matches)

## maximal-independent-set  (1 papers, 0 eligible, 0 to integrate)

## membership-filter  (1 papers, 1 eligible, 1 to integrate)
- [TODO] 2023 PPoPP — High-Performance Filters for GPUs  (core, regime matches)

## mst  (2 papers, 1 eligible, 1 to integrate)
- [TODO] 2023 SC — A High-Performance MST Implementation for GPUs  (core, regime matches)

## signed-graph-balancing  (1 papers, 1 eligible, 1 to integrate)
- [TODO] 2021 SC — Discovering and balancing fundamental cycles in large signed graphs  (core, regime matches)

## spatial-join  (1 papers, 1 eligible, 1 to integrate)
- [TODO] 2024 ICS — RayJoin: Fast and Precise Spatial Join  (core, regime matches)

## spectral-sparsification  (1 papers, 0 eligible, 0 to integrate)

## spiking-neural-network  (1 papers, 1 eligible, 1 to integrate)
- [TODO] 2020 TPDS — High Performance Simulation of Spiking Neural Network on GPGPUs  (core, regime matches)

## tensor-train-decomposition  (2 papers, 2 eligible, 1 to integrate)
- [TODO] 2026 IPDPS — STTID: High-Performance Sparse Tensor-Train Interpolative Decomposition  (core, regime matches)
- [TODO] 2022 SC — EL-Rec: Efficient Large-Scale Recommendation Model Training via Tensor-Train Embedding Tab  (component, regime matches)

## vertex-cover  (1 papers, 1 eligible, 1 to integrate)
- [TODO] 2022 IPDPS — Parallel Vertex Cover Algorithms on GPUs  (core, regime matches)

## pagerank  (4 papers, 2 eligible, 0 to integrate)
- [TODO] 2023 IPDPS — Traversing Large Compressed Graphs on GPUs  (component, regime matches)
- [TODO] 2022 TPDS — Workload Balancing via Graph Reordering on Multicore Systems  (component, regime matches)

## preconditioner  (5 papers, 1 eligible, 1 to integrate)
- [TODO] 2025 SC — Sparsified Preconditioned Conjugate Gradient Solver on GPUs  (core, regime matches)

## qr  (4 papers, 2 eligible, 1 to integrate)
- [TODO] 2023 IPDPS — PAQR: Pivoting Avoiding QR factorization  (core, regime matches)
- [TODO] 2020 TPDS — cuTensor-Tubal: Efficient Primitives for Tubal-Rank Tensor Learning Operations on GPUs  (component, regime matches)

## quantized-gemm  (16 papers, 12 eligible, 0 to integrate)
- [integrated] 2025 PPoPP — MARLIN: Mixed-Precision Auto-Regressive Parallel Inference on Large Language Models  (core, regime matches) → `marlin`
- [integrated] 2025 SC — MXBLAS: Accelerating 8-bit Deep Learning with a Unified Micro-Scaled GEMM Library  (core, regime matches) → `mxblas`
- [integrated] 2024 ATC — Quant-LLM: Accelerating the Serving of Large Language Models via FP6-Centric Algorithm-Sys  (core, regime matches) → `fp6llm`
- [integrated] 2026 ASPLOS — ZipServ: Fast and Memory-Efficient LLM Inference with Hardware-Aware Lossless Compression  (core, regime partial) → `zipserv`
- [integrated] 2026 PPoPP — High-Throughput Non-uniformly Quantized 3-bit LLM Inference  (core, regime partial) → `quantix`
- [demoted] `tilus` — Tilus: A Tile-Level GPGPU Programming Language for Low-Precision Computation (component, regime matches): keep as competitor, not a SOTA baseline
- [demoted] `qfactory` — QFactory: Accelerating Quantized Large Language Model Serving with Qtile Graphs (component, regime matches): keep as competitor, not a SOTA baseline

## scan-reduction  (3 papers, 3 eligible, 3 to integrate)
- [TODO] 2026 ASPLOS — RedFuser: An Automatic Operator Fusion Framework for Cascaded Reductions on AI Accelerator  (core, regime matches)
- [TODO] 2021 TPDS — GPU Tensor Cores for Fast Arithmetic Reductions  (core, regime matches)
- [TODO] 2020 SC — Compiling generalized histograms for GPU  (core, regime matches)

## sddmm  (13 papers, 9 eligible, 0 to integrate)
- [integrated] 2025 ICS — Fused3S: Fast Sparse Attention on Tensor Cores  (core, regime matches) → `fused3s`
- [integrated] 2025 PPoPP — FlashSparse: Minimizing Computation Redundancy for Fast Sparse Matrix Multiplications on T  (core, regime matches) → `flashsparse`
- [integrated] 2024 PPoPP — A Row Decomposition-based Approach for Sparse Matrix Multiplication on GPUs  (core, regime matches) → `rode`
- [integrated] 2023 IPDPS — Fast Sparse GPU Kernels for Accelerated Training of Graph Neural Networks  (core, regime matches) → `hp-spmm-sddmm`
- [integrated] 2022 SC — Efficient Quantized Sparse Matrix Operations on Tensor Cores  (core, regime matches) → `magicube`
- [demoted] `rassm` — RASSM: Residue-based Acceleration of Single Sparse Matrix Computation via Adapti (component, regime matches, no single-GPU path): keep as competitor, not a SOTA baseline
- [kept] `tc-gnn` — TC-GNN: Bridging Sparse GNN Computation and Dense Tensor Cores on GPUs (core, regime partial; outside the top-5 cutoff only)
- [kept] `sputnik` — Sparse GPU kernels for deep learning (core, regime matches; outside the top-5 cutoff only)

## sequence-alignment  (6 papers, 5 eligible, 0 to integrate)
- [integrated] 2021 TPDS — Parallel Fine-Grained Comparison of Long DNA Sequences in Homogeneous and Heterogeneous GP  (core, regime matches) → `masa-cudalign`
- [integrated] 2020 IPDPS — AnySeq: A High Performance Sequence Alignment Library based on Partial Evaluation  (core, regime matches) → `anyseq`
- [integrated] 2020 IPDPS — LOGAN: High-Performance GPU-Based X-Drop Long-Read Alignment  (core, regime matches) → `logan`
- [integrated] 2020 SC — SegAlign: a scalable GPU-based whole genome aligner  (core, regime matches) → `segalign`
- [integrated] 2022 SC — Memory Optimizations in an Array Language  (component, regime matches) → `futhark-mem-sc22`

## set-intersection  (1 papers, 1 eligible, 1 to integrate)
- [TODO] 2022 IPDPS — Degree-Aware Kernels for Computing Jaccard Weights on GPUs  (core, regime matches)

## sparse-attention-kernel  (4 papers, 4 eligible, 0 to integrate)
- [integrated] 2026 IPDPS — Achieving Low Latency Inference on High Resolution Images by Exploiting Sparsity in Vision  (core, regime matches) → `vit-sparse`
- [integrated] 2026 PPoPP — Accelerating Sparse Transformer Inference on GPU  (core, regime matches) → `sparse-transformer`
- [integrated] 2025 ICS — Fused3S: Fast Sparse Attention on Tensor Cores  (core, regime matches) → `fused3s`
- [integrated] 2025 IPDPS — Longer Attention Span: Increasing Transformer Context Length With Sparse Graph Processing   (core, regime matches) → `gpa`

## sparse-factorization  (5 papers, 1 eligible, 1 to integrate)
- [TODO] 2022 TPDS — gSoFa: Scalable Sparse Symbolic LU Factorization on GPUs  (core, regime matches)

## sparse-format-conversion  (1 papers, 1 eligible, 1 to integrate)
- [TODO] 2025 HPDC — LiteForm: Lightweight and Automatic Format Composition for Sparse Matrix-Matrix Multiplica  (core, regime matches)

## sparse-tensor-contraction  (3 papers, 1 eligible, 1 to integrate)
- [TODO] 2026 ASPLOS — Insum: Sparse GPU Kernels Simplified and Optimized with Indirect Einsums  (core, regime matches)

## spgemm  (13 papers, 6 eligible, 0 to integrate)
- [integrated] 2026 ICS — Ocean: Fast Estimation-Based Sparse General Matrix-Matrix Multiplication on GPU  (core, regime matches) → `ocean`
- [integrated] 2022 PPoPP — TileSpGEMM: a tiled algorithm for parallel sparse general matrix-matrix multiplication on   (core, regime matches) → `tilespgemm`
- [TODO] 2022 IPDPS — Bit-GraphBLAS: Bit-Level Optimizations of Matrix-Centric Graph Processing on GPU  (core, regime mismatch)
- [TODO] 2025 TPDS — ToT: Triangle Counting on Tensor Cores  (component, regime partial)
- [integrated] 2025 PPoPP — Popcorn: Accelerating Kernel K-means on GPUs through Sparse Linear Algebra  (component, regime mismatch) → `popcorn`
- [demoted] `amgt` — AmgT: Algebraic Multigrid Solver on Tensor Cores (component, regime mismatch): keep as competitor, not a SOTA baseline

## spmm  (36 papers, 26 eligible, 0 to integrate)
- [integrated] 2025 ATC — GeneralSparse: Bridging the Gap in SpMM for Pruned Large Language Model Inference on GPUs  (core, regime matches) → `generalsparse`
- [integrated] 2025 ATC — Voltrix: Sparse Matrix-Matrix Multiplication on Tensor Cores with Asynchronous and Balance  (core, regime matches) → `voltrix`
- [integrated] 2025 HPDC — LiteForm: Lightweight and Automatic Format Composition for Sparse Matrix-Matrix Multiplica  (core, regime matches) → `liteform`
- [integrated] 2025 PPoPP — FlashSparse: Minimizing Computation Redundancy for Fast Sparse Matrix Multiplications on T  (core, regime matches) → `flashsparse`
- [integrated] 2025 TPDS — SSpMM: Efficiently Scalable SpMM Kernels Across Multiple Generations of Tensor Cores  (core, regime matches) → `sspmm`
- [demoted] `insum` — Insum: Sparse GPU Kernels Simplified and Optimized with Indirect Einsums (component, regime partial): keep as competitor, not a SOTA baseline
- [demoted] `inferfast` — InferFast: Bridging the Gap Between Unstructured LLM Sparsity and Practical GPU  (component, regime mismatch): keep as competitor, not a SOTA baseline
- [demoted] `rassm` — RASSM: Residue-based Acceleration of Single Sparse Matrix Computation via Adapti (core, regime matches, no single-GPU path): keep as competitor, not a SOTA baseline
- [demoted] `sputnik` — Sparse GPU kernels for deep learning (core, regime mismatch): keep as competitor, not a SOTA baseline
- [kept] `nm-spmm` — NM-SpMM: Accelerating Matrix Multiplication Using N: M Sparsity with GPGPU (core, regime partial; outside the top-5 cutoff only)
- [kept] `mp-spmm` — Bridging the Gap between Unstructured SpMM and Structured Sparse Tensor Cores (core, regime partial; outside the top-5 cutoff only)
- [kept] `dtcspmm` — DTC-SpMM: Bridging the Gap in Accelerating General Sparse Matrix Multiplication  (core, regime matches; outside the top-5 cutoff only)
- [kept] `rode` — A Row Decomposition-based Approach for Sparse Matrix Multiplication on GPUs (core, regime matches; outside the top-5 cutoff only)
- [kept] `smat` — High Performance Unstructured SpMM Computation Using Tensor Cores (core, regime matches; outside the top-5 cutoff only)
- [kept] `tc-gnn` — TC-GNN: Bridging Sparse GNN Computation and Dense Tensor Cores on GPUs (core, regime matches; outside the top-5 cutoff only)
- [kept] `ge-spmm` — GE-SpMM: general-purpose sparse matrix-matrix multiplication on GPUs for graph n (core, regime matches; outside the top-5 cutoff only)

## spmspv  (2 papers, 2 eligible, 1 to integrate)
- [TODO] 2021 TPDS — Adaptive SpMV/SpMSpV on GPUs for Input Vectors of Varied Sparsity  (core, regime matches)
- [TODO] 2026 ICS — BLEST: Blazingly Efficient BFS using Tensor Cores  (component, regime matches)

## spmv  (20 papers, 8 eligible, 0 to integrate)
- [integrated] 2025 ICS — CB-SpMV: A Data Aggregating and Balance Algorithm for for Cache-Friendly Block-Based SpMV   (core, regime matches) → `cb-spmv`
- [integrated] 2023 HPDC — Efficient Algorithm Design of Optimizing SpMV on GPU  (core, regime matches) → `spmv-acc`
- [integrated] 2021 IPDPS — TileSpMV: A Tiled Algorithm for Sparse Matrix-Vector Multiplication on GPUs  (core, regime matches) → `tilespmv`
- [integrated] 2021 TPDS — Adaptive SpMV/SpMSpV on GPUs for Input Vectors of Varied Sparsity  (core, regime partial) → `adaptive-spmv`
- [TODO] 2022 IPDPS — Bit-GraphBLAS: Bit-Level Optimizations of Matrix-Centric Graph Processing on GPU  (core, regime mismatch)
- [demoted] `diaq` — Diagonal-Budgeted Trotterization for Efficient Quantum Hamiltonian Simulation (component, regime mismatch): keep as competitor, not a SOTA baseline
- [demoted] `sspmv` — SSpMV: A Sparsity-aware SpMV Framework Empowered by Multimodal Machine Learning (core, regime matches, no single-GPU path): keep as competitor, not a SOTA baseline

## sptrsv  (5 papers, 2 eligible, 0 to integrate)
- [integrated] 2021 TPDS — A Split Execution Model for SpTRSV  (core, regime matches) → `split-sptrsv`
- [integrated] 2021 TPDS — YuenyeungSpTRSV: A Thread-Level and Warp-Level Fusion Synchronization-Free Sparse Triangul  (core, regime matches) → `yysptrsv`

## sssp  (3 papers, 2 eligible, 1 to integrate)
- [TODO] 2021 PPoPP — A fast work-efficient SSSP algorithm for GPUs  (core, regime matches)
- [TODO] 2023 IPDPS — Traversing Large Compressed Graphs on GPUs  (component, regime partial)

## stencil  (21 papers, 12 eligible, 0 to integrate)
- [integrated] 2026 IPDPS — High-Order-Preserving Acceleration of a Shallow-Water Dynamical Core Using Tensor Units  (core, regime matches) → `ozhope`
- [integrated] 2026 PPoPP — SPIDER: Unleashing Sparse Tensor Cores for Stencil Computation via Strided Swapping  (core, regime matches) → `spider`
- [integrated] 2025 PPoPP — FlashFFTStencil: Bridging Fast Fourier Transforms to Memory-Efficient Stencil Computations  (core, regime matches) → `flashfftstencil`
- [integrated] 2024 PPoPP — ConvStencil: Transform Stencil Computation to Matrix Multiplication on Tensor Cores  (core, regime matches) → `convstencil`
- [integrated] 2024 SC — LoRAStencil: Low-Rank Adaptation of Stencil Computation on Tensor Cores  (core, regime matches) → `lorastencil`
- [kept] `an5d` — AN5D: automated stencil framework for high-degree temporal blocking on GPUs (core, regime matches; outside the top-5 cutoff only)

## string-regex-matching  (5 papers, 3 eligible, 0 to integrate)
- [integrated] 2024 ASPLOS — ngAP: Non-blocking Large-scale Automata Processing on GPUs  (core, regime matches) → `ngap`
- [integrated] 2022 IPDPS — GSpecPal: Speculation-Centric Finite State Machine Parallelization on GPUs  (core, regime matches) → `gspecpal`
- [integrated] 2020 ASPLOS — Why GPUs are Slow at Executing NFAs and How to Make them Faster  (core, regime matches) → `gpunfa`

## svd  (2 papers, 2 eligible, 1 to integrate)
- [TODO] 2020 TPDS — cuTensor-Tubal: Efficient Primitives for Tubal-Rank Tensor Learning Operations on GPUs  (core, regime matches)
- [TODO] 2020 TPDS — High Performance GPU Tensor Completion With Tubal-Sampling Pattern  (component, regime matches)

## tensor-contraction  (4 papers, 2 eligible, 0 to integrate)
- [integrated] 2024 IPDPS — cuKE: An Efficient Code Generator for Score Function Computation in Knowledge Graph Embedd  (core, regime matches) → `cuke`
- [integrated] 2024 PPoPP — Fast Kronecker Matrix-Matrix Multiplication on GPUs  (core, regime matches) → `fastkron`

## tensor-network  (5 papers, 2 eligible, 2 to integrate)
- [TODO] 2021 ICS — HyQuas: hybrid partitioner based quantum circuit simulation system on GPU  (core, regime matches)
- [TODO] 2021 SC — SV-sim: scalable PGAS-based state vector simulation of quantum circuits  (core, regime matches)

## topk-selection  (3 papers, 1 eligible, 0 to integrate)
- [integrated] 2023 SC — Parallel Top-K Algorithms on GPU: A Comprehensive Study and New Methods  (core, regime matches) → `gpu-topk-study`

## transformer-inference-fused  (1 papers, 1 eligible, 1 to integrate)
- [TODO] 2023 IPDPS — ByteTransformer: A High-Performance Transformer Boosted for Variable-Length Inputs  (core, regime matches)

## triangle-counting  (5 papers, 2 eligible, 0 to integrate)
- [integrated] 2025 TPDS — ToT: Triangle Counting on Tensor Cores  (core, regime matches) → `tot`
- [integrated] 2024 IPDPS — A Comparative Study of Intersection-Based Triangle Counting Algorithms on GPUs  (core, regime matches) → `tc-compare`

## trsm  (2 papers, 0 eligible, 0 to integrate)

## tucker  (2 papers, 0 eligible, 0 to integrate)

## winograd  (1 papers, 0 eligible, 0 to integrate)
