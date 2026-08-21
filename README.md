# InstantGR

InstantGR is a GPU-accelerated global routing tool. 

Check out the following paper for more details.

* Shiju Lin, Liang Xiao, Jinwei Liu and Evangeline Young, ["InstantGR: Scalable GPU Parallelization for Global Routing"](https://shijulin.github.io/files/1239_Final_Manuscript.pdf), ACM/IEEE International Conference on Computer-Aided Design (ICCAD), New Jersey, USA, Oct 27–31, 2024.

## Compile
```bash
cd src
nvcc main.cpp -o ../run/InstantGR -std=c++17 -x cu -O3 -arch=sm_80
```
You may want to change `-arch=sm_80` according to your GPU. For example, use
`-arch=sm_86` for NVIDIA RTX 3090 and `-arch=sm_75` for NVIDIA TITAN RTX.
The GPU-FLUTE path uses CUDA Thrust (bundled with the CUDA toolkit); it has no
additional library dependency.

## Run
```bash
cd run
./InstantGR -cap <cap_file_path> -net <net_file_path> -out <output_path>
```
Example:
```bash
./InstantGR -cap ../benchmarks/mempool_tile_rank.cap -net ../benchmarks/mempool_tile_rank.net -out mempool_tile_rank.out
```

### Optional GPU tree-center root

The default preserves the legacy root-selection behavior.  To make only
GPU-FLUTE high-degree nets use their GPU-computed RSMT tree center as the
Stage-2 augmented-DAG root, run:

```tcsh
setenv INSTANTGR_GPU_TREE_CENTER 1
setenv INSTANTGR_GPU_FLUTE_PROFILE 1 # optional timing output
```

Unset `INSTANTGR_GPU_TREE_CENTER` (or set it to `0`) to revert.  With
`INSTANTGR_GPU_FLUTE_VALIDATE=1`, the host checks every GPU-selected root and
falls back to the legacy root if it is not a center of the reconstructed RSMT.

### Optional exact CPU tree-center root

This is the safe A/B experiment for the root-depth hypothesis.  It computes
the center on the host RSMT, and therefore does not depend on the experimental
GPU tree-center reconstruction.  It changes both the Stage-1 basic routing DAG
and the Stage-2 augmented routing DAG.  By default it affects only nets with
degree at least the GPU-FLUTE threshold (10 by default).

```tcsh
setenv INSTANTGR_TREE_CENTER cpu
unsetenv INSTANTGR_GPU_TREE_CENTER
```

Use `INSTANTGR_TREE_CENTER_MIN_DEGREE` to change that population; for example,
`setenv INSTANTGR_TREE_CENTER_MIN_DEGREE 2` applies it to every multi-pin net.
Unset `INSTANTGR_TREE_CENTER` to restore the legacy root.

## Evaluate
```bash
cd run
g++ -o evaluator evaluator.cpp -O3 -std=c++17 #compile the evaluator
./evaluator <cap_file_path> <net_file_path> <output_path> # run the evaluator
```
Example:
```bash
./evaluator ../benchmarks/mempool_tile_rank.cap ../benchmarks/mempool_tile_rank.net mempool_tile_rank.out
```

## Benchmarks

The ISPD2024 benchmarks can be downloaded [here](https://drive.google.com/drive/folders/1bon65UEAx8cjSvVhYJ-lgC8QMDX0fvUm).
We provide a small case `mempool_tile_rank` in the folder `benchmarks` for simple testing.

## Contact
[Shiju Lin](https://shijulin.github.io/) (email: sjlin@cse.cuhk.edu.hk)

## License
BSD 3-Clause License
