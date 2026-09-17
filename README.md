# InstantGR-2026

Runtime optimizations on top of [InstantGR](https://github.com/cuhk-eda/InstantGR) (ICCAD 2024), a GPU global router for the ISPD 2024 benchmarks.
`src/` is the optimized router, `baseline/` is the upstream code at `cfce09f` kept unchanged for comparison.

## Requirements

- CUDA toolkit 12.x (`nvcc`), g++ with C++17, an NVIDIA GPU
- Benchmarks (`.cap` / `.net`): [ISPD 2024 contest, Google Drive](https://drive.google.com/drive/folders/1afrsbeS_KuSeHEVfuQOuLWPuuZqlDVlw?hl=ko). Put them in one directory.

```bash
export BENCH=/path/to/benchmarks
export ARCH=sm_86        # match your GPU (RTX 3060: sm_86, TITAN RTX: sm_75)
```

## Build

```bash
cd src && nvcc main.cpp -o ../run/InstantGR.opt -std=c++17 -x cu -O3 -arch=$ARCH
```

```bash
cd baseline/src && nvcc main.cpp -o ../run/InstantGR -std=c++17 -x cu -O3 -arch=$ARCH
```

```bash
cd run && g++ -O3 -std=c++17 -o evaluator evaluator.cpp
```

## Run

```bash
cd run && ./InstantGR.opt -cap $BENCH/mempool_group.cap -net $BENCH/mempool_group.net -out mempool_group.out
```

All optimizations are on by default. Each one can be switched off with an environment variable, e.g. `INSTANTGR_INCREMENTAL_VCOST=0`. The full list is in [docs/env-vars.md](docs/env-vars.md).

## Evaluate

```bash
cd run && ./evaluator $BENCH/mempool_group.cap $BENCH/mempool_group.net mempool_group.out
```

## A/B scripts

- `tools/ab_matrix.sh` — one binary, every optimization toggled on/off at runtime; writes `ab_results_*/SUMMARY.md`
- `run_ab_no_treecenter.sh [design ...]` — optimized build vs. the upstream baseline binary

```bash
env BENCH=$BENCH ARCH=$ARCH ./tools/ab_matrix.sh
```

## Layout

| Path | Contents |
| --- | --- |
| `src/` | optimized router |
| `baseline/` | upstream InstantGR, unchanged (`UPSTREAM_COMMIT.txt`) |
| `run/` | evaluator source, FLUTE lookup tables |
| `tools/` | A/B matrix and Nsight Systems profiling scripts |
| `docs/` | measurements and notes (Korean) |

## License

BSD 3-Clause, same as upstream InstantGR (CUHK EDA). See [LICENSE](LICENSE).
