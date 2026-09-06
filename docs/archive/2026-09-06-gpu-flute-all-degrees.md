# GPU-FLUTE 전량 적용 실험 (`mempool_group`, RTX 3060)

- 브랜치 `gpu-flute-all-degrees`, 노브 `INSTANTGR_GPU_FLUTE_MIN_DEGREE` (기본 10, `2`면 모든 넷을 GPU FLUTE로)
- 절차는 [measure-runtime.md](../measure-runtime.md) §2·§4 그대로, `opt` 플래그에 `MIN_DEGREE`만 추가

## 결론

- **`-fopenmp`를 켜고 비교하면 이득이 없다.** CPU FLUTE 8스레드 1.19s vs 전량 GPU 1.36s
- 전량 GPU가 이겨 보였던 첫 표는 CPU 쪽이 단일 스레드였기 때문 (아래 "같이 발견한 것")
- 전량 GPU 경로의 병목은 GPU solve(0.4s)가 아니라 호스트 트리 재구성(8스레드 1.05s). 넷당 `gpu_flute_tree_to_graph`의 sort/lower_bound + `finalize_rsmt`의 unordered_map이 CPU FLUTE 한 번 값과 맞먹음. 이걸 더 줄이지 않는 한 min-degree 10을 유지하는 게 맞음

## 결과 — `-Xcompiler -fopenmp` 빌드 (기준)

| config | S1 RSMT | Lshape route | total | evaluator total |
| --- | --- | --- | --- | --- |
| min-degree 10 (기존) | **1.19 s** (CPU FLUTE, 8스레드) | 6.99 s | **24.90 s** | 397,655,610 |
| min-degree 2 (전량 GPU) | 1.36 s (GPU tail) | 7.30 s | 25.44 s | 397,657,365 |

- 로그·`.out`·`.eval` : `results_mempool_group_gpuflute_all/omp_opt.*`, `omp_all.*`

## OpenMP 빌드로 기존 config(degree 10) 정식 페어 재측정

[measure-runtime.md](../measure-runtime.md) §2 그대로, 빌드에만 `-Xcompiler -fopenmp` 추가 :

| config | S1 RSMT | Lshape route | DAG/detour | total | evaluator total |
| --- | --- | --- | --- | --- | --- |
| `off` (4개 전부 off) | 2.13 s | 29.01 s | 40.71 s | 76.61 s | 397,666,067 |
| `opt` (degree 10, 1회) | 1.19 s | 6.99 s | 11.06 s | 24.90 s | 397,655,610 |
| `opt` (degree 10, 2회) | 1.16 s | 6.97 s | 11.18 s | 24.96 s | 397,655,210 |

- 배율 76.61 / 24.93 = **3.07×** (OpenMP 없는 `rtx3060-ab.md` : 92.47 → 28.28 s, 3.27×)
- `off`가 더 많이 줄어드는 이유 : `off`는 249만 넷 전부를 CPU FLUTE로 돌리므로 8스레드 효과가 더 큼 (RSMT 약 10 s → 2.1 s). `opt`는 이미 GPU FLUTE·오버랩으로 가려져 있던 5.1 s 구간만 1.2 s로 줄어듦
- 따라서 빌드 명령을 OpenMP로 바꾸면 헤드라인 배율은 3.27× → 3.07×로 **낮아지고** 절대 시간은 둘 다 줄어듦. 보고서에 어느 빌드를 기준으로 쓸지 정해야 함
- `opt` 2회 편차 0.06 s, 스코어 편차 0.0001% — 노이즈 대역
- 로그 : `results_mempool_group_gpuflute_all/omp_off.*`, `omp_opt.*`, `omp_opt_r2.*`

## 결과 — `-fopenmp` 없는 빌드 (기존 README 빌드 명령)

| config | S1 RSMT | Lshape route | total | evaluator total |
| --- | --- | --- | --- | --- |
| min-degree 10 (기존) | 5.29 s (CPU FLUTE, 실제로는 단일 스레드) | 11.16 s | 29.48 s | 397,654,787 |
| min-degree 2 (전량 GPU) | 1.37 s (GPU tail) | 7.34 s | 25.36 s | 397,656,662 |

- 스코어 차이 +0.0005% — 런간 노이즈 대역 (`rtx3060-ab.md` 기준 ±0.0002%)
- `INSTANTGR_GPU_FLUTE_VALIDATE=1` : 2,491,299 넷 중 Hanan 불일치 0, WL 불일치 1 (degree 34 넷 1건, 기존 고차수 경로에도 있던 것)

## 왜 처음엔 안 빨라졌나

- 첫 시도(호스트 루프 그대로) : GPU wall 5.12s ≈ CPU FLUTE 5.15s → 이득 0
- `INSTANTGR_GPU_FLUTE_PROFILE=1` : solve wall 0.41s, **host tree rebuild 4.75s**
  - `construct_rsmt_gpu()`의 넷별 `gpu_flute_tree_to_graph` + `finalize_rsmt` 루프가 직렬
- 이 루프를 `std::thread` 8개로 나눔 → rebuild 1.05s, `construct_rsmt_gpu` 총 1.44s

## 같이 발견한 것

- README/measure-runtime의 빌드 명령에 `-fopenmp`가 없어 `Lshape_route.hpp`의 `#pragma omp parallel for num_threads(8)`가 무시됨 (`ldd`에 libgomp 없음). 기존 CPU FLUTE 5.3s는 **단일 스레드** 수치였고, `-Xcompiler -fopenmp`를 붙이면 1.19s. 이것만으로 `mempool_group` total 29.5s → 24.9s. **지금까지의 3060 측정치(`rtx3060-ab.md`)는 전부 OpenMP 꺼진 빌드** — 빌드 명령을 바꾸면 재측정 필요
- 다른 디자인(`mempool_tile_rank`, `mempool_cluster_ranking`, `bsg_chip`)은 미측정
