# InstantGR-2026

GPU global router **InstantGR** (ICCAD 2024)의 runtime 최적화.
2026 한국 대학생 반도체 설계 경진대회 출품작이며, 이 저장소가 설계보고서의 소스 코드·측정 원본이다.

- 출발점 : [cuhk-eda/InstantGR](https://github.com/cuhk-eda/InstantGR) @ `cfce09f` (`baseline/`에 원본 그대로 보존)
- 대상 : ISPD 2024 GPU/ML-Enhanced Large Scale Global Routing Contest 벤치마크
- 결과 : 같은 바이너리에서 최적화를 전부 끈 것(base) 대비 전부 켠 것(opt)이 **최대 3.27× 빠름**, 라우팅 품질은 노이즈 수준(−0.035% ~ +0.012%)

---

## 1. 결과

RTX 3060 12GB, 최적화 7개를 런타임 스위치로 하나씩 켜고 끈 60런 매트릭스. 전체 표·재현성·품질은 **[docs/rtx3060-ab.md](docs/rtx3060-ab.md)**.

| 디자인 | base | opt | 배율 |
| --- | ---: | ---: | ---: |
| `mempool_group` | 92.47 s | **28.28 s** | **3.27×** |
| `nvdla` | 8.45 s | 3.21 s | 2.63× |
| `bsg_chip` | 24.51 s | 11.52 s | 2.13× |
| `mempool_tile_rank` | 3.27 s | 2.34 s | 1.40× |

- `mempool_cluster_ranking`(10.6M net)은 12GB 카드에서 opt가 GPU-FLUTE scratch OOM → 매트릭스에서 제외. TITAN RTX 24GB 결과는 [docs/archive/2026-08-23-best-result.md](docs/archive/2026-08-23-best-result.md)
- 품질 : 프로그램 자체 report 기준 base 대비 −0.035% ~ +0.012%, 전 런 open net 0 · incomplete 0 ([rtx3060-ab.md §5](docs/rtx3060-ab.md))

## 2. 기여

`mempool_group`에서 opt로부터 해당 기여 하나만 껐을 때 늘어나는 시간(leave-one-out 마진)과 전체 절감(64.19 s) 대비 비중.
각 항목의 설계·시행착오·검증은 **[docs/optimizations.md](docs/optimizations.md)** 의 해당 절에 있다.

| # | 기여 | 출처 | 마진 | 비중 | 설명 |
| ---: | --- | --- | ---: | ---: | --- |
| 1 | **incremental vcost / presum** — batch가 건드린 dirty cell·track만 재계산 | 우리 것 | **+47.08 s** | **73%** | optimizations.md §3 |
| 2 | **incremental wire-demand commit** — 1과 같은 원리를 commit 커널에 적용 | 우리 것 | +6.89 s | 11% | §5 |
| 3 | **GPU batch generation 재설계** — 논문의 우선순위 중재는 계승, 스케줄링은 window/wavefront + net당 warp로 재설계 | TCAD §III-D 계승 + 재설계 | +4.14 s | 6% | §4 |
| 4 | **CPU/GPU FLUTE 오버랩** — 고차수 net은 GPU, 저차수 net은 CPU에서 동시에 처리 | 우리 것 | +1.90 s | 3% | §1 |
| 5 | GPU-FLUTE 알고리즘 도입 (+ break score O(1) precompute) | ICCAD 2022 논문 구현 | +5.58 s | 9% | §1 |
| 6 | tree-center root 선택 (leaf peeling) | 우리 것 | +0.55 s | 1% | §2 |
| 7 | input 파싱 ∥ net 분할 파이프라인 | 우리 것 | pre-route −0.9 s (매트릭스 외) | — | §7 |

- 1 + 2 = 절감의 84%가 한 가지 원리 — "batch는 grid의 P50 1.2%만 건드리는데 baseline은 매번 100%를 재계산한다"
- 3은 `bsg_chip`에서 이득 0 (−0.02 s, 런 간 폭 안). batch generation이 wall에서 차지하는 비중이 큰 디자인에서만 이득 ([rtx3060-ab.md §4](docs/rtx3060-ab.md))
- 미병합 : FLT (Flexible Layer Transition, TCAD §V) — 구현했으나 score −0.256% / 런타임 +11.9%라 기본 빌드에 넣지 않음 (optimizations.md §6)

---

## 3. 저장소 구성

| 경로 | 내용 |
| --- | --- |
| `src/` | 최적화 버전. 진입점 `main.cpp`, 신규 파일은 `gpu_flute.hpp` · `gpu_batch_gen.hpp` · `nvtx_profile.hpp` |
| `baseline/` | upstream 원본 (`UPSTREAM_COMMIT.txt`), A/B 대조용. 수정 없음 |
| `run/` | evaluator 소스, FLUTE lookup table (`POWV9.dat`, `POST9.dat`) |
| `tools/` | `ab_matrix.sh` + `ab_summary.py` (A/B 매트릭스), `nsys_profile.sh` + `nsys_summarize.py` (Nsight Systems 프로파일) |
| `run_ab_no_treecenter.sh` | opt vs 논문 baseline 바이너리 A/B 한 번에 실행 |
| `docs/` | 아래 §6 |
| `.github/workflows/compile.yml` | GPU 없는 CI에서 `src/`·`baseline/`·evaluator 컴파일만 검증 |

---

## 4. 빌드 · 실행 · 평가

- 요구 : CUDA toolkit 12.x (`nvcc`), g++ (C++17), NVIDIA GPU
- 벤치마크 : [Google Drive](https://drive.google.com/drive/folders/1afrsbeS_KuSeHEVfuQOuLWPuuZqlDVlw?hl=ko) 에서 `.cap` / `.net` 을 받아 한 디렉터리에 둔다
- 아래는 bash 기준. tcsh는 `export A=B` → `setenv A B`, 한 번만 적용은 `env A=B cmd`

```bash
export BENCH=/path/to/benchmarks   # .cap / .net 디렉터리
export ARCH=sm_86                  # GPU에 맞게 (RTX 3060 sm_86, TITAN RTX sm_75)
```

빌드 :

```bash
cd src && nvcc main.cpp -o ../run/InstantGR.opt -std=c++17 -x cu -O3 -arch=$ARCH
```

```bash
cd baseline/src && nvcc main.cpp -o ../run/InstantGR -std=c++17 -x cu -O3 -arch=$ARCH
```

```bash
cd run && g++ -O3 -std=c++17 -o evaluator evaluator.cpp
```

실행 (최적화 전부 on이 기본값) :

```bash
cd run && ./InstantGR.opt -cap $BENCH/mempool_group.cap -net $BENCH/mempool_group.net -out mempool_group.out
```

평가 :

```bash
cd run && ./evaluator $BENCH/mempool_group.cap $BENCH/mempool_group.net mempool_group.out
```

- 최적화를 끄려면 환경 변수 (예: `INSTANTGR_INCREMENTAL_VCOST=0`). 전체 목록은 [docs/env-vars.md](docs/env-vars.md)
- GPU 지정 : `CUDA_VISIBLE_DEVICES=0`

## 5. 재현

§1 매트릭스 (바이너리 1개 + 런타임 토글, 결과는 `ab_results_MMDD_HHMMSS/SUMMARY.md`) :

```bash
env BENCH=$BENCH ARCH=$ARCH ./tools/ab_matrix.sh
```

opt vs 논문 baseline 바이너리 :

```bash
env BENCH=$BENCH ARCH=$ARCH ./run_ab_no_treecenter.sh mempool_group
```

- 측정 원칙(연속 실행, 호스트 타이머와 nsys의 역할 분담) : [docs/measure-runtime.md](docs/measure-runtime.md)
- GPU 커널 단위 확인 : [docs/profiling-nsys.md](docs/profiling-nsys.md)

## 6. 문서

| 문서 | 내용 |
| --- | --- |
| [docs/optimizations.md](docs/optimizations.md) | 기여별 설계 · 시행착오 · 결과 · 검증 (본문) |
| [docs/rtx3060-ab.md](docs/rtx3060-ab.md) | 기준 측정 — RTX 3060 60런 A/B 매트릭스 |
| [docs/env-vars.md](docs/env-vars.md) | 런타임 스위치 · 튜닝 노브 · 검증 옵션 |
| [docs/measure-runtime.md](docs/measure-runtime.md) | runtime 측정 절차 |
| [docs/profiling-nsys.md](docs/profiling-nsys.md) | Nsight Systems로 vcost/presum 재측정 |
| [docs/archive/](docs/archive/) | 날짜별 실험 기록 (이전 환경 TITAN RTX 수치 포함) |

## 7. 참고 문헌

- Shiju Lin, Liang Xiao, Jinwei Liu, Evangeline F. Y. Young, ["InstantGR: Scalable GPU Parallelization for Global Routing"](https://shijulin.github.io/files/1239_Final_Manuscript.pdf), ICCAD 2024 — 기반 코드 ([cuhk-eda/InstantGR](https://github.com/cuhk-eda/InstantGR))
- InstantGR 확장판 (TCAD, vol. 45, no. 1, pp. 441–452, 2026) — GPU batch generation(§III-D)과 FLT(§V)의 근거
- "GPU-Accelerated Rectilinear Steiner Tree Generation", ICCAD 2022 — GPU-FLUTE 알고리즘
- ISPD 2024 Contest: GPU/ML-Enhanced Large Scale Global Routing — 벤치마크 · evaluator

## 라이선스

BSD 3-Clause. upstream InstantGR (CUHK EDA)의 라이선스를 그대로 따른다 — [LICENSE](LICENSE).
