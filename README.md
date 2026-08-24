# InstantGR-2026

- 목적 : InstantGR을 더 빠른 global routing으로 최적화

## 진행 상황 <sub>최신 2026-08-24</sub>

| 최적화 | 결과 | 상태 |
| --- | --- | --- |
| FLUTE (GPU-FLUTE) | RSMT **−35~65%** | 완료 |
| Augmented DAG depth | leaf peeling tree-center **기본 on** — S2 GPU −0.6~−1.7s, 순이득 | 완료 ([분석](docs/2026-08-24-s2-critical-path.md)) |
| vcost / presum | 전체 runtime **−30~40%** | nsys 재측정 완료 — 수치 확인됨 |
| GPU batch generation | 전체 **−16.7%** (`mempool_group`), **−10.9%** (`mempool_cluster_ranking`) | 완료 ([정리](docs/2026-08-22-opt.md)) |

- 전체 : `mempool_group` 62.38s → **31.36s (1.99×)**, `mempool_cluster_ranking` 225.28s → **119.93s (1.88×)**, ISPD score 변화 없음
- 항목별 분해 : **[docs/2026-08-23-best-result.md](docs/2026-08-23-best-result.md)** · 측정 절차 : [docs/measure-runtime.md](docs/measure-runtime.md)
- 상세 : **[docs/optimizations.md](docs/optimizations.md)**, 최신 작업 : **[docs/2026-08-22-opt.md](docs/2026-08-22-opt.md)**

---

## 개요

- `src/` : 최적화 버전
- `baseline/` : 비교용 논문 원본
- 셸 : **tcsh 기준** (bash/zsh는 `setenv A B` → `export A=B`)
- 모든 명령 : 리포지토리 루트 기준 상대 경로
- `$BENCH` : `.cap` / `.net` 벤치마크 디렉터리
- `$ARCH` : GPU 아키텍처 (TITAN RTX → `sm_75`)
- 벤치마크 다운로드 : [Google Drive](https://drive.google.com/drive/folders/1afrsbeS_KuSeHEVfuQOuLWPuuZqlDVlw?hl=ko)

```bash
setenv BENCH /path/to/benchmarks
setenv ARCH sm_75
```

---

## 0. GPU 선택

- 빈 GPU 확인 : `nvidia-smi`
- GPU 지정 : `CUDA_VISIBLE_DEVICES`
- 번호 : `nvidia-smi` 인덱스, 여러 장은 `0,1`
- 지정 후 프로그램 내부에서는 항상 device 0
- `-arch` : 실제 사용 GPU와 일치시킬 것

- 세션 전체 적용 :

```bash
setenv CUDA_VISIBLE_DEVICES 0
```

- 한 번만 적용 (tcsh는 `VAR=val cmd` 문법 없음 → `env` 사용) :

```bash
env CUDA_VISIBLE_DEVICES=0 ./InstantGR.opt -cap $BENCH/mempool_cluster_ranking.cap -net $BENCH/mempool_cluster_ranking.net -out test.out
```

---

## 1. 빌드

- 최적화 버전 : `src/` → `run/InstantGR.opt`

```bash
cd src && nvcc main.cpp -o ../run/InstantGR.opt -std=c++17 -x cu -O3 -arch=$ARCH
```

- 논문 baseline : `baseline/src/` → `baseline/run/InstantGR`

```bash
cd baseline/src && nvcc main.cpp -o ../run/InstantGR -std=c++17 -x cu -O3 -arch=$ARCH
```

- evaluator : 최초 1회만

```bash
cd run && g++ -O3 -std=c++17 -o evaluator evaluator.cpp && chmod +x ./evaluator
```

---

## 2. 실행

```bash
cd run && time ./InstantGR.opt -cap $BENCH/mempool_cluster_ranking.cap -net $BENCH/mempool_cluster_ranking.net -out test.out |& tee test.log
```

---

## 3. 평가

```bash
cd run && ./evaluator $BENCH/mempool_cluster_ranking.cap $BENCH/mempool_cluster_ranking.net test.out
```

---

## 4. A/B 자동 실행

- 동작 : 빌드 → opt/baseline 양쪽 실행 → evaluator → 요약표
- 기본 디자인 : `mempool_tile_rank`, `mempool_group`, `mempool_cluster_ranking`, `bsg_chip`
- 디자인 지정 : `./run_ab_no_treecenter.sh bsg_chip`
- 결과 : `ab_results_MMDD_HHMMSS/` 에 로그·출력·eval 저장

```bash
env BENCH=$BENCH ARCH=$ARCH ./run_ab_no_treecenter.sh
```

---

## 5. 참고

- 환경 변수 전체 : [docs/env-vars.md](docs/env-vars.md)
- runtime 측정 절차 (재사용) : [docs/measure-runtime.md](docs/measure-runtime.md)
- 최신 결과 : [docs/2026-08-23-best-result.md](docs/2026-08-23-best-result.md)
- nsys 프로파일링 (vcost/presum 검증) : [docs/profiling-nsys.md](docs/profiling-nsys.md)
- 서버 전용 경로·명령 : [docs/my-setup.md](docs/my-setup.md)
- 논문 원본 : [InstantGR.pdf](docs/papers/InstantGR.pdf) (ICCAD), [InstantGR(Journal).pdf](docs/papers/InstantGR%28Journal%29.pdf) (TCAD, 확장판), [GPU_FLUTE.pdf](docs/papers/GPU_FLUTE.pdf)
- baseline upstream 커밋 : `baseline/UPSTREAM_COMMIT.txt`
