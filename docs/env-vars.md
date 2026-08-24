# 환경 변수 (A/B 스위치)

- 셸 : tcsh 기준 — `VAR=val cmd` 문법 없음
- 한 번만 적용 : `env VAR=val ./InstantGR.opt ...`
- 세션 고정 : `setenv VAR val`
- 빌드·실행 절차 : [README](../README.md)

- 사용 예 (tree-center 없이 나머지 최적화만) :

```bash
env INSTANTGR_TREE_CENTER=0 INSTANTGR_GPU_TREE_CENTER=0 ./InstantGR.opt -cap $BENCH/bsg_chip.cap -net $BENCH/bsg_chip.net -out bsg.out
```

---

## 최적화 on/off (기본 on, `0`으로 끔)

| 변수 | 기본 | 설명 |
| --- | --- | --- |
| `INSTANTGR_GPU_FLUTE` | on | 고차수(degree ≥ 10) 넷의 FLUTE를 GPU에서 수행. `0`이면 전량 CPU FLUTE |
| `INSTANTGR_INCREMENTAL_VCOST` | on | via cost를 dirty-cell 리스트로만 갱신 (전체 그리드 재계산 회피) |
| `INSTANTGR_INCREMENTAL_PRESUM` | on | presum도 동일하게 증분 갱신 |
| `INSTANTGR_GPU_BATCH_GEN` | on | batch generation을 GPU에서 수행 (journal Sec. III-D). `0`이면 CPU first-fit |
| `INSTANTGR_TREE_CENTER` | on | 호스트 측 정확한 tree-center 루트 선택 (leaf peeling). `0`이면 legacy BFS 루트 |

## 튜닝 노브

| 변수 | 기본 | 설명 |
| --- | --- | --- |
| `INSTANTGR_GPU_FLUTE_MAX_DEGREE` | 0 (무제한) | 이 차수 초과는 CPU FLUTE로 우회 |
| `INSTANTGR_INCREMENTAL_VCOST_MAX_DIRTY` | 0.20 | dirty-cell 리스트 크기 상한(전체 그리드 대비 비율, (0,1]) |
| `INSTANTGR_TREE_CENTER_MIN_DEGREE` | 기본 10 | tree-center 적용 최소 차수 |

고정값 (환경 변수 없음) :

- GPU/CPU FLUTE 분기 차수 : 10 (`DEGREE + 1`) — `src/database_cuda.hpp`의 `GPU_FLUTE_MIN_DEGREE`
- break score 계산 : O(1) precompute — `src/gpu_flute.hpp`의 `precompute_break_arrays()`

## 실험 (기본 off, opt-in)

| 변수 | 값 | 설명 |
| --- | --- | --- |
| `INSTANTGR_GPU_TREE_CENTER` | `1` | GPU 측 근사 tree-center 루트 선택 — CPU tree-center가 우선하므로 `INSTANTGR_TREE_CENTER=0`과 함께 써야 함 |

## 검증 / 프로파일링 (`1`로 켬, 느려짐)

| 변수 | 설명 |
| --- | --- |
| `INSTANTGR_GPU_FLUTE_VALIDATE` | GPU FLUTE 결과를 CPU 결과와 대조 |
| `INSTANTGR_GPU_BATCH_GEN_VALIDATE` | batch 내 충돌 검사 + CPU batch 결과와 비교 (CPU 경로도 같이 돌림) |
| `INSTANTGR_INCREMENTAL_VCOST_VALIDATE` | 매 배치마다 전체 그리드를 재계산해 vcost 일치 확인 |
| `INSTANTGR_INCREMENTAL_PRESUM_VALIDATE` | presum에 대한 동일 검증 |
| `INSTANTGR_GPU_FLUTE_PROFILE` | GPU FLUTE 단계별 시간 출력 |
| `INSTANTGR_AUGMENTED_DAG_PROFILE` | Stage 2 augmented-DAG 깊이 / 구간별 시간 출력 + tree-center의 legacy 대비 depth 통계·peel 오라클 검증 |
| `INSTANTGR_BATCH_COVERAGE` | Stage 2 배치가 실제로 건드리는 그리드 비율 출력 |
| `INSTANTGR_RSMT_DEPTH_PROFILE` | `1`이면 `rsmt_depth_profile.csv` 생성, 경로를 주면 그 경로로 기록 |
| `INSTANTGR_RSMT_DEPTH_PROFILE_MIN_PINS` | 위 프로파일 대상 최소 핀 수 (기본 10) |
