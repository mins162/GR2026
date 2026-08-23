# InstantGR 최적화 정리

- 최종 업데이트 : **2026-08-23**
- 기준 자료 : 2026-08-20 미팅 발표 (`0820 논문.pptx`, 슬라이드 21~33)
- 대상 : ISPD 2024 global routing (InstantGR)
- 벤치마크 : `mempool_group` (3.2M net), `mempool_cluster_ranking` (10.6M net)

| # | 항목 | 상태 | 갱신일 |
| --- | --- | --- | --- |
| 1 | FLUTE 개선 (GPU-FLUTE) | 완료 | 2026-08-20 |
| 2 | Augmented DAG depth 개선 | depth ↓, runtime 변화 없음 → critical path 분석 예정 | 2026-08-20 |
| 3 | vcost / presum 계산 절감 | **nsys 재측정 완료 — 기존 수치 확인됨** | 2026-08-23 |

---

## 1. FLUTE 개선 (GPU-FLUTE) <sub>2026-08-20</sub>

- 근거 논문 : *GPU-Accelerated Rectilinear Steiner Tree Generation* (ICCAD'22)
- 문제 : high-degree net이 RSMT runtime의 대부분을 차지
- 접근
  - net을 subnet으로 분할
  - degree < 10 subnet에 스레드 1개씩 할당해 GPU 병렬 처리
  - 재귀(DFS) → 레벨 병렬(BFS) levelized 구조
  - break 단계는 prefix-sum으로 다음 레벨 메모리 오프셋 계산
  - merge 단계는 마지막 레벨부터 역순 병합
  - 루프 동안 CPU-GPU 복사 없이 GPU 내부에서 처리
- 구현 위치 : `src/gpu_flute.hpp`, `src/database_cuda.hpp`
- 분기 차수 : degree ≥ 10 → GPU, degree < 10 → CPU (`GPU_FLUTE_MIN_DEGREE`, 고정값)
- break score : O(1) precompute 방식 (`precompute_break_arrays()`)
  - 원래의 storage-free 재계산은 candidate당 O(d²) → 2153-pin net에서 ~100s로 붕괴
  - 두 방식의 스코어는 bit-identical → 트리 동일

### 결과 (RSMT 구간)

| 벤치마크 | upstream | +GPU-FLUTE | +FLUTE pipeline |
| --- | --- | --- | --- |
| `mempool_group` | 13.7s (CPU) | 7.27s (−47%) | 7.25s |
| `mempool_cluster_ranking` | 58.6s (CPU) | 38.09s (−35%) | 20.72s (−64.6%) |

- 품질 : ISPD score 변화 없음 (노이즈 수준)

---

## 2. Augmented DAG depth 개선 <sub>2026-08-20</sub>

- 목표 : 최대 병렬화를 위한 DAG depth 최소화
- 파이프라인 : `net → FLUTE → RSMT → center 노드 검색 → center 기준 DFS → DAG(Directed)`
- 기존 방식 : BFS로 잡은 임의 노드 기준 → depth가 커짐
- 수정한 알고리즘 : BFS 2회로 tree center 탐색
  - 1차 BFS : 임의 노드에서 가장 먼 노드 탐색
  - 2차 BFS : 그 노드에서 가장 먼 노드 탐색
  - 두 노드를 잇는 경로의 중간점 = depth 최소화 root

### 결과 (mempool_tile_rank, 유효 net 125,054개 · RSMT hop 기준)

| 대상 | net 수 | baseline 평균 | center 평균 | 개선 net 비중 | 최대 depth |
| --- | --- | --- | --- | --- | --- |
| degree ≥ 10 (적용 대상) | 6,537 | 11.10 | 8.58 (−23%) | 77.9% | 45 → 28 |
| degree < 10 | 118,517 | 1.38 | 1.18 | 19.7% | 7 → 4 |
| 전체 | 125,054 | 1.89 | 1.56 | 22.7% | 45 → 28 |

### 문제 및 다음 작업

- **depth는 평균 23% 줄었으나 runtime은 거의 그대로**
- 원인 후보
  - augmentation된 edge 위에서 수행 → center를 미리 찾아둬도 augment 시 center가 이동
  - 평균 depth가 아니라 **critical path(가장 긴 serial chain)** 가 runtime을 지배
- 다음 작업 : **critical path 탐색**
  - 실제 GPU level chain 중 가장 긴 경로를 프로파일링해 병목 확인
  - `INSTANTGR_AUGMENTED_DAG_PROFILE=1` 로 Stage 2 depth·구간별 시간 수집
  - critical depth 자체를 줄이는 방향(augment 생성량 조절, score 개선, 다른 탐색 방법) 검토
- 현재 코드에서는 opt-in 실험으로만 유지
  - `INSTANTGR_TREE_CENTER=cpu` (호스트 측 정확한 tree-center)
  - `INSTANTGR_GPU_TREE_CENTER=1` (GPU 측 근사)

---

## 3. vcost / presum 계산 절감 <sub>2026-08-20</sub>

- 대상 : Stage 2의 `update_cost` (vcost) + `compute_presum`
- baseline 방식 : 갱신마다 전체 grid 스캔·수정
- 관찰 : batch의 bounding box coverage가 매우 낮음

| 커버리지 | 평균 | P50 | P90 | max |
| --- | --- | --- | --- | --- |
| cells | 13.80% | 1.20% | 61.89% | 97.25% |
| tracks | 37.30% | 23.72% | 98.64% | 99.90% |

### batch 파이프라인

1. **ripup** — 이전 route의 wire/via demand 제거, 바뀐 셀을 `dirty_cells` · `track_dirty`에 기록
2. **update_cost (vcost)** — `update_vcost_selective` : dirty 셀 + 방향상 이웃 1칸만 재계산 (vcost는 지역 계산)
3. **compute_presum** — track 단위 dirty skip
   - prefix sum이라 track 중간이 바뀌면 뒤가 전부 오염 → track 단위로 관리
   - 깨끗한 track은 즉시 return (저장된 presum 재사용)
   - dirty cell > grid 20% → 스캔 중지, full rebuild
4. **DP** — bottom-up DP → traceback (변경 없음)
5. **commit** — `commit_all_edge` / `commit_wire_demand` / `commit_via_demand`
   - demand 변경 + dirty 기록
   - wire는 segment당 atomic 1회로 구간 예약 (카운터 경합 방지)

- presum이 예상(−67%)을 상회한 이유
  - 커버리지 히스토그램은 **읽는 영역(bbox)**, skip 조건은 **직전 batch가 쓴(commit) 영역**
  - dirty cell stamp ≪ 읽기 영역 → skip이 더 자주 성립

### 결과 — mempool_group (Stage 2)

| 구간 | +GPU-FLUTE | +Vcost+presum | 개선 |
| --- | --- | --- | --- |
| `update_cost` | 9.98s | 0.09s | −99.1% |
| `compute_presum` | 2.41s | 0.56s | −76.8% |
| ripup-commit | 0.80s | 0.84s | +5.0% |
| bottom-up DP | 3.48s | 3.52s | +1.1% |
| commit | 2.20s | 2.24s | +1.8% |
| traceback | 0.26s | 0.26s | −0.8% |
| 합계 | 19.13s | 7.51s | −60.7% |

### 결과 — mempool_cluster_ranking (GPU 구간)

| 항목 | base | incremental | 개선 |
| --- | --- | --- | --- |
| `update_cost` | 28.57s | 0.33s | −98.8% |
| `compute_presum` | 8.85s | 3.82s | −56.8% |
| ripup-commit | 4.45s | 4.69s | +5.4% |
| bottom-up DP | 9.69s | 9.70s | — |
| commit | 6.12s | 6.32s | +3.3% |
| GPU 합계 | 58.45s | 25.63s | −56.2% |

### 전체 runtime

| 벤치마크 | upstream | +GPU-FLUTE | +Vcost+presum | +FLUTE pipeline |
| --- | --- | --- | --- | --- |
| `mempool_group` | 66.19s | 61.67s (−6.7%) | 40.93s (−39.3%) | 38.20s (−42.3%) |
| `mempool_cluster_ranking` | 233.34s | 220.09s (−5.7%) | 162.17s (−30.5%) | 140.81s (−39.7%, 1.66×) |

- **runtime은 줄었고, evaluate(ISPD score)도 정상적으로 나옴**
  - `mempool_group` : 397,600,453 → 397,601,920 (노이즈 수준)
  - `mempool_cluster_ranking` : 1,780,762,387 → 1,780,724,749 (+0.0002%, 노이즈)

### 재측정 — nsys (2026-08-23) <sub>결론 : 기존 수치가 맞았음</sub>

- 방법 : [profiling-nsys.md](profiling-nsys.md), 브랜치 `profile/nsys-vcost-presum`
- 코드가 찍는 `cudaEvent` 대신 **nsys(CUPTI)** 가 드라이버에서 직접 받은 커널별 시간
- `mempool_group`, TITAN RTX (sm_75), `cells=38,763,846` · `tracks=18,578`, 배치 = Stage 1 601 + Stage 2 866
- config
  - `paper` : `baseline/` 논문 원본 (CPU FLUTE)
  - `full` : `src/` + incremental 둘 다 off (전체 재계산)
  - `incr` : `src/` + incremental 둘 다 on (현재)
  - `full` / `incr`은 GPU-FLUTE를 양쪽 다 켜서 vcost·presum만 변수로 남김

| config | wall | GPU busy | 커널 안 도는 시간 | vcost | presum | vcost+presum |
| --- | --- | --- | --- | --- | --- | --- |
| `paper` | 64.27s | 30.59s (47.6%) | 33.68s (52.4%) | 16.99s | 4.20s | **21.19s = wall의 33.0%** (GPU의 69.3%) |
| `full` | 56.24s | 29.51s (52.5%) | 26.73s (47.5%) | 10.88s | 8.93s | 19.81s = wall의 35.2% (GPU의 67.1%) |
| `incr` | 36.77s | 10.99s (29.9%) | 25.78s (70.1%) | 0.17s | 0.85s | 1.02s = wall의 2.8% (GPU의 9.3%) |

- **논문 원본에서 vcost+presum은 wall의 33%, GPU 작업의 69%다.**
  단일 최대 커널이 `update_vcost_ispd24` (10.98s), 그다음이 `update_wcost_cuda_ispd24` (6.02s),
  `compute_presum` (4.20s). 나머지 GPU 커널을 다 합쳐야 9.4s.
  → "원래 오래 안 걸린다"는 전제는 이 디자인·이 GPU에서 성립하지 않는다.
- **`full` → `incr` : wall 56.24 → 36.77s (−34.6%)**, GPU −18.52s, vcost+presum −18.79s.
  wall 감소와 GPU 감소가 거의 1:1 → 기존 −30~40%는 계측 아티팩트가 아니었다.
- `paper` → `incr` 는 −42.8%지만 여기엔 GPU-FLUTE 몫이 섞여 있다.
- `paper`와 `full`의 vcost+presum 총량이 21.19 vs 19.81s로 비슷하다. wcost-presum fusion은 일을
  옮겼을 뿐 (`paper` = wcost 6.02 + vcost 10.98 / presum 4.20, `full` = vcost 10.88 / presum 8.93 —
  presum이 wcost를 인라인으로 재계산). `full`이 `paper`의 공정한 대역이고, fusion 자체는
  non-incremental 조건에서 ~1.4s 이득이며 track 단위 skip을 가능하게 한다.
- 기존 `cudaEvent` 측정은 **Stage 1을 아예 안 보고 있었다**. `update_cost` 런치가
  **1467 = Stage 1의 601 + Stage 2의 866** 으로 갈린다 — Stage 1 몫 41%가 기존 표에 없었다.
  방식이 틀린 게 아니라 범위가 좁았고, 방향은 과소 보고였다.
- GPU-FLUTE의 이득은 **커널 표에 안 보인다**. GPU FLUTE 커널 합계는 0.08s뿐이고, 효과는
  `paper` 33.68s → `full` 26.73s 즉 **호스트 idle 6.95s 감소**로 나타난다.
  기존 RSMT 표의 13.7s → 7.27s (−6.4s)와 일치한다.

### 남은 문제 — 이제 병목은 호스트다

- `incr` 기준 wall 36.77s 중 **25.78s(70.1%)가 GPU 커널이 안 도는 시간**이다.
- 이 값은 최적화와 **무관하게 고정**이다 : `full` 26.73s → `incr` 25.78s.
- 즉 GPU 커널을 앞으로 아무리 더 줄여도 상한이 11s다.
- `cudaDeviceSynchronize` 7.43s / 19,599회, 커널 런치 51,681회, `cudaMemcpy` 2.19s / 10,588회.
- 다음 측정 : `./tools/nsys_profile.sh -d mempool_group -c incr --cpu` +
  요약의 "the router's own stage timing" 표로 어느 스테이지인지 좁힌다.

### 트레이스 검증은 필수

`paper` config를 처음 돌렸을 때 `compute_presum` 런치가 1305로 나왔다 — 실제 배치 수 1466보다
161 적다. `quick_exit()`이 CUPTI flush를 건너뛰어 트레이스 뒷부분이 **조용히 잘린** 것이고,
`tools/profiling_exit.h`로 고쳤다. 그 잘린 트레이스에서는 Stage 2 DP 런치가 8,328로 보여
"src가 논문보다 DP를 2배 돌린다"는 잘못된 결론이 나왔었다. 온전한 트레이스에서는
**`paper` 18,573 vs `incr` 18,729, DP 시간 4.69 vs 4.65s로 사실상 동일**하다.

→ 어떤 수치든 쓰기 전에 **런치 수를 배치 수와 대조**할 것.

---

## 다음 할 일 <sub>2026-08-20 미팅 기준</sub>

- **FLUTE 가속 파이프라인**
  - 현재 degree < 10 (CPU)와 degree ≥ 10 (GPU)이 순차 진행
  - CPU 또는 GPU 시간을 서로 숨기도록 오버랩
- **Augmented DAG depth 최소화**
  - center를 미리 찾아둬도 augment 시 center가 이동하는 문제
  - score 개선 : augment를 더 많이 생성하거나 다른 탐색 방법 추가
  - critical depth 자체를 줄이기 (평균 depth는 23% 감소했으나 runtime 변화 거의 없음)
- **호스트 병목 분석** : `incr` 기준 wall의 70%가 커널이 안 도는 시간. GPU 쪽에 남은 여지는 11s뿐
  - `./tools/nsys_profile.sh -d mempool_group -c incr --cpu` 로 어느 호스트 구간인지 확인
- **`mempool_cluster_ranking` nsys 재측정** : 위 표를 최대 디자인으로 확장

---

## 변경 이력

| 날짜 | 내용 |
| --- | --- |
| 2026-08-20 | 최적화 1·2·3 정리 (미팅 발표). 2번 critical path 분석, 3번 재측정 과제로 남김 |
| 2026-08-23 | 3번을 nsys로 재측정 — 기존 수치 확인. 호스트 병목·Stage 2 DP 런치 건을 새로 발견 |
