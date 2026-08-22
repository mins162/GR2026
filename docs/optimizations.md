# InstantGR 최적화 정리

- 최종 업데이트 : **2026-08-22**
- 기준 자료 : 2026-08-20 미팅 발표 (`0820 논문.pptx`, 슬라이드 21~33)
- 대상 : ISPD 2024 global routing (InstantGR)
- 벤치마크 : `mempool_group` (3.2M net), `mempool_cluster_ranking` (10.6M net)

| # | 항목 | 상태 | 갱신일 |
| --- | --- | --- | --- |
| 1 | FLUTE 개선 (GPU-FLUTE) | 완료 | 2026-08-20 |
| 2 | Augmented DAG depth 개선 | depth ↓, runtime 변화 없음 → critical path 분석 예정 | 2026-08-20 |
| 3 | vcost / presum 계산 절감 | runtime·품질 양호, 시간 측정 방식 재검토 후 재측정 예정 | 2026-08-20 |
| 4 | GPU batch generation | 구현 완료, 서버 측정 전 | 2026-08-22 |

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

### 문제 및 다음 작업

- **선배님 피드백 : 시간 측정 방식이 잘못되었음(난 맞는거같은데)**
  - 위 수치는 그 측정 방식으로 얻은 값 → 그대로 신뢰할 수 없음
- 다음 작업 : **올바른 방식으로 재측정**
  - 비동기 GPU 커널 구간을 host wall clock으로 나눠 재는 부분 점검 (동기화 위치 확인)
  - CUDA event 기반 측정 / 전체 `total` 기준 비교로 재정리
  - 재측정 후 위 표 갱신
- 결론 자체(runtime 감소 + 품질 유지)는 유지되지만, 구간별 절감률 수치는 재측정 값으로 대체할 것

---

## 4. GPU batch generation <sub>2026-08-22</sub>

- 근거 논문 : InstantGR journal (TCAD 2026) Section III-D *GPU-Accelerated Batch Generation*
- 문제 : batch generation이 완전 순차 — net을 우선순위(hpwl 내림차순) 순으로 훑으며 충돌 없는 첫 batch에 넣음
- 접근 : batch를 하나씩 만들면서 남은 net **전부**를 병렬로 commit → check
  - **commit** : 각 net이 자기 mark를 batch map에 `atomicMin`으로 기록 → 우선순위가 높은(인덱스가 작은) net이 셀을 차지
  - **check** : 자기 point를 전부 자기가 소유한 net을 이번 batch에 배정
  - **rule out** : 이미 배정된 net에 막힌 net은 이번 batch를 포기하고 다음 batch로, 자기 mark는 회수
  - 더 배정할 net이 없을 때까지 commit-check 반복 → 다음 batch 시작
- 구현 위치 : `src/gpu_batch_gen.hpp`, 호출부는 `src/database_cuda.hpp`의 `generate_batches_rsmt()`
- 기하·판정은 CPU 경로 그대로
  - mark : h/v RSMT segment + point 주변 4셀 십자(`mark_3x3`)
  - 충돌 판정 : 자기 **point만** 검사 (segment가 남의 셀을 지나가는 것은 허용)
- batch 당 net 수 상한(Stage 1 `1,000,000` / Stage 2 `300,000`)은 후보가 아니라 **배정 수**에 적용
  - 후보를 상한으로 자르면 우선순위 뒤쪽 net이 batch에 못 들어와 batch 수가 크게 늘어남

### CPU 경로와의 차이

- 논문의 rule-out은 **배정된 net 전체**와 비교, CPU first-fit은 **자기보다 앞선 net**과만 비교
- 따라서 우선순위가 낮은 net이 먼저 배정되면 그보다 높은 net이 다음 batch로 밀릴 수 있음 → batch 구성이 동일하지는 않음 (논문도 "similar results")
- 두 경로 모두 유지하는 불변식 : **batch 안에서 어떤 net의 point도 자기보다 앞선 멤버의 mark에 덮이지 않는다**

### 영향 구간 (`mempool_group`, +RSMT 오버랩 기준)

| 구간 | 현재 | 영향 |
| --- | --- | --- |
| Stage 1 batch generation | 4.13s | 전량 대체 |
| Stage 2 전처리 + batch gen | 6.14s | batch gen 부분만 (`S2: RSMT/DAG preprocessing`은 그대로) |

- 논문 보고치 : CPU representative point exhaustion 대비 **약 3×** (Table II, 벤치 10~13)
- **서버 측정 아직 안 함** — 위 표는 대체 대상 구간이지 측정 결과가 아님

### 검증

- 호스트 시뮬레이션 (커널을 CPU에서 순차 실행, 무작위 net)
  - 모든 net이 정확히 한 batch에 배정 · batch 내 충돌 없음 · 상한 준수 · 재실행 시 동일 결과
  - batch 수는 상한이 안 걸리는 케이스에서 CPU first-fit과 **동일** (12/12, 31/31, 135/135, 94/94), 상한이 걸리면 +1
  - net별 배치는 위 차이 때문에 상당수 이동 → 실제 벤치마크에서 score 노이즈 확인 필요
- 실기 검증 : `INSTANTGR_GPU_BATCH_GEN_VALIDATE=1` (batch 충돌 검사 + CPU 결과와 batch 수/이동 net 수 비교)
- 폴백 : GPU 메모리 부족 등으로 실패하면 경고를 찍고 CPU 경로로 자동 복귀

### 다음 작업

- 서버에서 A/B 측정 (`INSTANTGR_GPU_BATCH_GEN=0` 과 비교), Stage 1/2 구간 시간과 ISPD score 기록
- batch map 메모리 : owner map이 `X*Y` int (최대 설계 기준 약 460MB) — 큰 설계에서 여유 확인
- commit-check 반복 횟수(로그의 `commit-check rounds`)가 크면 우선순위 처리 방식 재검토

---

## 다음 할 일 <sub>2026-08-20 미팅 기준</sub>

- **FLUTE 가속 파이프라인**
  - 현재 degree < 10 (CPU)와 degree ≥ 10 (GPU)이 순차 진행
  - CPU 또는 GPU 시간을 서로 숨기도록 오버랩
- **Augmented DAG depth 최소화**
  - center를 미리 찾아둬도 augment 시 center가 이동하는 문제
  - score 개선 : augment를 더 많이 생성하거나 다른 탐색 방법 추가
  - critical depth 자체를 줄이기 (평균 depth는 23% 감소했으나 runtime 변화 거의 없음)
- **최적화 3 재측정** : 위 3번 참고

---

## 변경 이력

| 날짜 | 내용 |
| --- | --- |
| 2026-08-22 | journal의 GPU batch generation 구현 (측정 전) |
| 2026-08-20 | 최적화 1·2·3 정리 (미팅 발표). 2번 critical path 분석, 3번 재측정 과제로 남김 |
