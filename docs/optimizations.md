# InstantGR 최적화 정리

- 최종 업데이트 : **2026-08-25**
- 기준 자료 : 2026-08-20 미팅 발표 (`0820 논문.pptx`, 슬라이드 21~33)
- 대상 : ISPD 2024 global routing (InstantGR)
- 벤치마크 : `mempool_group` (3.2M net), `mempool_cluster_ranking` (10.6M net)

| # | 항목 | 상태 | 갱신일 |
| --- | --- | --- | --- |
| 1 | FLUTE 개선 (GPU-FLUTE) | 완료 | 2026-08-20 |
| 2 | Augmented DAG depth 개선 | **완료 — leaf peeling으로 순이득 전환, 기본 on** | 2026-08-24 |
| 3 | vcost / presum 계산 절감 | **nsys 재측정 완료 — 기존 수치 확인됨** | 2026-08-23 |
| 4 | GPU batch generation | `mempool_group` 전체 −11.9%, `mempool_cluster_ranking` −5.9% | 2026-08-22 |
| 5 | wire demand commit 증분화 | **완료 — 전체 −17.6%(`mempool_group`) / −8.8%(`bsg_chip`)** | 2026-08-25 |
| 6 | FLT (journal Sec. V) | **구현 완료, 미병합** — score −0.256%, 런타임 +11.9%(`mempool_group`) | 2026-08-25 |

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

### critical path 분석 (2026-08-24) <sub>결론 : 이득은 실재, host 비용이 상쇄</sub>

- 방법·수치 전체 : **[2026-08-24-s2-critical-path.md](archive/2026-08-24-s2-critical-path.md)**
- depth 체인은 **Stage 2 전용** (S1은 batch당 커널 1개) — 직렬 phases = Σ batch max depth
- tree-center는 phases를 **−23.6%(group) / −30.5%(cluster)** 줄이고 S2 GPU도 −0.56 / −1.11s 따라옴
  — 단 host BFS 1.19 / 3.88s(S1 몫은 단일 스레드 wall)가 상쇄해 전체 wall은 손해
- DP 시간 ≈ **level 고정비(81~121µs) × level 수 + 45~53ns × 노드 수** — 고정비 몫 48~64% (latency-bound)
- "augment가 center를 무효화" 가설 **기각** — RSMT depth 감소율과 phases 감소율 일치
- traceback per-level sync는 총 0.3s 미만 → sync 제거는 우선순위 탈락
- 지렛대 상한 : level 고정비 전부 제거해도 group ~2.5s / cluster ~5.4s (wall의 4~7%)
- **후속 완료 (같은 날)** : leaf peeling으로 재구현 — center-find 6.5 → 1.4µs/net, 오라클(이중 BFS 대조) 0건 불일치
  - 프로덕션 A/B : **순이득 전환** — group GPU −0.59 vs host +0.34, cluster GPU −1.72 vs host +0.48
  - → **기본 on** (`INSTANTGR_TREE_CENTER=0`으로 끔), 수치는 분석 문서 §6

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

- **runtime 감소, evaluate(ISPD score) 정상**
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

- **논문 원본에서 vcost+presum은 wall의 33%, GPU 작업의 69%**
  - 단일 최대 커널 `update_vcost_ispd24` 10.98s, 다음 `update_wcost_cuda_ispd24` 6.02s, `compute_presum` 4.20s
  - 나머지 GPU 커널 전부 합쳐야 9.4s
  - → "원래 오래 안 걸린다"는 전제는 이 디자인·이 GPU에서 미성립
- **`full` → `incr` : wall 56.24 → 36.77s (−34.6%)**
  - GPU −18.52s, vcost+presum −18.79s → wall 감소와 GPU 감소가 거의 1:1
  - → 기존 −30~40%는 계측 아티팩트 아님
- `paper` → `incr` 는 −42.8%지만 GPU-FLUTE 몫이 혼입
- `paper`와 `full`의 vcost+presum 총량 유사 (21.19 vs 19.81s)
  - wcost-presum fusion은 일을 옮겼을 뿐 (`paper` = wcost 6.02 + vcost 10.98 / presum 4.20, `full` = vcost 10.88 / presum 8.93 — presum이 wcost를 인라인 재계산)
  - `full`이 `paper`의 공정한 대역
  - fusion 자체는 non-incremental 조건에서 ~1.4s 이득, track 단위 skip을 가능하게 함
- 기존 `cudaEvent` 측정은 **Stage 1 미포함**
  - `update_cost` 런치 **1467 = Stage 1 601 + Stage 2 866**
  - Stage 1 몫 41%가 기존 표에 없었음 → 방식이 틀린 게 아니라 범위가 좁았고, 방향은 과소 보고
- GPU-FLUTE의 이득은 **커널 표에 안 보임**
  - GPU FLUTE 커널 합계 0.08s
  - 효과는 `paper` 33.68s → `full` 26.73s, 즉 **호스트 idle 6.95s 감소**로 발현
  - 기존 RSMT 표의 13.7s → 7.27s (−6.4s)와 일치

### 남은 문제 — 병목은 호스트

- `incr` 기준 wall 36.77s 중 **25.78s(70.1%)가 GPU 커널이 안 도는 시간**
- 이 값은 최적화와 **무관하게 고정** : `full` 26.73s → `incr` 25.78s
- → GPU 커널을 아무리 줄여도 상한이 11s
- `cudaDeviceSynchronize` 7.43s / 19,599회, 커널 런치 51,681회, `cudaMemcpy` 2.19s / 10,588회

호스트 27.2s의 내역 (`mempool_group`, `incr`, wall 37.04s 런)
— **4번(GPU batch generation) 반영 전 코드** :

| 구간 | 시간 | wall % | 현재 상태 |
| --- | --- | --- | --- |
| batch generation (S1 3.33 + S2 3.55) | 6.88s | 18.6% | **4번에서 해결됨** → 약 2.0s |
| DAG 구성 (S1 DFS 1.75 + S2 detour generation 2.09 + S2 host prep/upload 2.73) | 6.57s | 17.7% | 남음 |
| CPU FLUTE, degree < 10 (overlapped) | 4.94s | 13.3% | 남음 |
| input 파싱 | 4.79s | 12.9% | 남음 |
| 출력 (finish nets + close output) | 2.13s | 5.8% | 남음 |
| build CUDA database | 1.88s | 5.1% | 남음 |
| **호스트 합** | **27.19s** | **73.4%** | |
| GPU route batches (S1 3.23 + S2 5.82) | 9.05s | 24.4% | |

- 최대 호스트 항목이던 batch generation은 **4번이 GPU로 이전 완료** (S1 3.60 → 0.80s, S2 3.81 → 1.18s)
  - 프로파일이 가리킨 1순위와 실제 작업이 일치
- 이를 제외한 호스트 1순위 : **DAG 구성 6.57s**, 다음 CPU FLUTE 4.94s, input 파싱 4.79s
- DAG 구성은 세 스테이지에 분산되어 일괄 제거 곤란
- CPU FLUTE 4.94s는 이미 overlap 중 → 추가 이득은 GPU FLUTE 커버리지 확대 필요 (현재 degree ≥ 10만 GPU)
- **갱신 완료** : 4번 반영 후 재측정 결과는 [2026-08-23-best-result.md](archive/2026-08-23-best-result.md)

### 트레이스 검증은 필수

- `paper` config 첫 실행 시 `compute_presum` 런치 1305 — 실제 배치 수 1466보다 161 적음
- 원인 : `quick_exit()`이 CUPTI flush를 건너뛰어 트레이스 뒷부분이 **조용히 절단** → `tools/profiling_exit.h`로 수정
- 잘린 트레이스에서는 Stage 2 DP 런치가 8,328로 보여 "src가 논문보다 DP를 2배 돌린다"는 오결론 발생
- 온전한 트레이스 : **`paper` 18,573 vs `incr` 18,729, DP 시간 4.69 vs 4.65s로 사실상 동일**
- → 어떤 수치든 쓰기 전에 **런치 수를 배치 수와 대조**할 것

---

## 4. GPU batch generation <sub>2026-08-22</sub>

- 근거 논문 : InstantGR journal (TCAD 2026) Section III-D *GPU-Accelerated Batch Generation*
- 문제 : batch generation이 완전 순차 — net을 우선순위(hpwl 내림차순) 순으로 훑으며 충돌 없는 첫 batch에 넣음
- 구현 위치 : `src/gpu_batch_gen.hpp`, 호출부는 `src/database_cuda.hpp`의 `generate_batches_rsmt()`

### 1차 시도 : 논문 스케줄링 그대로 → **실패**

- batch를 하나씩 만들면서 남은 net **전부**가 자기 mark 전체를 commit → check → rule out

| 구간 (`mempool_group`) | CPU first-fit | 논문 방식 |
| --- | --- | --- |
| S1 batch generation | 3.15s | **5.51s** |
| S2 batch generation | 3.50s | **12.93s** |

- 구조 개선 실측 : 2차 S1 3.44 / S2 13.41s, 3차 S1 3.62 / S2 13.68s (아래 3·4차 참고)

- 원인 : net 하나가 "자기가 최종적으로 들어갈 batch 번호"만큼 full commit을 반복
  - S2는 net 197k에 batch 869개 → net당 평균 **~430회** commit, 거기에 batch당 commit-check 라운드 5.1회가 곱해짐
- CPU가 빠른 이유는 **실패가 싸기** 때문 : `has_conflict`가 point 몇 개 읽고 첫 충돌에서 리턴

### 2차 : 실패를 싸게 만든 구조 (현재 구현)

- 논문의 **commit-check 우선순위 중재**(`atomicMin`으로 높은 우선순위가 셀 선점 → 자기 point를 전부 소유한 net만 배정)는 유지
- batch를 하나씩 닫는 스케줄링만 CPU first-fit 구조로 회귀

- **window** : batch 여러 개를 동시에 열어두고 bitmap을 유지 → 새 batch뿐 아니라 예전 batch에도 들어갈 수 있음
- **wavefront** : 우선순위 앞쪽 일부만 in-flight. batch 크기에 맞춰 매 라운드 자동 조절 (배정 수의 2~8배)
- **pick** : 각 net이 자기가 멈춘 지점부터 열린 batch를 스캔 (막힌 batch는 계속 막힘) → 첫 번째로 point가 비어 있는 batch 선택. **실패는 bitmap 읽기 몇 번, commit 없음**
- **commit / check** : 고른 batch 하나에만 라운드당 1회 commit
- 열린 batch에 다 못 들어가면 새 batch를 열고, ring이 꽉 차면 가장 오래된 batch를 닫음

- 효과 (호스트 시뮬레이션, net 60k) : net당 commit **6~8회** — 논문 방식의 수백 회 대비 두 자릿수 배 감소

### 3차 : 라운드 오버헤드 제거

- 2차 실측(`mempool_group`) : S1 5.51 → **3.44s**(CPU 3.15), S2 12.93 → **13.41s**로 거의 불변
- net당 commit은 이미 2.8 / 5.1회로 저렴한데 라운드가 1847 / 2761회 → **라운드당 4.9ms**
- 커널 자체가 아니라 라운드 고정비용이 전부

- 원인 : `thrust::copy_if`가 호출마다 임시 버퍼를 `cudaMalloc`/`cudaFree` — `cudaFree`는 디바이스 전체를 동기화하고, 수 GB를 잡고 있는 상태에선 ms 단위
- 조치
  - 미배치 리스트 compaction을 **호스트에서** 수행 (status 다운로드 + 남은 net 업로드, 라운드당 수십 KB) → thrust 의존 제거
  - 항상 빈 batch 하나를 열어둬서 "새 batch 필요?" D2H 왕복 제거
  - pending 리스트는 head 오프셋으로 관리 → 라운드마다 tail을 복사하지 않음
- 라운드 수는 batch 수의 약 3배로 **구조적으로 고정**(wavefront 크기와 무관, 시뮬레이션에서 확인) → 라운드당 비용을 줄이는 것 외에 방법이 없음
- wavefront는 좁을수록 commit이 줄고 라운드 수는 그대로 → 하한 1024, 배정 수의 8배 초과 시 축소

### 기하·판정은 CPU 경로 그대로

- mark : h/v RSMT segment + point 주변 4셀 십자(`mark_3x3`)
- 충돌 판정 : 자기 **point만** 검사 (segment가 남의 셀을 지나가는 것은 허용)
- 불변식 : batch 안에서 어떤 net의 point도 **자기보다 앞선 멤버**의 mark에 덮이지 않음
- batch 당 net 수 상한(Stage 1 `1,000,000` / Stage 2 `300,000`)은 배정 수에 적용

### 검증

- 호스트 시뮬레이션 (커널을 CPU에서 순차 실행, 무작위 net, 최대 60k net)
  - 모든 net이 정확히 한 batch에 배정 · batch 내 충돌 없음 · 상한 준수 · 재실행 시 동일 결과
  - batch 수가 CPU first-fit과 거의 동일 (18/18, 152/152, 135/135, 94/94; 최악 +3)
- 실기 검증 : `INSTANTGR_GPU_BATCH_GEN_VALIDATE=1` — 1차 시도 실측에서 `conflict-free: yes`, batch 602 vs CPU 601 / 869 vs 868, score 397,604,233 (노이즈)
- 폴백 : GPU 메모리 부족·이상 상황이면 경고를 찍고 CPU 경로로 자동 복귀
- 로그 : `rounds N, X commits per net, wavefront W, K batches open at once (R retired early)`

### 4차 : net당 warp

- 3차 실측에서도 S2 13.68s로 불변 (S1 3.62s), 라운드당 **5.07ms**
- thrust 제거로 안 움직임 → 원인은 커널 내부

- 원인 : **커널이 net당 스레드 1개**. wavefront 1024면 스레드가 32워프뿐이라 (TITAN RTX는 SM만 72개) 지연이 하나도 안 숨겨짐
  - pick : net 하나가 열린 batch 871개를 **순차 의존 체인**으로 훑음 → 스레드당 수천 번의 dependent load
  - commit / check / release : augmented DAG net은 mark가 수백~수천 개인데 그걸 스레드 하나가 다 순회
- 조치 : pick / commit / check / stamp / release 전부 **net당 warp(32 레인)**
  - pick : 레인이 batch 32개를 동시에 테스트하고 `__ballot_sync`로 가장 낮은 빈 batch 선택 → 체인 길이 1/32
  - commit / stamp / release : 레인이 net의 segment·point를 나눠 처리 (전부 순서 무관 연산이라 결과 동일)
  - check : 레인이 point를 나눠 검사하고 ballot으로 합침
  - wavefront 1024 → 스레드 32k
- `kPickLanes = 1`로 두면 기존 순차 동작과 동일 — 호스트 시뮬레이션은 이 설정으로 검증

### 결과 — `mempool_group` (5차 반영 후)

| 구간 | CPU first-fit | GPU | 개선 |
| --- | --- | --- | --- |
| S1 batch generation | 3.60s | **0.80s** | −78% |
| S2 batch generation | 3.81s | **1.18s** | −69% |
| 전체 runtime | 37.89s | **31.55s** | −16.7% |

- batch 수 : S1 603 vs CPU 601, S2 869 vs 866
- **다운스트림 영향 없음** : 4차 시점 측정에서 S1 GPU route 3.47s (CPU 때 3.52s), S2 GPU route 5.71s (5.77s)
- ISPD score 397,595,645 — 노이즈 수준 (슬라이드 기준 397,601,526)
- 라운드당 비용 : S2 기준 5.07ms → **0.43ms**

### 결과 — `mempool_cluster_ranking` (main 대비, 같은 머신 상태)

| 구간 | main | GPU batch gen | 개선 |
| --- | --- | --- | --- |
| S1 batch generation | 16.33s | **6.68s** | −59% |
| S2 batch generation | 14.56s | **14.23s** | −2% |
| S2 GPU route | 20.75s | 22.11s | +1.36s (batch 474 vs 461) |
| 전체 runtime | 133.25s | **125.36s** | −5.9% |

- ISPD score 1,781,089,663 vs main 1,780,725,674 (+0.02%, 노이즈)
- **주의**
  - 공유 서버 → 다른 사용자와 겹치면 호스트 구간이 크게 흔들림
  - 실측 사례 : 같은 코드·같은 입력(Stage 1 결과 자릿수까지 동일)인데 S2 detour generation이 5.60s ↔ 19.19s로 3.4배 차이
  - 비교는 반드시 **연속 실행**으로

### 5차 : 큰 디자인의 Stage 2

- `mempool_cluster_ranking`의 S2만 이득 없음, 로그에 원인 노출

- `retired 101` — grid 20.6M 셀 → bitmap 하나가 2.58MB, 1GB 예산이면 ring이 374개인데 batch는 474개 필요 → 100개를 조기에 닫아 batch 수가 늘고(474 vs CPU 461) 그만큼 S2 GPU route가 +1.36s
  - → ring 예산 1GB → **2GB** (`free/8` → `free/4`)
- `16.8 commits per net` — wavefront 하한 1024에 배정이 라운드당 62개뿐이라 커밋의 94%가 헛일
  - → net당 warp로 바꾼 뒤론 256 net이면 이미 8k 스레드 → 하한 **1024 → 256**

### 결과 — 5차 반영 후 `mempool_cluster_ranking`

| 구간 | main | GPU batch gen | 개선 |
| --- | --- | --- | --- |
| S1 batch generation | 16.33s | **5.50s** | −66% |
| S2 batch generation | 14.56s | **8.90s** | −39% |
| S2 GPU route | 20.75s | 20.78s | 동일 (batch 460 vs 461) |
| 전체 runtime | 133.25s | **118.69s** | −10.9% |

- `retired 0`, `commits per net` 2.6 / 5.5, ring 748 슬롯
- ISPD score 1,780,669,528 vs main 1,780,725,674 (노이즈)
- 상세 : **[2026-08-22-opt.md](archive/2026-08-22-opt.md)**

### 다음 작업

- S2 라운드 수 : batch 수의 13.6배 (`mempool_group`은 3배) — 한 라운드에 batch를 여러 개 여는 방식 검토
- 나머지 디자인(`bsg_chip`, `mempool_tile_rank`) 측정
- `INSTANTGR_GPU_BATCH_GEN_VALIDATE=1` 로 정합성 재확인 (`INSTANTGR_GPU_BATCH_GEN=0` 과 비교) — 아직 안 함
- `retired`가 0이 아니면 ring이 부족한 것 → batch 수 증가 여부 확인 (`kRingBudgetBytes`)
- `commits per net`이 10을 넘으면 wavefront 조절 규칙 재검토

---

## 5. wire demand commit 증분화 <sub>2026-08-25</sub>

- 3번(vcost/presum)이 안 건드린 구간 : 배치가 새 route의 demand를 기록하는 `batch_wire_update`는
  격자 커버리지와 무관하게 매번 L×X×Y 전체를 prefix-sum → commit → clear
- 3번과 같은 골격 적용 : traceback이 실제로 쓴 셀의 track만 `pre_demand_track_dirty`로 마킹,
  마킹 안 된 track은 (한 번도 안 써서 값이 항상 0이므로) 세 단계를 통째로 스킵
- Stage 1·2 공용 경로라 절감이 양쪽에 다 반영 — 오히려 배치 수가 많은 Stage 1 쪽이 더 큼
- 스위치 : `INSTANTGR_INCREMENTAL_COMMIT` (기본 on)

### 결과 (RTX 3060, 같은 바이너리 스위치 토글)

| 디자인 | off | on | 개선 |
| --- | ---: | ---: | ---: |
| `mempool_group` | 35.11 s | 28.94 s | **−17.6%** |
| `bsg_chip` | 13.07 s | 11.92 s | **−8.8%** |

- `mempool_group` S2 commit 버킷만 보면 4.65s → 1.01s (−78%)
- score는 두 디자인 다 런간 노이즈 범위 안, open·incompleted 항상 0
- 상세 : **[archive/2026-08-25-incremental-commit.md](archive/2026-08-25-incremental-commit.md)**

---

## 6. FLT (Flexible Layer Transition, journal Sec. V) <sub>2026-08-25</sub>

- 저널 논문의 마지막 미구현 기여. DAG 두 노드를 잇는 wire가 한 층에서 시작·종료해야 하는 제약을
  풀어, edge 도중 층 변경 1회를 허용 — 혼잡 구간을 층을 갈아타며 회피
- `feat/journal-flt` 브랜치에 구현 완료, **main 미병합**. 스위치 `INSTANTGR_FLT` (기본 off — 아래처럼
  품질과 런타임이 반대 방향이라, 지금까지의 최적화(전부 무손실)와 성격이 다름)

### 결과 (`mempool_group`, 같은 바이너리 스위치 토글)

| 항목 | off | on | 차이 |
| --- | ---: | ---: | ---: |
| score | 397,657,854 | 396,641,942 | **−0.256%** (overflow 감소, via +0.31%) |
| 런타임 | 36.85 s | 41.23 s | **+11.9%** (최초 구현 +15.7%에서 precompute 재작성으로 단축) |

- 논문은 이 벤치마크에서 −0.53%를 보고 — 우리는 그 절반 수준. 원인 미규명
- 런타임 증가는 배치마다 새로 채우는 edge lookup table(Algorithm 3) 비용. DP의 O(L)→O(L²)
  층 열거는 알고리즘 본질 비용이라 더 줄지 않음
- 상세 : `feat/journal-flt` 브랜치의 `docs/2026-08-25-flt.md`, 격차 배경은
  [archive/2026-08-25-journal-gap.md](archive/2026-08-25-journal-gap.md)

---

## 다음 할 일 <sub>2026-08-20 미팅 기준</sub>

- **FLUTE 가속 파이프라인**
  - 현재 degree < 10 (CPU)와 degree ≥ 10 (GPU)이 순차 진행
  - CPU 또는 GPU 시간을 서로 숨기도록 오버랩
- **Augmented DAG depth 최소화**
  - center를 미리 찾아둬도 augment 시 center가 이동하는 문제
  - score 개선 : augment를 더 많이 생성하거나 다른 탐색 방법 추가
  - critical depth 자체를 줄이기 (평균 depth는 23% 감소했으나 runtime 변화 거의 없음)
- **현재 main 재프로파일** : 위 호스트 내역은 4번 반영 전 코드 기준이다. 4번이 들어간 상태로 다시 재서 갱신
- **DAG 구성 6.57s** : 4번을 빼면 남은 호스트 최대 항목. 세 스테이지에 흩어져 있음
- **`mempool_cluster_ranking` nsys 재측정** : 3번 표를 최대 디자인으로 확장

---

## 변경 이력

| 날짜 | 내용 |
| --- | --- |
| 2026-08-25 | 6번 FLT 구현 (`feat/journal-flt`, 미병합) — score −0.256% / 런타임 +11.9% |
| 2026-08-25 | 5번 wire demand commit 증분화 — 전체 −17.6%(`mempool_group`) / −8.8%(`bsg_chip`) |
| 2026-08-24 | 2번 critical path 분석 완료 — tree-center 이득 실재(host 비용에 상쇄), DP는 level 고정비 지배 |
| 2026-08-24 | tree-center를 leaf peeling으로 재구현 — 두 디자인 모두 순이득, 기본 on 전환 |
| 2026-08-23 | 3번을 nsys로 재측정 — 기존 수치 확인. 호스트가 wall의 70%임을 새로 확인 |
| 2026-08-22 | journal의 GPU batch generation 구현 (측정 전) |
| 2026-08-20 | 최적화 1·2·3 정리 (미팅 발표). 2번 critical path 분석, 3번 재측정 과제로 남김 |
