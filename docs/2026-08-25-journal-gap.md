# 2026-08-25 — 저널 논문 대비 미구현 항목

- 기준 논문 : `docs/papers/InstantGR(Journal).pdf` (TCAD 2026, vol.45 no.1 pp.441–452)
- 기준 코드 : `baseline/` = ICCAD'24 open source (`cuhk-eda/InstantGR @ cfce09f`) = 논문의 **InstantGR 1.0**
- 논문의 InstantGR **2.0** = 1.0 + 저널 신규 기법 → 우리 작업은 1.0에서 출발했으므로
  저널 기여 5개를 코드와 대조

## 기여별 상태

| 논문 기여 | 섹션 | 상태 |
| --- | --- | --- |
| 1. Segment 기반 routing graph + point exhaustion overlap checking | §III-B/C | **이미 1.0에 포함** — `generate_batches_rsmt_cpu()`의 `has_conflict`가 point 검사, mark는 h/v segment (`src/database_cuda.hpp:286`) |
| 2. GPU batch generation | §III-D | **완료** (2026-08-22, optimizations.md §4) — 단 논문의 배치별 commit-check 스케줄링은 실측 실패로 window/wavefront 구조로 대체, 우선순위 중재(atomicMin)는 논문대로 |
| 3. Node-level parallelism (depth별 병렬 DP) | §IV | **이미 1.0에 포함** — S2의 level별 커널 체인이 바로 이것 (`src/Lshape_route_detour.hpp:860`) |
| 4. **FLT (Flexible Layer Transition)** | §V | **미구현** — 코드에 layer-change-on-edge / edge precompute 흔적 없음 |
| 5. 전체 라우터 (위 조합) | §VI | n/a |

→ **남은 것은 사실상 FLT 하나.**

## FLT가 뭔가

- 기존 DAG 라우팅의 제약 : DAG의 두 노드를 잇는 wire segment는 **한 층에서 시작해 같은 층에서 끝나야** 함
- FLT : edge **중간에 층 변경 1회**를 허용 — 시작층 ≠ 도착층 연결이 가능해져
  혼잡 지역을 wire 도중에 층을 갈아타며 회피 (논문 Fig. 9 — 14개 segment 전부 혼잡인 상황도 통과)
- 구현 방식 (Algorithm 3, Edge Precompute) :
  1. edge마다 등간격 후보점 P = (p₀…pₙ) 보간 — 길이 d의 edge에 `max(⌊d/20⌋, 1)`개
  2. (시작층 l_start, 도착층 l_arr) 쌍마다 최적 층변경 지점과 비용을 GPU 병렬로 계산해 **lookup table** 저장
     (스레드 1개 = 층 쌍 1개, edge의 모든 후보점 순회)
  3. DP의 비용 계산에서 `wireCost(한 층)` 대신 `edgeCost(u→v, i→l)` 조회 — 층 열거가 O(L) → O(L²)이지만
     DP 자체가 이미 O(L²)라 복잡도 불변
- 층 변경은 edge당 **최대 1회**로 제한 (via 수 억제)

## 예상 효과와 비용 (논문 §VI, A800 기준)

- 품질 : 1.0 → 2.0 개선 **0.7%** 의 대부분이 FLT 몫 (batch gen은 runtime 담당)
- 우리 벤치마크의 Table III 대조 :

| 벤치마크 (= 논문 BM#) | 1.0 score | 2.0 score | 차이 | 우리 현재 (evaluator) |
| --- | ---: | ---: | ---: | ---: |
| `mempool_group` (BM 4) | 397,658,013 | 395,530,319 | **−0.53%** | 397,652,351 ≈ 1.0 |
| `mempool_cluster_ranking` (BM 12) | 1,780,897,390 | 1,771,528,815 | **−0.53%** | 1,780,854,733 ≈ 1.0 |

  — 우리 score가 1.0과 자릿수 수준으로 일치 = 품질 개선분(−0.5%)이 통째로 남아 있음
- runtime 비용 : case 13 기준 edge precompute가 **augmented routing의 22%, 전체의 7.1%**
  - 단 2.0 전체로는 1.0 대비 오히려 1.56× 빨라짐 (GPU batch gen 등이 상쇄)
  - 우리는 batch gen을 이미 흡수했으므로 FLT를 넣으면 **wall은 순증**할 것 — 품질 ↔ runtime 트레이드오프
- via 영향 : FLT는 층 변경을 늘리므로 via 증가 방향 — 논문 Table IV에서 2.0이 via/WL/OF 모두 1.0보다 개선이긴 함

## 구현 시 건드릴 곳 (예상)

- Stage 2 DP 비용 모델 : `src/Lshape_route_detour.hpp` DP 커널 — incoming node 비용 집계에 `edgeCost` 조회 추가
- edge precompute 커널 신설 + lookup table 메모리 (edge × L² — cluster grid에서 메모리 예산 확인 필요)
- traceback / commit : 층 변경 지점이 경로에 추가되므로 wire·via demand 기록 경로 수정
- 주의 : 3번 최적화(incremental vcost/presum)와의 상호작용 — edgeCost 재계산도 dirty 기반 증분으로 해야
  precompute가 batch마다 full rebuild로 돌아가지 않음

## 참고 — 미구현은 아니지만 논문과 다른 부분

- GPU batch gen 스케줄링 : 논문의 "batch 하나씩 닫는 commit-check 루프"는 실측에서 CPU보다 느려
  (S2 12.93s vs 3.50s) window/wavefront + warp-per-net으로 재설계함 — optimizations.md §4 1~5차
- 논문 실험 환경은 A800 + Xeon Gold 6326인데 우리는 TITAN RTX + 공용 CPU —
  runtime 수치는 직접 비교 불가, score만 대조 가능
