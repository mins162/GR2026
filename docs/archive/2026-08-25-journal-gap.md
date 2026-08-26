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
| 4. **FLT (Flexible Layer Transition)** | §V | **구현 완료, 미병합** — `FLT` 브랜치, `INSTANTGR_FLT=1` opt-in |
| 5. 전체 라우터 (위 조합) | §VI | n/a |

→ **저널 기여 5개 모두 반영됨** (FLT는 별도 브랜치, main 미병합).

## FLT 결과 요약 — 품질 ↔ 런타임 트레이드오프

`mempool_group` 기준, 같은 바이너리에서 스위치만 토글 :

| 항목 | off | on | 차이 |
| --- | ---: | ---: | ---: |
| score | 397,657,854 | 396,641,942 | **−0.256%** (overflow 감소가 전부, via +0.31%) |
| 런타임 | 36.85 s | 41.23 s | **+11.9%** |

- 개선분의 근원은 overflow — FLT가 혼잡 구간을 wire 도중 층을 갈아타며 회피
- 논문은 이 벤치마크에서 −0.53%를 보고 — 우리 구현은 그 **절반 수준**. 원인 미규명
- 런타임 증가는 배치마다 새로 채우는 edge lookup table(Algorithm 3) 비용. 최초 구현 +15.7%에서
  precompute 커널 재작성으로 +11.9%까지 줄였으나 DP의 O(L)→O(L²) 열거는 알고리즘 본질 비용이라 못 줄임
- 품질 대비 런타임이 손해라 다른 최적화(전부 품질 무변화+속도 개선)와 방향이 반대 → **기본 off**로 opt-in
- 상세 : `FLT` 브랜치의 `docs/2026-08-25-flt.md` (미병합이라 main에는 없음)

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

논문 §VI(A800 기준) 예상치와의 대조는 위 "FLT 결과 요약"에 반영 — 논문은 `mempool_group`에서
−0.53% + case13 기준 edge precompute가 augmented routing의 22%를 보고, 우리 실측은 −0.256% /
런타임 +11.9%(precompute가 S2 GPU의 21%)로 precompute 비중은 논문과 근접, 품질은 절반 수준.

## 참고 — 미구현은 아니지만 논문과 다른 부분

- GPU batch gen 스케줄링 : 논문의 "batch 하나씩 닫는 commit-check 루프"는 실측에서 CPU보다 느려
  (S2 12.93s vs 3.50s) window/wavefront + warp-per-net으로 재설계함 — optimizations.md §4 1~5차
- 논문 실험 환경은 A800 + Xeon Gold 6326인데 우리는 TITAN RTX + 공용 CPU —
  runtime 수치는 직접 비교 불가, score만 대조 가능
