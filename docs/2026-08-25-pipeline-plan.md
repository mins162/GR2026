# 2026-08-25 — 파이프라인 계획 (batch gen ∥ route)

- 상태 : **계획만. 아직 안 함.** 담당자 배정 예정
- 배경 : 현재는 `generate_batches_rsmt()`가 전부 끝난 뒤 route 루프 시작 (완전 순차)
- 아이디어 : 닫힌 batch를 순서대로 route에 흘려보내 gen과 route를 겹침
- 관련 : [optimizations.md](optimizations.md) §4 (GPU batch gen 1~5차), [2026-08-25-journal-gap.md](archive/2026-08-25-journal-gap.md)

## 핵심 판단 (2026-08-25 논의 결론)

- 파이프라인 총합 ≈ **max(gen, route)** — 느린 생산자는 파이프라인으로 못 숨김
- **신형(window/wavefront) gen은 파이프라인 부적합** : ring이 넉넉하면 batch가 끝까지 안 닫힘(`retired 0`이 정상).
  일찍 닫으면 batch 수 증가 → 5차에서 확인된 역효과 (cluster retired 101 → batch +13 → S2 route +1.36s)
- **논문(1차) 스케줄링은 구조적으로 이상적** : batch 0부터 순서대로 확정, 닫기 정책 불필요.
  단 1차 실측(S2 12.93s)은 **warp-per-net(4차) 이전** 측정 — 커널 개선과 스케줄링은 직교이므로 재측정 필요
- GPU 경합 주의 : 신형 gen은 0.43ms 소형 커널이라 route 빈틈에 끼울 수 있지만,
  논문 gen은 무거운 commit 커널이라 route와 경쟁함

## 실험 1 — 논문 스케줄링 + warp 커널 재측정 (선행, 가장 가치 큼)

- 내용 : 1차의 "batch 하나씩 닫는 commit-check 루프"에 4차 warp-per-net 커널(pick/commit/check/release)을 물려 gen 시간만 측정
- 왜 : net당 commit ~430회는 그대로지만 commit 1회 비용이 1/32 — 어디서 바닥 치는지 실측으로만 알 수 있음
- 판정 기준 (group S2 기준) :
  - gen ≤ **3s** → 파이프라인 총합 max(3, 5.73) < 현재 순차 6.9s → **실험 2로 진행**
  - gen > 5.7s → 파이프라인 무익, 중단
- 방법 : 호스트 시뮬레이션(`kPickLanes` 검증 인프라)으로 먼저 어림 → 실기 측정
- 산수 (group S2, 8-23 수치) :

| 구성 | 시간 |
| --- | ---: |
| 현재 순차 (신형 gen 1.18 + route 5.73) | 6.9s |
| 논문 gen(1차 실측) ∥ route | max(12.93, 5.73) = 12.9s ❌ |
| 논문 gen(+warp, 목표) ∥ route | max(≤3, 5.73) ≈ 5.7s ✅ |

## 실험 2 — 파이프라인 구현 (실험 1 통과 시)

- 생산자 : 논문 스케줄링 gen — batch i 확정 즉시 큐에 push
- 소비자 : route 루프 — 큐에서 순서대로 pop (routing 순서는 지금과 동일 → 품질 영향 없음)
- 정확성 근거 : gen은 기하만 보고 비용을 안 봄 → route의 commit이 gen 결과에 영향 없음.
  요건은 "넘긴 batch에 net 추가 금지" 하나뿐 — 논문 스케줄링은 자동 만족
- 구현 요소 : gen용 별도 CUDA stream + 호스트 스레드 1개, batch 큐
- 리스크 :
  - GPU 메모리 동시 상주 (ring 2GB + routing 자료구조) — cluster에서 빠듯할 수 있음
  - 논문 gen 커널이 route 커널과 SM 경쟁 — nsys로 route 구간 GPU 유휴율 먼저 확인
- 검증 : batch 구성 재현성(`INSTANTGR_GPU_BATCH_GEN_VALIDATE=1`), score 노이즈 수준, wall A/B 연속 실행

## 대안 (실험 1 실패 시) — 신형 gen + prefix-close

- window 유지하되 "K 라운드 동안 새 배정 없는 앞쪽 batch"만 보수적으로 닫아 route에 공급
- 선행 측정 : 호스트 시뮬레이션으로 batch 증가량 — **+1% 이내면 진행**, 5차 수준(+3%)이면 정책 재설계
- 이득 상한 : gen 시간만큼 (group ~2s, cluster ~14s = wall의 5~10%)

## 다른 구간의 파이프라인 후보 (추가 메모)

같은 생산자-소비자 패턴을 적용할 수 있는 곳. batch gen보다 리스크 낮은 것부터 :

| 후보 | 구간 | 규모 (group/cluster) | 비고 |
| --- | --- | --- | --- |
| **S1 route ∥ S2 detour generation** | S2 detour generation이 overlap 없음 (`src/Lshape_route_detour.hpp:541-550`, 8-23 확인) | ~1.8 / 7.1s | 닫기 정책 문제 없음 — **가장 안전한 첫 적용처** |
| input 파싱 ∥ CUDA DB build | 파싱 끝나야 build 시작 | 4.7+1.6 / 14.2+6.0s | net 단위 스트리밍 파싱 필요, 파서 구조 확인부터 |
| S2 host DAG prep/upload ∥ 직전 batch route | batch별 host 준비를 미리 | 2.6 / 8.3s | 이미 일부 겹치는지 코드 확인 필요 |

- 공통 원칙 : 겹치는 두 쪽이 CPU↔GPU면 이득이 크고, GPU↔GPU면 유휴율부터 nsys로 확인할 것
- 호스트가 wall의 70%(8-23 확인)이므로 CPU↔GPU 페어를 우선

## 할 일 순서 (담당자용)

1. [ ] 호스트 시뮬레이션으로 논문 스케줄링 + warp 어림 (실기 불필요, 로컬 가능)
2. [ ] 실기 : 실험 1 gen 시간 측정 (group → cluster)
3. [ ] nsys로 S1/S2 route 구간 GPU 유휴율 측정 (실험 2와 대안 공통 선행)
4. [ ] 판정 기준에 따라 실험 2 또는 대안 진행
5. [ ] (독립) S1 route ∥ S2 detour generation 오버랩 — 위 표의 1순위, 병행 가능
