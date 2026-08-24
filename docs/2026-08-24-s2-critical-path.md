# 2026-08-24 — Stage 2 critical path와 runtime의 관계

- 목적 : 2번(Augmented DAG depth)이 멈춘 지점 규명 — "depth −23%인데 runtime 불변"
- 환경 : `gpu-5`, TITAN RTX (`sm_75`), 연속 실행, GPU batch gen on
- 방법 : `INSTANTGR_AUGMENTED_DAG_PROFILE=1` + tree-center off/`cpu` A/B
  (로그의 `critical path:` 줄 = batch별 depth 분포, `GPU profile:` 줄 = DP/traceback/commit 분해)

> 결론 : **tree-center는 실제로 작동한다** (phases −24/−31%, S2 GPU −0.56/−1.11s).
> runtime이 안 움직였던 건 효과가 없어서가 아니라 **① host BFS 비용(1.2/2.5s)이 이득을 상쇄**하고
> **② 지렛대의 상한 자체가 작기** 때문 (wall의 4~7%).

---

## 1. 구조 — depth 체인은 Stage 2 전용

- S2 GPU 루프는 batch마다 `cur_batch_depth`만큼 level별 커널을 순차 launch
  (`src/Lshape_route_detour.hpp:860` DP, `:871` traceback — traceback은 level마다 `cudaDeviceSynchronize()`)
- 직렬 phases = **Σ batch max depth** — batch 안에 깊은 net 하나만 있어도 체인은 그대로
  → net별 **평균** depth −23%가 runtime에 안 보였던 1차 원인
- S1은 batch당 커널 1개(net당 스레드가 트리 전체 순회, `src/Lshape_route.hpp:318`)라 depth 체인 없음
  — 실측으로도 S1 GPU route는 tree-center on/off 무관 (3.52 = 3.52s / 12.08 ≈ 12.21s)

## 2. A/B 실측

| Stage 2 | group `off` | group `cpu` | cluster `off` | cluster `cpu` |
| --- | ---: | ---: | ---: | ---: |
| 직렬 phases | 18,714 | 14,294 (**−23.6%**) | 56,960 | 39,595 (**−30.5%**) |
| p50/p90/p99/max | 4/63/81/109 | 4/46/59/71 | 131/225/272/294 | 88/149/169/180 |
| bottom-up DP | 3.523 s | 2.989 s | 9.676 s | 8.270 s |
| traceback | 0.273 s | 0.230 s | 0.781 s | 0.648 s |
| commit | 2.247 s | 2.249 s | 6.276 s | 6.439 s |
| remove+cost | 1.575 s | 1.568 s | 8.875 s | 9.091 s |
| level-node-visits | 28.28 M | 28.25 M | 95.74 M | 95.88 M |
| S2 GPU route | 8.09 s | 7.53 s | 26.86 s | 25.75 s |
| S1 DAG DFS | 1.65 s | 2.72 s | 7.09 s | 9.47 s |
| S2 preprocessing | 2.22 s | 2.66 s | 8.93 s | 9.44 s |
| **전체 wall** | **37.06 s** | 39.01 s | **140.26 s** | 141.61 s |

- score : group 397,594,483 ↔ 397,603,852 / cluster 1,780,695,551 ↔ 1,780,733,108 — 둘 다 노이즈 수준
- cluster `cpu`의 host prep +1.63s는 tree-center가 안 건드리는 구간 → 공유 서버 노이즈로 판단
- 분포 차이 : group은 **꼬리 지배형**(p50=4, 상위 10% batch가 phases의 ~37%), cluster는 **전 batch가 깊음**(p50=131)
  → cluster에서 tree-center 감소율이 더 큼

## 3. DP 비용 모델 — level 고정비가 절반 이상

노드 방문 수는 불변(±0.1%)인데 phases만 줄었으므로 두 런의 연립으로 분해 가능:

**DP 시간 ≈ a × level 수 + b × 노드 수**

| 디자인 | a (level 고정비) | b (노드당) | 고정비 몫 (`off` DP 기준) |
| --- | ---: | ---: | ---: |
| `mempool_group` | 121 µs | 45 ns | 2.26 s / 3.52 s = **64%** |
| `mempool_cluster_ranking` | 81 µs | 53 ns | 4.64 s / 9.68 s = **48%** |

- level당 평균 노드 1,511 / 1,681개 = 512스레드 블록 3개 — TITAN RTX 72 SM이 거의 빈 채로 돎
- a는 커널 launch 오버헤드(~5µs)의 15~25배 → launch가 아니라 **작은 grid의 커널 실행 지연 자체**
- **traceback per-level sync 가설은 기각** : sync 18,714회 포함 0.27s (level당 ~15µs)
  — nsys의 `cudaDeviceSynchronize` 7.43s는 sync 고정비가 아니라 DP 커널 완료 대기가 흡수된 것

## 4. 기존 가설 정리

| 가설 (optimizations.md §2) | 판정 |
| --- | --- |
| augment 시 center가 이동해 효과가 사라진다 | **기각** — RSMT 기준 depth 감소율(23.8/25.1%)과 augmented phases 감소율(23.6/30.5%)이 거의 일치. center 효과는 augment를 통과함 |
| 평균이 아니라 critical path가 지배한다 | **확인** — phases = Σ batch max depth, DP의 절반 이상이 level 수에 비례 |
| (미검토) 이득이 host 비용에 상쇄된다 | **신규 확인** — GPU 이득 0.56/1.11s < host BFS 1.19/3.88s (S1 몫은 단일 스레드 루프라 wall에 1:1) |

## 5. 상한과 우선순위

- level 고정비를 **전부** 없애도(DP+traceback) : group ~2.5s(wall의 7%), cluster ~5.4s(3.9%)
- 잡는 방법 : ① tree-center를 공짜로 만들기(아래), ② per-level 고정비 공략(persistent kernel / cooperative grid sync)
- **새 관찰 — cluster S2의 진짜 큰 덩어리는 depth 체인이 아님** :
  commit 6.28s(batch당 13.6ms) + ripup 4.74s + presum 3.80s = **14.8s** > DP 9.7s
  → S2 GPU 다음 타깃 후보

## 6. 후속 : tree-center 비용 절감 (leaf-peeling)

현재 `cpu_tree_center_root`(`src/database_cuda.hpp:878`)는 net마다 **순회 4회 + 힙 할당 8회** :

1. legacy root BFS — 순수 계측용 (profile 숫자에만 쓰임)
2. 노드 0 → 최원점 BFS
3. 최원점 → 반대 끝 BFS
4. parent 재구성 BFS (3번에서 parent를 같이 기록하면 제거 가능)

- 실측 6.5µs/net — 수십~수백 노드 트리에서 순회는 sub-µs이므로 **할당이 지배**
- 개선안 : degree 기반 **leaf-peeling**(잎을 라운드마다 제거, 마지막 1~2개 = exact center)
  - 순회 ~1회 비용, 경로 복원 불필요, degree는 `rsmt[i].size()`로 이미 있음
  - thread-local 버퍼 재사용으로 할당 0회, 계측 BFS는 profile env 뒤로 게이팅
  - 기존 `select_root_net`은 leaf들의 multi-source BFS라 exact center가 아님 (그래서 23% 갭이 존재)
- 예상 : host 비용 1.19/3.88s → ~0.2/0.5s → tree-center가 **순이득 전환** (group ~+0.4s, cluster ~+0.7s)
- 검증 : 이중 BFS의 `min_max_depth`와 peeling center의 depth 일치 대조

### 구현 결과 (같은 날) — 순이득 전환 확인, 기본 on 전환

- 구현 : `rsmt_tree_center_peel()` — thread-local 버퍼, 할당 0회.
  legacy 대비 통계와 이중 BFS 오라클은 `INSTANTGR_AUGMENTED_DAG_PROFILE=1` 뒤로 게이팅
- 알고리즘 검증 : 무작위 트리 20,396개(경로·스타·skew, ≤2,000노드)에서 peel center가 항상 radius 달성
- 통합 검증 (오라클 런, group) : `peel mismatch` 0건 / 126,006 + 81,968 net,
  S1 depth sum `1899146 -> 1069761` 기존과 자릿수까지 동일
  (S2의 미세 차이는 peel이 2-center 트리에서 반대쪽 center를 골라 S1 라우팅이 노이즈 수준으로 달라진 것)

| 프로덕션 A/B | group `off` | group `cpu` | cluster `off` | cluster `cpu` |
| --- | ---: | ---: | ---: | ---: |
| S2 직렬 phases | 18,714 | 14,389 (−23.1%) | 56,960 | 39,916 (−29.9%) |
| S2 GPU route | 5.80 s | **5.21 s** | 20.51 s | **18.79 s** |
| S1 DAG DFS | 1.43 s | 1.56 s | 6.06 s | 6.51 s |
| center-find | — | 0.184+0.073 s | — | 0.646+0.257 s |
| 전체 wall | 38.27 s | 34.69 s | 137.44 s | 134.53 s |

- center-find 단가 : **1.4~1.5µs/net** (기존 BFS 4.6~6.5µs의 ~4×)
- 귀속 수지 : group GPU −0.59 vs host +0.34 → **순이득 ~0.25s**,
  cluster GPU −1.72 vs host +0.48 → **순이득 ~1.2s** (S2 preproc +0.89는 서버 부하 노이즈로 판단;
  전부 귀속시켜도 부호 유지)
- score : group +2,875 / cluster +45,159 (+0.0025%) — 노이즈 수준
- → **기본 on 전환** : `INSTANTGR_TREE_CENTER` 미설정 시 켜짐, `0`으로 끔.
  GPU 근사 실험(`INSTANTGR_GPU_TREE_CENTER=1`)은 `INSTANTGR_TREE_CENTER=0`과 함께 써야 함

## 재현

```bash
cd run && env INSTANTGR_AUGMENTED_DAG_PROFILE=1 ./InstantGR.opt -cap $BENCH/mempool_group.cap -net $BENCH/mempool_group.net -out test.out |& tee s2_critpath_group.log
cd run && env INSTANTGR_AUGMENTED_DAG_PROFILE=1 INSTANTGR_TREE_CENTER=cpu ./InstantGR.opt -cap $BENCH/mempool_group.cap -net $BENCH/mempool_group.net -out test.out |& tee s2_critpath_group_center.log
```

- depth 분포(`critical path:` 줄)는 profile env 없이도 LOG 빌드에서 항상 출력됨
- 로그 : `run/s2_critpath_{group,cluster}[_center].log` (서버)
