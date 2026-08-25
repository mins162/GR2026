# 2026-08-25 — wire demand commit의 dirty-track 증분화

- 배경 : vcost/presum을 증분화([optimizations.md](optimizations.md))할 때 commit(`batch_wire_update`)은
  그대로 뒀던 부분. FLT 런타임을 분석하다 이 버킷이 S2 GPU의 상당 몫(사용 GPU 기준 수 초)임을 재확인해 착수
- 브랜치 : `opt/incremental-commit` (FLT와 독립 — `main`에서 분기, 건드리는 파일이 겹치지 않아 이후 자유롭게 병합 가능)
- 스위치 : `INSTANTGR_INCREMENTAL_COMMIT` (**기본 on**)

> - `mempool_group` : **35.08s → 28.91s (−17.6%)**, S2 commit **4.65s → 1.01s (−78%)**
> - `bsg_chip` : **13.14s → 12.00s (−8.7%)**
> - 절감의 상당 부분이 S1에서 나옴 (S1 GPU route 5.16 → 2.89s) — `batch_wire_update`가
>   Stage 1에서도 호출되고, 배치 수가 많아 full-grid 낭비가 더 컸던 것으로 추정
> - 품질 무변화 (런간 노이즈 범위), open 0 · incompleted 0 항상

---

## 1. 왜 남아 있었나

vcost/presum 최적화의 논리는 "입력이 안 바뀐 **파생 데이터**의 재계산은 건너뛴다"였다.
commit은 성격이 다르다 — 새 route의 demand를 **쓰는** 단계라 그 자체를 스킵할 순 없다.
하지만 그 쓰기를 실어 나르는 파이프라인의 세 단계는 배치가 격자의 일부만 건드려도 전체를 돈다 :

1. `compute_presum_general` — `pre_demand` 전체 track prefix-sum
2. `commit_all_edge` — L×X×Y **전체 격자**를 순회하며 demand에 반영
3. `cudaMemset(pre_demand, 0, ...)` — 전체 격자 클리어

presum에서 없앤 것과 같은 "배치당 full-grid 고정 비용" 패턴이 demand 쪽에만 남아 있었다.

## 2. 구현 — presum과 같은 dirty-track 골격

| presum(기존) | commit(신규) |
| --- | --- |
| ripup/commit이 바뀐 셀을 `track_dirty`에 기록 | traceback이 `pre_demand`에 쓰는 그 지점(`atomic_add_unit_demand_wire_segment`)에서 `pre_demand_track_dirty` 마킹 |
| `compute_presum`이 깨끗한 track은 즉시 return | `compute_presum_general`에 `dirty` 마스크 인자 추가, 깨끗하면 즉시 return |
| — | `commit_dirty_tracks` 신설 — track 단위로 `commit_all_edge` + `cudaMemset`을 대체 |

`commit_dirty_tracks`는 track당 블록 1개로, 그 track이 안 건드려졌으면 아예 커널 바디에 안 들어간다.
안건드려진 셀은 `pre_demand`가 항상 0이라 스킵이 정확하다는 게 presum보다 오히려 단순하다 —
presum은 "재사용해도 되나"를 판단하지만 이건 "쓴 적 없으니 볼 필요 없다"는 사실 자체다.

commit 안에서 neighbor cell의 prefix sum을 읽으므로, 클리어(`for(... = 0)`)는 `__syncthreads()`
뒤에 배치 — 같은 track 안에서 읽기가 전부 끝난 다음에 지운다.

마킹은 **세그먼트가 아니라 실제로 쓴 셀 단위**로 한다 (`mark_pre_demand_track`). 세그먼트 전체가
한 track에 있다는 가정에 기대지 않기 위함 — S1 쪽 호출부(`Lshape_route.hpp`)는 그 가정을
assert하지 않는 경로가 있어서, 셀 단위 마킹으로 그 전제 자체를 없앴다.

## 3. 결과

같은 바이너리 `run/InstantGR.commit`, 스위치만 토글. 로그 : `run/logs-commit/`

### 프로파일 분해 (`mempool_group`, `INSTANTGR_AUGMENTED_DAG_PROFILE=1`)

| 구간 | off | on |
| --- | ---: | ---: |
| commit | 4.651 s | **1.014 s (−78%)** |
| S2 GPU 합계 | 9.995 s | **6.259 s (−37%)** |

### clean wall

| 디자인 | off | on | 개선 |
| --- | ---: | ---: | ---: |
| `mempool_group` | 35.11 s | **28.94 s** | **−17.6%** |
| `bsg_chip` | 13.07 s | **11.92 s** | **−8.8%** |

- S1 GPU route가 가장 크게 줄었다 (`mempool_group` 5.18 → 2.88s, `bsg_chip` 4.51 → 3.82s) —
  `batch_wire_update`는 S1(`Lshape_route.hpp:320`)에서도 불리는데, S1은 배치 수가 더 많고
  배치당 커버리지가 더 작아 절대 낭비가 더 컸던 것으로 보인다
- S2 host DAG prep + upload도 줄었다 (`mempool_group` 5.04 → 1.42s) — 이 구간이 재는 건
  cudaMemcpy 자체가 아니라 **직전 배치의 GPU 완료 대기**라, commit이 짧아지면 여기로 흡수돼 보인다

### 품질

| 디자인 | off | on |
| --- | ---: | ---: |
| `mempool_group` | 397,658,864 | 397,659,360 |
| `bsg_chip` | 112,435,752 | 112,435,793 |

두 값 다 런간 노이즈(±3,650, [08-25 재측정](2026-08-25-rtx3060-remeasure.md) §1) 안. 전 런 open 0 · incompleted 0.

## 4. 남은 것

- [ ] presum처럼 full-grid 결과와 대조하는 `VALIDATE` 스위치는 아직 없음 —
      스킵 판정이 "마킹된 적 없는 track은 전부 0" 하나뿐이라 presum보다 단순하지만,
      확실히 하려면 추가할 것
- [ ] 반복 측정 없이 각 1회 — 노이즈 범위 재확인 필요
- [ ] `mempool_tile_rank` 미측정
- [ ] `feat/journal-flt`와 병합 시 재측정 — FLT의 precompute·traceback도 이 track 상태를
      건드리므로 조합 효과 확인 필요
