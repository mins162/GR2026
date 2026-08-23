# 2026-08-23 — 현재 최고 성능

- 환경 : `gpu-5`, TITAN RTX (`sm_75`), 연속 실행
- 코드 : 현재 `main` (최적화 1·3·4 반영)
- 측정 절차 : [measure-runtime.md](measure-runtime.md)

> - `mempool_group` : 62.38s → **31.36s (1.99×)**
> - `mempool_cluster_ranking` : 225.28s → **119.93s (1.88×)**
> - 품질 : evaluator 기준 −0.0021% / −0.0049% — 변화 없음
> - 감소분 귀속 §3 · 남은 병목 §4

---

## 1. `mempool_group`

- grid : `L=9 X=1782 Y=2417 cells=38,763,846 tracks=18,578`
- batch : `opt` S1 603 / S2 869 · `off` S1 601 / S2 865

| 항목 | `off` | `opt` | 개선 |
| --- | ---: | ---: | ---: |
| input | 4.61 s | 4.65 s | +0.9% |
| build CUDA database | 1.66 s | 1.63 s | −1.8% |
| **Lshape route (S1)** | **28.63 s** | **10.98 s** | **−61.6%** |
| &nbsp;&nbsp;S1: RSMT, CPU FLUTE (overlapped) | 12.50 s | 4.93 s | −60.6% |
| &nbsp;&nbsp;S1: batch generation | 3.56 s | 0.82 s | −77.0% |
| &nbsp;&nbsp;S1: DAG build (DFS) | 1.44 s | 1.36 s | −5.6% |
| &nbsp;&nbsp;S1: GPU route batches | 10.75 s | 3.47 s | −67.7% |
| **DAG/detour route (S2)** | **25.34 s** | **11.37 s** | **−55.1%** |
| &nbsp;&nbsp;S2: RSMT/DAG preprocessing | 1.79 s | 1.80 s | +0.6% |
| &nbsp;&nbsp;S2: batch generation | 3.69 s | 1.18 s | −68.0% |
| &nbsp;&nbsp;S2: host DAG prep + upload | 3.02 s | 2.61 s | −13.6% |
| &nbsp;&nbsp;S2: GPU route batches | 16.61 s | 5.73 s | −65.5% |
| finish nets + final output | 1.25 s | 1.23 s | −1.6% |
| close output + other | 0.88 s | 1.51 s | +0.63 s |
| **total** | **62.38 s** | **31.36 s** | **−49.7%** |
| *— GPU 커널 (nsys), 위 `GPU route batches` 내역* ⚠ | | | |
| &nbsp;&nbsp;vcost | 10.88 s | 0.17 s | −98.4% |
| &nbsp;&nbsp;presum | 8.93 s | 0.85 s | −90.5% |
| &nbsp;&nbsp;나머지 커널 (역산) | 9.70 s | 9.97 s | +2.8% |
| &nbsp;&nbsp;**GPU busy 합** | **29.51 s** | **10.99 s** | **−62.8%** |

⚠ nsys 블록 단서

- nsys 블록만 **다른 런** — `full`/`incr` config (GPU-FLUTE·batch gen 양쪽 on, vcost·presum만 변수), **4번 반영 전** 코드
- 두 계측은 서로 일치 : nsys vcost+presum 절감 **−18.79s** vs 호스트 타이머 `GPU route batches` 절감 **−18.16s** (차이 3.5%)
- `나머지 커널` 9.70 → 9.97로 불변 → 최적화가 vcost/presum만 건드렸다는 신호

### score

| 계측 | `off` | `opt` | 차이 |
| --- | ---: | ---: | ---: |
| 프로그램 자체 report | 397,599,647 | 397,591,407 | −8,240 (−0.0021%) |
| **evaluator** (ISPD 확정치) | **397,660,537** | **397,652,351** | **−8,186 (−0.0021%)** |

evaluator 내역:

| | WL | via | overflow |
| --- | ---: | ---: | ---: |
| `off` | 260,155,936 | 71,297,256 | 66,207,346 |
| `opt` | 260,156,140.5 | 71,298,796 | 66,197,414.6 |

- **품질 변화 없음 확정** — evaluator 기준 −0.0021%, 자체 report와 소수점 자리까지 동일한 결론
- `opt` : open nets 0 · incompleted nets 0
- 두 계측 차이는 **overflow 집계 방식**뿐 — WL·via는 자릿수까지 동일
- 차이가 `off` +60,890 / `opt` +60,944 로 **상수** → 어느 계측으로 재도 delta 동일
- 단 **한 표 안에서 섞지 말 것** (상수 6만이 그대로 개선분으로 잡힘)

<sub>`paper`(논문 원본 바이너리) 대조 : total 63.06s로 `off` 62.38s와 −1.1%, batch 수 601/865 동일,
Stage 1 score `406,296,750` 자릿수까지 동일. `off`가 논문 대역임이 확인되어 이후 표에서는 생략.
새 디자인 추가·구조 변경 시에만 재대조.</sub>

---

## 2. `mempool_cluster_ranking`

- grid : `L=9 X=4113 Y=5580 cells=206,554,860 tracks=42,885`
- batch : `opt` S1 383 / S2 460 · `off` S1 383 / S2 461

| 항목 | `off` | `opt` | 개선 |
| --- | ---: | ---: | ---: |
| input | 14.30 s | 14.22 s | −0.6% |
| build CUDA database | 6.18 s | 5.99 s | −3.1% |
| **Lshape route (S1)** | **117.50 s** | **46.18 s** | **−60.7%** |
| &nbsp;&nbsp;S1: RSMT, CPU FLUTE (overlapped) | 54.60 s | 20.44 s | −62.6% |
| &nbsp;&nbsp;S1: batch generation | 16.57 s | 5.43 s | −67.2% |
| &nbsp;&nbsp;S1: DAG build (DFS) | 6.05 s | 6.47 s | +6.9% |
| &nbsp;&nbsp;S1: GPU route batches | 38.56 s | 12.01 s | −68.9% |
| **DAG/detour route (S2)** | **79.94 s** | **44.66 s** | **−44.1%** |
| &nbsp;&nbsp;S2: RSMT/DAG preprocessing | 5.86 s | 7.07 s | **+20.6%** |
| &nbsp;&nbsp;S2: batch generation | 14.03 s | 8.83 s | −37.1% |
| &nbsp;&nbsp;S2: host DAG prep + upload | 8.57 s | 8.28 s | −3.4% |
| &nbsp;&nbsp;S2: GPU route batches | 51.26 s | 20.31 s | −60.4% |
| 출력 + other | 7.36 s | 8.88 s | +1.52 s |
| **total** | **225.28 s** | **119.93 s** | **−46.8% (1.88×)** |

- `off`에서는 `finish nets` / `close output`이 2% 문턱 아래라 `other`에 접힘 → 합산 비교

### score

| 계측 | `off` | `opt` | 차이 |
| --- | ---: | ---: | ---: |
| 프로그램 자체 report | 1,780,756,702 | 1,780,670,923 | −85,779 (−0.0048%) |
| **evaluator** (ISPD 확정치) | 1,780,901,267 | **1,780,813,614** | **−87,653 (−0.0049%)** |

- `opt` evaluator 내역 : WL 1,189,022,536.4 · via 267,483,444 · overflow 324,307,633.9
- `opt` : open nets 0 · incompleted nets 0

<sub>evaluator는 **직전 런 페어**의 `.out`에 실행, 위 표의 로그는 그 뒤 재실행분 (`.out`이 덮어써짐).
두 페어의 델타가 −85,779 / −87,653으로 일치하므로 결론 동일.</sub>

### 눈에 띄는 것

- **batch 수 사실상 동일** — S1 383 vs 383, S2 460 vs 461
  - 5차 튜닝(ring 2GB, wavefront 하한 256) 유지 확인 : `retired 0`, ring 748 슬롯, commits per net 2.6 / 5.5
- **GPU-FLUTE 완전히 숨음** — 고차수 net 453,136개, GPU wall 9.39s, CPU 루프 넘긴 tail **0.000s**
- **`S2: RSMT/DAG preprocessing`만 느려짐** (5.86 → 7.07s, **+20.6%**)
  - `mempool_group`에서는 1.79 ↔ 1.80으로 변화 없던 구간
  - Stage 2 RSMT 재구성은 **overlap 없음** (`src/Lshape_route_detour.hpp:541-550` — CPU 루프 완료 후 `construct_rsmt_gpu()` 호출) → GPU FLUTE 호출 비용이 그대로 노출
  - **단 이 구간은 서버 부하에 따라 편차가 매우 큼** — 같은 코드·같은 입력에 5.60s ↔ 19.19s(3.4배) 사례 존재 → +20.6%가 회귀인지 노이즈인지 반복 측정 필요
  - 원인 확정 전 (§5)

---

## 3. 감소분의 귀속

| 최적화 | 해당 항목 | `mempool_group` | `mempool_cluster_ranking` |
| --- | --- | ---: | ---: |
| incremental vcost / presum | S1 + S2 `GPU route batches` | **−18.16 s** (58.5%) | **−57.50 s** (54.6%) |
| GPU-FLUTE | S1: RSMT, CPU FLUTE | **−7.57 s** (24.4%) | **−34.16 s** (32.4%) |
| GPU batch generation | S1 + S2 `batch generation` | **−5.25 s** (16.9%) | **−16.34 s** (15.5%) |
| 그 외 (증가분 포함) | | −0.04 s | +2.65 s |
| | **실측 total 델타** | **−31.02 s** | **−105.35 s** |

- 두 디자인에서 비중 순서 동일 : vcost/presum > GPU-FLUTE > batch generation
- vcost/presum 몫의 상당 부분이 **Stage 1**에서 발생 (group −7.28 / cluster −26.55)
  - 기존 `cudaEvent` 표는 Stage 1을 미포함

---

## 4. 남은 병목 — 호스트 70~73%

| 남은 항목 | `mempool_group` | | `mempool_cluster_ranking` | |
| --- | ---: | ---: | ---: | ---: |
| | 시간 | wall % | 시간 | wall % |
| GPU route batches (S1+S2) | 9.20 s | 29.3% | 32.32 s | 27.0% |
| **DAG 구성** (S1 DFS + S2 preproc + S2 host prep/upload) | **5.77 s** | **18.4%** | **21.82 s** | **18.2%** |
| S1: RSMT, CPU FLUTE (degree < 10) | 4.93 s | 15.7% | 20.44 s | 17.0% |
| input 파싱 | 4.65 s | 14.8% | 14.22 s | 11.9% |
| batch generation (S1+S2) | 2.00 s | 6.4% | 14.26 s | 11.9% |
| 출력 (finish nets + close output) | 2.41 s | 7.7% | 7.04 s | 5.9% |
| build CUDA database | 1.63 s | 5.2% | 5.99 s | 5.0% |
| **호스트 합** | **22.16 s** | **70.7%** | **87.61 s** | **73.1%** |

- 두 디자인에서 순위 동일 : 1순위 **DAG 구성** (18.4% / 18.2%), 2순위 CPU FLUTE, 3순위 input
- GPU 커널 절감의 상한 : group 9.2s / cluster 32.3s → 다음 대상은 호스트

---

## 5. 다음 할 일

- [ ] **`S2: RSMT/DAG preprocessing` +20.6%** (cluster) — 회귀인지 노이즈인지 반복 측정 후 판정. Stage 2 RSMT는 overlap 없음
- [ ] **DAG 구성** — 두 디자인 공통 1순위 호스트 항목 (18%)
- [ ] GPU FLUTE 커버리지 확대 (현재 degree ≥ 10만)
- [ ] `bsg_chip`, `mempool_tile_rank` 측정
