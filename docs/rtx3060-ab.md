# RTX 3060 전체 A/B 매트릭스

- 측정일 : **2026-08-26**
- 환경 : `intern`, **RTX 3060 12GB** (`sm_86`), 단독 사용. nvcc 12.6.85 (`~/cuda-12.6`)
- 코드 : `main` @ `667bc12` (워킹 트리 클린, 미추적 파일은 이 매트릭스 스크립트뿐)
- 방법 : **바이너리 1개 + 런타임 토글**. `tools/ab_matrix.sh`가 실행, `tools/ab_summary.py`가 표 생성
- 원본 : `ab_results_0826_143856/` (`runs.csv`, `SUMMARY.md`, `ENV.txt`, 로그 60건)
- 이전 기록 : 2026-08-25-rtx3060-remeasure.md — **이 문서가 대체함**

> - `mempool_group` : base 92.47 s → opt **28.28 s (3.27×)**
> - `bsg_chip` 2.13× · `nvdla` 2.63× · `mempool_tile_rank` 1.40×
> - 절감의 **73%가 incremental vcost/presum 하나** (`mempool_group` 기준 +47.08 s)
> - **CPU/GPU 오버랩 = +1.90 s**(`mempool_group`) / +1.01 s(`bsg_chip`) — 전용 노브로 격리 측정 (§2)
> - **GPU batch generation은 `bsg_chip`에서 이득 0** (−0.02 s) — 디자인 의존 (§4)

---

## 0. 매트릭스 구성

노브 7개를 두 가지 방식으로 돌렸다. 같은 런 집합에서 두 표가 나온다.

| 방식 | config | 답하는 질문 |
| --- | --- | --- |
| **leave-one-out** | `loo-*` | opt에서 이 기여만 끄면 얼마나 느려지나 = **최종 형상에서의 마진** |
| **누적 사다리** | `l1`~`l5` | base에서 하나씩 켜면 = **스택 막대용 누적 절감** |

- 사다리 순서는 의존성이 강제한다 — `flute-overlap`은 GPU-FLUTE가 켜져야 의미가 있으므로 그 뒤에 온다
- `l5-bgen`(= opt에서 tree-center만 off)은 두 방식이 공유하므로 한 번만 실행
- 디자인 : `mempool_group` 전체 12 config × 3회, `bsg_chip` 앵커+LOO × 2회, `mempool_tile_rank`·`nvdla` 앵커 × 2회
- `mempool_cluster_ranking`은 제외 — 이 카드에서 opt가 GPU-FLUTE scratch OOM, base는 S1 GPU route가 17.8배로 이상 동작 (08-25 문서 §3)

### 측정 유효성 가드

- 매 런마다 **7개 노브를 전부 명시적으로 지정** — tcsh 세션에 남은 `setenv`가 조용히 뒤집는 것을 차단
- 런 직후 `config:` 줄과 `Generation (GPU)/(CPU)` 줄을 요청값과 대조 — **60/60 일치**
  (GPU batch generation은 `config:` 줄에 없어서 batch 표로 검증)
- 매 런 전 `nvidia-smi` 확인 — **60/60에서 다른 compute 프로세스 0개**
- 디자인 단위로 연속 실행, 반복 안에서 config를 로테이션 → 서버 드리프트가 특정 config에 몰리지 않음
- 총 60런 / 33분, 실패·불일치 0건

재현성(총 시간 min~max 폭) :

| 디자인 | 대부분 | 최악 |
| --- | ---: | --- |
| `mempool_group` | ≤ 1.3% | `l4-ovl` 5.5%, `loo-bgen` 4.7% |
| `bsg_chip` | ≤ 0.5% | `l5-bgen` 2.6% |
| `mempool_tile_rank` · `nvdla` | ≤ 1.2% | — |

---

## 1. 앵커 — base vs opt

`base` = 7개 노브 전부 off, `opt` = 전부 on. 같은 바이너리다.

| 디자인 | base | opt | 배율 |
| --- | ---: | ---: | ---: |
| `mempool_group` | 92.47 s | **28.28 s** | **3.27×** |
| `nvdla` | 8.45 s | 3.21 s | 2.63× |
| `bsg_chip` | 24.51 s | 11.52 s | 2.13× |
| `mempool_tile_rank` | 3.27 s | 2.34 s | 1.40× |

### `mempool_group`

- grid : `L=9 X=1782 Y=2417 cells=38,763,846 tracks=18,578`
- batch : `base` S1 601 / S2 865 · `opt` S1 603 / S2 871

| 항목 | `base` | `opt` | 개선 |
| --- | ---: | ---: | ---: |
| input | 4.67 | 4.67 | — |
| **Lshape route (S1)** | **41.68** | **10.90** | **−73.8%** |
| &nbsp;&nbsp;S1: RSMT (CPU+GPU 합) | 12.47 | 5.16 | −58.6% |
| &nbsp;&nbsp;S1: batch generation | 3.16 | 0.97 | −69.3% |
| &nbsp;&nbsp;S1: DAG build (DFS) | 1.32 | 1.45 | +9.8% |
| &nbsp;&nbsp;S1: GPU route batches | 24.34 | 2.88 | −88.2% |
| **DAG/detour route (S2)** | **43.89** | **10.47** | **−76.1%** |
| &nbsp;&nbsp;S2: detour generation | 1.82 | 1.65 | −9.3% |
| &nbsp;&nbsp;S2: batch generation | 3.08 | 1.30 | −57.8% |
| &nbsp;&nbsp;S2: host DAG prep + upload | 5.16 | 1.44 | −72.1% |
| &nbsp;&nbsp;S2: GPU route batches | 33.77 | 6.01 | −82.2% |
| finish nets + final output | (접힘) | 1.43 | — |
| other | 2.23 | 0.82 | −63.2% |
| **total** | **92.47** | **28.28** | **−69.4%** |

- S1 RSMT 행은 스케줄에 따라 이름이 갈린다(`(overlapped)` 1줄 vs `(serial)` 2줄) → 합산해 한 줄로 비교. 분해는 아래
- 2% 문턱 아래로 접힌 행은 0이 아니라 "접힘"으로 표기 — 두 config에서 접히는 행이 다르다

### `bsg_chip`

- grid : `L=9 X=1532 Y=2077 cells=28,637,676 tracks=15,968`
- batch : `base` S1 238 / S2 145 · `opt` S1 239 / S2 150

| 항목 | `base` | `opt` | 개선 |
| --- | ---: | ---: | ---: |
| input | 1.10 | 1.09 | −0.9% |
| wait for net split | 0.69 | 0.67 | −2.9% |
| **Lshape route (S1)** | **15.60** | **6.39** | **−59.0%** |
| &nbsp;&nbsp;S1: RSMT (CPU+GPU 합) | 4.47 | 1.61 | −64.0% |
| &nbsp;&nbsp;S1: batch generation | 0.68 | 0.44 | −35.3% |
| &nbsp;&nbsp;S1: DAG build (DFS) | 0.35 | 0.42 | +20.0% |
| &nbsp;&nbsp;S1: GPU route batches | 10.01 | 3.82 | −61.8% |
| **DAG/detour route (S2)** | **6.59** | **2.81** | **−57.4%** |
| &nbsp;&nbsp;S2: host DAG prep + upload | 0.70 | 0.32 | −54.3% |
| &nbsp;&nbsp;S2: GPU route batches | 5.61 | 2.04 | −63.6% |
| other | 0.54 | 0.56 | +3.7% |
| **total** | **24.51** | **11.52** | **−53.0%** |

### S1 RSMT 분해 — 오버랩 on/off

`loo-ovl`은 오버랩 노브만 끈 것이라 GPU/CPU FLUTE가 두 행으로 갈린다.

| 디자인 | serial (`loo-ovl`) | overlapped (`opt`) | 차이 |
| --- | --- | --- | ---: |
| `mempool_group` | GPU FLUTE 2.12 + CPU FLUTE 5.11 = 7.23 s | CPU FLUTE (overlapped) 5.16 s | **−2.07 s** |
| `bsg_chip` | GPU FLUTE 1.15 + CPU FLUTE 1.56 = 2.71 s | CPU FLUTE (overlapped) 1.61 s | **−1.10 s** |

- `opt` 로그의 GPU 측 기록 : `GPU wall=2.197s, tail beyond CPU loop=0.000s` (group) /
  `1.160s, 0.000s` (bsg) — GPU가 CPU 루프 뒤로 완전히 숨는다
- 전체 wall 기준 차이는 §2의 `loo-ovl` 행 (+1.90 s / +1.01 s)

---

## 2. 기여별 마진 — leave-one-out

opt에서 그 기여 하나만 껐을 때 늘어나는 시간.

| 기여 | config | `mempool_group` | 비중 | `bsg_chip` | 비중 |
| --- | --- | ---: | ---: | ---: | ---: |
| **incremental vcost / presum** | `loo-vcp` | **+47.08 s** | **73.3%** | **+8.72 s** | **67.1%** |
| GPU-FLUTE 전체 (알고리즘+오버랩) | `loo-flt` | +7.48 s | 11.7% | +2.81 s | 21.6% |
| &nbsp;&nbsp;— 알고리즘 단독 (파생) | | +5.58 s | 8.7% | +1.80 s | 13.9% |
| &nbsp;&nbsp;— CPU/GPU 오버랩 | `loo-ovl` | +1.90 s | 3.0% | +1.01 s | 7.8% |
| incremental wire-demand commit | `loo-cmt` | +6.89 s | 10.7% | +1.10 s | 8.5% |
| GPU batch generation | `loo-bgen` | +4.14 s | 6.4% | **−0.02 s** | — |
| tree-center + leaf peeling | `l5-bgen` | +0.55 s | 0.9% | +0.25 s | 1.9% |

- 비중 = 마진 / (base − opt). `mempool_group` 64.19 s, `bsg_chip` 12.99 s
- **알고리즘 단독 = `loo-flt` − `loo-ovl`** : `INSTANTGR_GPU_FLUTE=0`은 GPU 경로를 통째로 지워 알고리즘과 스케줄링을 섞으므로, 오버랩 몫을 빼야 알고리즘만 남는다
- 마진의 합(group 66.14 s)은 `base − opt`(64.19 s)와 일치하지 않는다 — 상호작용. 사다리(§3)와 대조할 것

---

## 3. 누적 사다리 — `mempool_group`

base에서 하나씩 켜 나간 것.

| 단계 | config | total | 이 단계 Δ | 누적 Δ |
| --- | --- | ---: | ---: | ---: |
| base (모두 off) | `base` | 92.47 | — | — |
| + incremental vcost/presum | `l1-vcp` | 46.29 | **−46.18 s** | −46.18 s |
| + incremental wire-demand commit | `l2-cmt` | 40.83 | −5.46 s | −51.64 s |
| + GPU-FLUTE (serial) | `l3-flt` | 34.88 | −5.95 s | −57.59 s |
| + CPU/GPU 오버랩 | `l4-ovl` | 32.83 | −2.05 s | −59.64 s |
| + GPU batch generation | `l5-bgen` | 28.83 | −4.00 s | −63.64 s |
| + tree-center | `opt` | 28.28 | −0.55 s | −64.19 s |

- 사다리 단계 Δ와 §2의 마진이 거의 같다 (예: 오버랩 −2.05 vs +1.90, batch gen −4.00 vs +4.14)
  → 이 기여들 사이의 상호작용이 작다는 뜻. 순서를 바꿔도 결론이 바뀌지 않는다
- 예외는 vcost/presum : 사다리에서 −46.18 s, 마진으로는 +47.08 s. 다른 최적화가 켜질수록 이 기여가 지우는 GPU 일의 비중이 커진다

---

## 4. GPU batch generation은 디자인을 탄다

| 디자인 | `loo-bgen` | `opt` | 마진 |
| --- | ---: | ---: | ---: |
| `mempool_group` | 32.42 s | 28.28 s | **+4.14 s** |
| `bsg_chip` | 11.50 s | 11.52 s | **−0.02 s** |

- `bsg_chip`의 −0.02 s는 그 config의 런 간 폭(0.5%, ±0.06 s) 안이다 → **이득도 손해도 아닌 무효**
- 스테이지로 보면 `bsg_chip`의 S1 batch generation은 0.68 → 0.44 s(−0.24 s)로 줄긴 한다.
  그런데 총 시간이 안 줄었다 = 그 절감이 다른 곳에서 상쇄됐다는 뜻 (batch 수 S1 238 → 239, S2 145 → 150)
- 이득은 batch generation이 wall에서 차지하는 비중이 큰 디자인에서만 나온다
  (`mempool_group` base 기준 S1 3.16 + S2 3.08 = 6.24 s, wall의 6.7%)

---

## 5. 품질

이번 매트릭스는 런타임만 재서 evaluator를 돌리지 않았다 (`SKIP_EVAL=1`). 아래는 **프로그램 자체 report**다
— evaluator와는 overflow 집계가 달라 한 표에서 섞으면 안 된다.

| 디자인 | `base` | `opt` | 차이 |
| --- | ---: | ---: | ---: |
| `mempool_group` | 397,601,464 (3회 평균) | 397,596,690 | −4,774 (−0.0012%) |
| `bsg_chip` | 112,468,928 (2회 평균) | 112,429,896 | −39,032 (−0.035%) |
| `mempool_tile_rank` | 13,772,197 | 13,773,858 | +1,661 (+0.012%) |
| `nvdla` | 47,979,948 | 47,981,478 | +1,530 (+0.003%) |

- 두 디자인 다 08-25 문서의 delta와 같은 자릿수 — 품질 변화 없음이 재확인됨
- **`bsg_chip`은 런 간 흔들림이 ±250 미만**이라 config별 delta가 노이즈가 아니라 신호다. 여기서 읽히는 것 :
  - **tree-center가 −34,163** (`l5-bgen` 112,464,059 → `opt` 112,429,896) — `bsg_chip` 품질 개선분 −39,032의 대부분이 tree-center 몫
  - CPU batch gen(`loo-bgen`)이 오히려 −5,367로 더 좋다 — 품질 차이는 있으나 절대값이 작다
- `mempool_group`은 런 간 ±1,500이라 config별 delta는 그 폭에 묻힌다 — 개별 값에 의미 두지 말 것
- **[ ] evaluator 검증은 별도로 돌릴 것.** `.out`은 4개 디자인 모두 `base`/`opt` 페어를 보존해 뒀다
  (`ab_results_0826_143856/*.r1.out`, 2.2 GB) — 그림 V2 혼잡도 히트맵에도 그대로 쓸 수 있다

---

## 6. 남은 것

- [ ] `optimizations.md` §4-2 등 TITAN 기준 절감 수치를 이 문서 값으로 교체
- [ ] evaluator로 품질 재확인 (§5)
- [ ] 7번 input 파싱 ∥ net 쪼개기는 **런타임 노브가 없어 이 매트릭스에서 빠졌다.**
      A/B가 필요하면 `INSTANTGR_NET_SPLIT_PIPELINE`을 추가하는 것이 맞다
      (커밋 간 빌드 비교는 그 사이 다른 변경분이 섞인다)
- [ ] `mempool_cluster_ranking` — 12GB 카드에서 opt가 안 돌아 이 매트릭스에 없다. 한계로 명시하거나 TITAN 결과 병기
- [ ] `mempool_tile_rank`가 1.40×로 유독 낮다 (grid 2.2M 셀). 작은 디자인에서 최적화 여지가 적은 것인지 확인
