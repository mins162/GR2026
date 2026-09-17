# nsys 프로파일링 — vcost / presum 재측정 (2026-08-23)

> **결론 : 기존 측정이 맞았다.**
> - "vcost / presum은 원래 오래 안 걸린다"는 지적을 드라이버 쪽에서 재측정
> - 논문 원본에서 이 두 구간이 **wall의 33%, GPU 작업의 69%**
> - 오히려 기존 표는 Stage 1을 빼고 세어 **과소 보고**였음

- 관련 : [optimizations.md](optimizations.md) 3번, [2026-08-23-best-result.md](archive/2026-08-23-best-result.md)
- 브랜치 : `profile/nsys-vcost-presum`

---

## 1. 왜 다시 쟀나

- 기존 수치의 출처 : 코드가 직접 찍는 `cudaEvent` 쌍
- 걸리는 점 2가지
  - **측정이 대상을 바꿈** — 이벤트에서 숫자를 읽으려면 `cudaEventSynchronize()` 필요 → 큐를 비워 파이프라인 변형. Stage 1은 원래 sync가 없는 루프인데 계측을 켜면 배치마다 정지
  - **감싼 것만 봄** — 구간 정의가 잘못되면 그 오류가 그대로 숫자가 됨
- nsys는 CUPTI로 드라이버에서 직접 수신
  - sync 추가 없이 **커널별 실제 실행 시간·실행 횟수** 획득
  - → 이 질문의 정답지는 코드 계측이 아니라 nsys

---

## 2. 측정 조건

- `mempool_group`, TITAN RTX (sm_75), `cells=38,763,846`, 배치 = Stage 1 601 + Stage 2 866

| config | 무엇 |
| --- | --- |
| `paper` | `baseline/` 논문 원본 |
| `full` | `src/` + incremental vcost·presum **off** (매 배치 전체 재계산) |
| `incr` | `src/` + 둘 다 **on** (현재) |

- `full` / `incr`은 GPU-FLUTE를 양쪽 다 켜서 **차이가 vcost·presum 하나만** 남게 함

---

## 3. 결과

| config | wall | GPU busy | 커널 안 도는 시간 | vcost | presum | vcost+presum |
| --- | --- | --- | --- | --- | --- | --- |
| `paper` | 64.27s | 30.59s (47.6%) | 33.68s (52.4%) | 16.99s | 4.20s | **21.19s = wall의 33.0%** |
| `full` | 56.24s | 29.51s (52.5%) | 26.73s (47.5%) | 10.88s | 8.93s | 19.81s = wall의 35.2% |
| `incr` | 36.77s | 10.99s (29.9%) | 25.78s (70.1%) | 0.17s | 0.85s | 1.02s = wall의 2.8% |

읽는 법 :

- **논문 원본에서 vcost+presum이 GPU 작업의 69%**
  - 단일 최대 커널 `update_vcost_ispd24` 10.98s, 다음 `update_wcost_cuda_ispd24` 6.02s, `compute_presum` 4.20s
  - 나머지 GPU 커널 전부 합쳐야 9.4s
  - → "원래 오래 안 걸린다"는 이 디자인·이 GPU에서 미성립
- **`full` → `incr` : wall 56.24 → 36.77s (−34.6%)**
  - GPU −18.52s, vcost+presum −18.79s로 거의 1:1
  - 줄어든 커널 시간이 그대로 wall로 반영 → 기존 −30~40%는 계측 아티팩트 아님
- **기존 `cudaEvent` 표는 Stage 1 미포함**
  - `update_cost` 런치 1467 = Stage 1 601 + Stage 2 866
  - Stage 1 몫 41%가 표에 없었음 → 방식이 틀린 게 아니라 범위가 좁았고, 방향은 과소 보고
- **GPU-FLUTE 이득은 커널 표에 안 보임**
  - FLUTE 커널 합계 0.08s
  - 효과는 호스트 idle 감소로 발현 (`paper` 33.68 → `full` 26.73s)
- **`paper`와 `full`의 vcost+presum이 21.19 vs 19.81s로 유사**
  - wcost-presum fusion은 일을 옮겼을 뿐 (`paper` = wcost 6.02 + vcost 10.98 / presum 4.20 ↔ `full` = vcost 10.88 / presum 8.93, presum이 wcost를 인라인 재계산)
  - → `full`은 논문의 **공정한 대역**이지 논문 그 자체는 아님

### 다음 병목은 호스트

- `incr` wall 36.77s 중 **25.78s(70.1%)가 GPU 커널이 안 도는 시간**
- 이 값은 최적화와 무관하게 고정 (`full` 26.73 → `incr` 25.78s)
- → GPU 커널을 아무리 줄여도 상한이 11s
- 호스트 내역 : [optimizations.md](optimizations.md)의 breakdown 표

---

## 4. 다시 돌리는 법

```bash
./tools/nsys_profile.sh -d mempool_group
```

- config 3종을 모두 실행, `nsys_results_MMDD_HHMMSS/`에 트레이스·CSV·`SUMMARY.txt` 생성
- 자주 쓰는 옵션 : `-c incr -c full` (config 지정), `--cpu` (CPU 샘플링), `--no-build`

### tree-center : depth에 비례해서 줄어드는가

```bash
./tools/nsys_profile.sh -d mempool_group -c nocenter -c center
```

- `nocenter` / `center` = 최적화 전부 on, `INSTANTGR_TREE_CENTER`만 `0` / `cpu`
- S2는 batch마다 level 수만큼 `Lshape_route_node_cuda`를 순차 launch → **DP launch 수 = 직렬 depth phases** (Σ batch max depth)
- `SUMMARY.txt`의 `depth chain vs. time` 표 : launch 감소율 vs DP 시간 감소율, `ratio` = 시간 감소 / launch 감소
  - 1.0이면 level 고정비가 전부 (depth에 완전 비례), 0이면 노드 수만 (depth 무관)
  - 이전 cudaEvent 측정([2026-08-24](archive/2026-08-24-s2-critical-path.md))은 ratio 0.64 — nsys로 재확인하는 것이 목적
- `phases` 열(프로그램 자체 카운트)과 `DP lnch` 열이 다르면 트레이스 절단
- tcsh에서도 그대로 실행 가능, 경로는 env로 : `env BENCH=... ARCH=sm_75 ./tools/nsys_profile.sh -d mempool_group`

요약만 다시 보기 / 타임라인 보기 :

```bash
python3 tools/nsys_summarize.py nsys_results_0823_141500
```

```bash
nsys-ui nsys_results_0823_141500/mempool_group.incr.nsys-rep
```

주의할 것 :

- `INSTANTGR_AUGMENTED_DAG_PROFILE=1`과 **동시 사용 금지** — 이벤트 sync가 파이프라인을 바꿔 의심하던 왜곡이 재유입
- nsys 실행의 **wall clock을 성능 수치로 사용 금지** — wall 비교는 `run_ab_no_treecenter.sh` 또는 [measure-runtime.md](measure-runtime.md)
  - 이유는 오버헤드가 아니라 **커버리지** : 커널 표는 wall의 ~29%만 덮고, 호스트 70%(DAG 구성·CPU FLUTE·input·출력)는 애초에 안 나옴
  - 오버헤드 자체는 `mempool_group`에서 노이즈 수준 (nsys `incr` 36.77s vs 같은 시기 비-nsys 37.04 / 37.89s)
  - 단 런치가 수백만 건인 `mempool_cluster_ranking`에서는 오버헤드가 실제로 문제
- 트레이스 절단 가능성 → **런치 수를 배치 수와 대조**할 것 (`compute_presum` 런치 = S1 배치 + S2 배치)
- 큰 디자인(`mempool_cluster_ranking`)은 런치가 수백만 건 → 트레이스가 GB 단위, 결론이 갈릴 때만 실행
- `nsys`를 못 찾거나 버전 오류 시 `env NSYS=/full/path/to/nsys ...`로 지정
