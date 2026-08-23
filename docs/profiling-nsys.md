# nsys 프로파일링 — vcost / presum 검증

- 작성 : 2026-08-23
- 목적 : "vcost / presum 계산이 원래 오래 걸리지 않는다"는 지적을 **드라이버 측 실측**으로 확인
- 브랜치 : `profile/nsys-vcost-presum`
- 관련 : [optimizations.md](optimizations.md) 3번 항목 (재측정 과제)

---

## 1. 왜 nsys인가

지금까지의 구간별 수치는 코드가 스스로 찍는 `cudaEvent` 쌍에서 나왔다. 두 가지 한계가 있다.

- **측정이 대상을 바꾼다** : 이벤트 쌍에서 숫자를 읽으려면 `cudaEventSynchronize()`가 필요하고, 이는 큐를 비워 파이프라인 자체를 바꾼다. Stage 1 배치 루프는 원래 sync가 하나도 없는 완전 비동기 루프인데, 계측을 켜면 배치마다 멈춘다.
- **계측이 감싼 것만 본다** : 이벤트 사이에 들어간 커널만 집계된다. 구간 정의가 잘못되면 그 오류가 그대로 숫자가 된다.

nsys는 CUPTI로 드라이버에서 직접 받는다. **커널 하나하나의 실제 GPU 실행 시간과 실행 횟수**를, 프로그램에 sync를 추가하지 않고 얻는다. 즉 이 질문의 정답지는 코드 계측이 아니라 nsys 쪽이다.

NVTX 구간(`-DINSTANTGR_NVTX`)은 보조 수단이다. `nvtxRangePush/Pop`은 마커만 쓰고 절대 sync하지 않으므로 파이프라인을 건드리지 않는다. 커널을 파이프라인 단계별로 묶어 보고, `S1/batch` · `S2/batch` 구간의 instance 수로 **배치 개수**를 바로 확인하는 용도다.

핵심 근거는 NVTX 없이도 나오는 `cuda_gpu_kern_sum` 표다.

---

## 2. 실행

```bash
cd /home/shkim/6_Internship/mskim/InstantGR && ./tools/nsys_profile.sh -d mempool_tile_rank
```

- 기본 config 3종을 모두 돌린다.
  - `incr` : `src/` + incremental vcost·presum **on** (현재 최적화본)
  - `full` : `src/` + 둘 다 **off** → 매 배치 전체 그리드 재계산 (논문 동작)
  - `paper` : `baseline/` 바이너리, 논문 원본 소스 그대로
- `incr` / `full`은 GPU-FLUTE를 양쪽 다 켜서, **차이가 vcost·presum 하나만** 남게 한다.
- 결과 : `nsys_results_MMDD_HHMMSS/` (`.nsys-rep` 트레이스, `nsys stats` CSV, `SUMMARY.txt`)

옵션:

```bash
./tools/nsys_profile.sh -d mempool_group -c incr -c full   # config 지정
./tools/nsys_profile.sh -d mempool_tile_rank --cpu         # CPU 샘플링 추가
./tools/nsys_profile.sh -d mempool_tile_rank --gpu-metrics # SM/DRAM 이용률 (권한 필요)
./tools/nsys_profile.sh --no-build                         # 빌드 생략
```

tcsh에서도 그냥 실행하면 된다 (스크립트 shebang이 bash라 커널이 알아서 bash로 띄운다). `BENCH` / `ARCH`는 env로 넘긴다:

```bash
env BENCH=/path/to/benchmarks ARCH=sm_75 ./tools/nsys_profile.sh -d mempool_group
```

요약만 다시 보려면:

```bash
python3 tools/nsys_summarize.py nsys_results_0823_141500
```

타임라인을 눈으로 볼 때:

```bash
nsys-ui nsys_results_0823_141500/mempool_tile_rank.incr.nsys-rep
```

### 시작은 작은 디자인부터

`mempool_cluster_ranking`은 커널 런치가 수백만 건이라 트레이스가 GB 단위로 커지고 nsys 오버헤드도 눈에 띈다. `mempool_tile_rank` → `mempool_group` 순으로 올라가면서 확인하고, 큰 디자인은 결론이 갈릴 때만 돌린다.

### 하지 말 것

- `INSTANTGR_AUGMENTED_DAG_PROFILE=1`과 **같이 켜지 말 것**. 이벤트 sync가 파이프라인을 바꿔서, 정확히 지금 의심하고 있는 그 왜곡을 다시 집어넣는다. 스크립트는 이 변수를 설정하지 않는다.
- nsys 실행의 **wall clock을 최적화 성능 수치로 쓰지 말 것**. nsys는 오버헤드가 있다. wall clock 비교는 `run_ab_no_treecenter.sh`로, 구간 분해는 nsys로 한다.

---

## 3. 결과 읽는 법

`SUMMARY.txt`가 config마다 이런 표를 찍는다.

```
  wall clock (program self-report) :    41.00 s
  GPU busy (sum of kernel time)    :    24.77 s   (60.4% of wall)
  batches per stage                : 812

  bucket                 GPU time     %GPU    %wall     launches
  DP route                 9.70 s    39.2%   23.7%        52000
  commit (demand)          7.52 s    30.4%   18.3%         2400
  presum (wire cost)       3.82 s    15.4%    9.3%          800
  vcost rebuild            0.33 s     1.3%    0.8%          800
```

보는 순서:

1. **`vcost rebuild` / `presum` 의 `%wall`** — 이 최적화가 지울 수 있는 시간의 상한. 여기가 결론이다.
2. **`launches`** — 커널 1회가 싸도 배치 수만큼 곱해진다. `full` config의 `update_vcost_ispd24` 1회 평균 시간 × 배치 수가 곧 전체 비용이다.
3. **`GPU busy` 대 `wall`** — GPU가 노는 비율. 50%를 크게 밑돌면 병목이 GPU 커널이 아니라 호스트 쪽이다.
4. **`vcost + presum, side by side`** — `full` → `incr`로 두 bucket이 얼마나 줄었고, 그게 wall 대비 몇 %인지.

### 판정 기준

`full`(또는 `paper`) config에서 `vcost rebuild + presum`의 `%wall`을 보고 판단한다.

| 결과 | 해석 | 할 일 |
| --- | --- | --- |
| 합쳐서 wall의 **5% 미만** | 선배님 말이 맞다. 기존 −30~40%는 측정 아티팩트 | optimizations.md 3번 수치를 nsys 값으로 교체, 최적화 우선순위 재조정 |
| 합쳐서 wall의 **20% 이상** | 기존 측정이 방향은 맞았다 | 논문/선배님 기준과 왜 다른지(디자인 크기·배치 수·GPU) 설명 붙여서 유지 |
| 그 사이 | 디자인마다 다르다 | 여러 디자인에서 돌려 표로 정리 |

그리고 **nsys 값과 `INSTANTGR_AUGMENTED_DAG_PROFILE=1` 값이 일치하는지**를 따로 확인한다.

- 일치 → 기존 측정 방식 자체는 문제없었고, 이견은 "이 디자인·이 GPU에서는 실제로 크다"는 사실 문제로 좁혀진다.
- 불일치 → 이벤트 계측이 틀렸다는 뜻. 어느 구간이 어긋나는지가 곧 원인이다.

### 그 커널이 원래 그만큼 걸리는 일인지 (roofline 체크)

`full` config의 `update_vcost_ispd24` 평균 1회 시간을 하드웨어 하한과 비교하면, "측정이 이상한 것"인지 "일이 원래 그만큼 큰 것"인지 갈린다.

- 전체 그리드 재계산은 셀당 `capacity`·`demand`(각 4B)를 이웃까지 읽고 `vcost`(4B)를 쓴다 → 셀당 대략 **16~20 B**
- 셀 수는 실행 로그의 `grid:` 줄에 찍힌다 (`cells=...`)
- 하한 = `cells × 18 B ÷ 대역폭`. TITAN RTX는 **672 GB/s**

예: `cells = 2.86e7` → `2.86e7 × 18 B ≈ 515 MB` → `≈ 0.77 ms` 가 1회 하한. 배치가 800개면 **0.6초**가 하한이다.

측정값이 이 하한 근처면 커널은 이미 대역폭 한계로 돌고 있고, 총량은 오직 **배치 수 × 그리드 크기**로 결정된다. 이 경우 "느리다/빠르다"는 논쟁이 아니라 배치 수를 확인하면 끝난다. 반대로 하한의 몇 배로 크게 나오면 커널 쪽에 따로 문제가 있는 것이다(occupancy, `__expf` 발산, managed memory 페이지 폴트 등).

---

## 4. CPU가 병목인지 (FLUTE가 논문보다 느린 건)

같은 트레이스로 답할 수 있다.

```bash
./tools/nsys_profile.sh -d mempool_tile_rank -c incr --cpu
```

- `SUMMARY.txt`의 **`GPU busy` / `wall`** 비율이 1차 지표. 이 값이 낮으면 GPU는 대기 중이고 호스트가 채우고 있다는 뜻이다.
- 타임라인(`nsys-ui`)에서 CUDA 행이 비어 있는 구간이 곧 CPU 구간이다. 어느 호스트 함수인지는 `osrt_sum` / 샘플링 결과가 알려준다.
- 결과 디렉터리의 `cpu.txt`(`lscpu`)에 서버 CPU 모델·클럭·코어 수가 저장된다. FLUTE의 CPU 구간(degree < 10)은 단일 스레드 성능에 직결되므로, 논문 환경과의 클럭 차이만으로도 차이가 설명될 수 있다.

여기서 GPU idle이 크게 나오면, 그건 실패가 아니라 **runtime breakdown이 제대로 나온 것**이다. 다음 최적화 대상이 GPU 커널이 아니라 호스트 구간(FLUTE CPU 경로, 배치 생성, 출력 스레드)이라는 결론이 된다.

---

## 5. 계측 구조

- nsys 바이너리는 스크립트가 고른다. 강제하려면 `NSYS` 환경 변수에 절대 경로를 준다.
- `src/nvtx_profile.hpp` — `-DINSTANTGR_NVTX` 없이는 모든 매크로가 사라진다. 평소 빌드는 완전히 그대로다.
- 구간 이름 : `S1/update_cost`, `S1/compute_presum`, `S1/DP`, `S1/commit`, `S1/batch`, `S2/ripup`, `S2/update_cost`, `S2/compute_presum`, `S2/bottom_up_DP`, `S2/traceback`, `S2/commit`, `S2/batch`
- 프로파일 빌드는 종료 시 `quick_exit()` 대신 정상 종료한다. CUPTI가 atexit에서 버퍼를 flush하는데 `quick_exit()`은 그걸 건너뛰어 트레이스 꼬리가 잘린다. 추가 시간은 마지막 측정 구간 뒤에 생기므로 숫자에는 영향이 없다.
- `runtime_breakdown()`에 `grid:` 줄이 추가됐다 (`L`, `X`, `Y`, `cells`, `tracks`). 위 roofline 계산에 필요하다.

---

## 6. 트러블슈팅

| 증상 | 원인 / 대응 |
| --- | --- |
| `Error: Nsight Systems X hasn't been installed with CUDA Toolkit Y` | `$CUDA/bin/nsys`가 버전 안 맞으면 거부하는 래퍼다. 스크립트가 `/opt/nvidia/nsight-systems/*/target-linux-x64/nsys` 등을 `--version`으로 확인해가며 자동으로 고른다 |
| `no working nsys found` | 설치된 사본이 없다. 경로를 알면 `env NSYS=/full/path/to/nsys ./tools/nsys_profile.sh ...` |
| `WARNING: no kernel rows for ...` | `quick_exit()`이 CUPTI flush(atexit)를 건너뛰어 트레이스가 잘린 것. `tools/profiling_exit.h`가 빌드 때 우회한다 |
| 트레이스가 조용히 잘림 | 위와 같은 원인의 부분 절단. **런치 수를 배치 수와 대조해서 검증할 것** — `compute_presum` 런치 = Stage 1 배치 + Stage 2 배치가 맞아야 한다. 안 맞으면 그 수치는 버린다 |
| NVTX 빌드 실패 | 스크립트가 자동으로 NVTX 없이 재빌드한다. 커널별 시간은 그대로 나오고 구간 묶음만 빠진다. 헤더는 `cuda-nvtx` 패키지(`<nvtx3/nvToolsExt.h>`) |
| `--gpu-metrics` 권한 오류 | 프로파일링이 admin 전용으로 잠겨 있음. `NVreg_RestrictProfilingToAdminUsers=0` 필요 → 관리자 문의. 이 옵션 없이도 결론은 난다 |
| 트레이스가 너무 큼 / 느림 | 더 작은 디자인으로. `--cpu`, `--gpu-metrics`를 빼면 오버헤드가 크게 준다 |
| `nsys stats` 리포트 이름 오류 | nsys 버전 차이. 스크립트가 신·구 이름을 모두 시도한다 (`cuda_gpu_kern_sum` / `gpukernsum`) |
| 다른 사람이 같은 GPU를 쓰는 중 | `nvidia-smi`로 빈 GPU 확인 후 `CUDA_VISIBLE_DEVICES` 지정. 공유 중이면 커널 시간이 부풀어 비교가 무의미 |

---

## 7. 결과 기록

돌린 뒤 여기에 표를 채우고, `optimizations.md` 3번의 수치를 교체한다.

| 디자인 | config | wall | GPU busy | vcost | presum | vcost+presum %wall |
| --- | --- | --- | --- | --- | --- | --- |
| | `paper` | | | | | |
| | `full` | | | | | |
| | `incr` | | | | | |
