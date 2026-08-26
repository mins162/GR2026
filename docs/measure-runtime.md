# runtime 측정 절차

- 결과를 낼 때마다 이 절차를 그대로 실행
- 결과는 날짜 붙인 문서에 병합 표로 기록 (예: [2026-08-23-best-result.md](archive/2026-08-23-best-result.md))

---

## config 2종

| config | 바이너리 | 설정 |
| --- | --- | --- |
| `opt` | `run/InstantGR.opt` | 4개 전부 on |
| `off` | `run/InstantGR.opt` | 4개 전부 off — **기준선** |

- `baseline/` 바이너리(`paper`)는 `input / Lshape Route / DAG Route / total` 4줄만 출력 → 항목별 비교 불가
- 따라서 같은 표를 찍는 `src/`를 최적화만 끈 `off`를 기준선으로 사용
- `paper`는 **`off`가 논문 대역인지 확인할 때만** 실행
  - 2026-08-23 `mempool_group` 대조 결과 : total −1.1%, batch 수 601/865 동일, Stage 1 score 자릿수까지 동일
  - 재대조 시점 : **새 디자인 추가 · 구조 큰 변경 시**

## 계측 2종

| 대상 | 계측 | 비고 |
| --- | --- | --- |
| wall clock, 호스트 구간 (input · FLUTE · DAG 구성 · batch gen · 출력) | `runtime_breakdown()` | 프로파일러 없이 |
| GPU 커널 (vcost · presum · DP · commit) | nsys (CUPTI) | `runtime_breakdown()`의 `GPU route batches` 내부 분해 |

- 배치 루프에 sync가 없어 호스트 타이머로는 커널 분리 불가
- `INSTANTGR_AUGMENTED_DAG_PROFILE=1`은 Stage 2만 보고 sync를 삽입해 파이프라인을 변형 → 사용 금지

---

## 커맨드

### 0. 준비

```bash
nvidia-smi
```

```bash
setenv CUDA_VISIBLE_DEVICES 0
```

```bash
setenv REPO /home/shkim/6_Internship/mskim/InstantGR-2026
```

```bash
setenv BENCH /home/shkim/6_Internship/mskim/benchmarks/benchmarks
```

```bash
setenv D mempool_group
```

```bash
mkdir -p $REPO/results_$D
```

### 1. 빌드

```bash
cd $REPO/src && nvcc main.cpp -o ../run/InstantGR.opt -std=c++17 -x cu -O3 -arch=sm_75
```

```bash
cd $REPO/baseline/src && nvcc main.cpp -o ../run/InstantGR -std=c++17 -x cu -O3 -arch=sm_75
```

- evaluator는 최초 1회만:

```bash
cd $REPO/run && g++ -O3 -std=c++17 -o evaluator evaluator.cpp && chmod +x ./evaluator
```

### 2. wall — 두 config를 연속으로

- `opt`도 명시적으로 `1`을 줄 것 — tcsh 세션에 남은 `setenv`가 조용히 끌 수 있음

```bash
cd $REPO/run && env INSTANTGR_GPU_FLUTE=1 INSTANTGR_INCREMENTAL_VCOST=1 INSTANTGR_INCREMENTAL_PRESUM=1 INSTANTGR_GPU_BATCH_GEN=1 ./InstantGR.opt -cap $BENCH/$D.cap -net $BENCH/$D.net -out $REPO/results_$D/opt.out |& tee $REPO/results_$D/opt.log
```

```bash
cd $REPO/run && env INSTANTGR_GPU_FLUTE=0 INSTANTGR_INCREMENTAL_VCOST=0 INSTANTGR_INCREMENTAL_PRESUM=0 INSTANTGR_GPU_BATCH_GEN=0 ./InstantGR.opt -cap $BENCH/$D.cap -net $BENCH/$D.net -out $REPO/results_$D/off.out |& tee $REPO/results_$D/off.log
```

- `off`가 논문 대역인지 대조할 때만:

```bash
cd $REPO/baseline/run && ./InstantGR -cap $BENCH/$D.cap -net $BENCH/$D.net -out $REPO/results_$D/paper.out |& tee $REPO/results_$D/paper.log
```

### 3. GPU 커널

```bash
cd $REPO && ./tools/nsys_profile.sh -d $D
```

- 결과 : `nsys_results_MMDD_HHMMSS/SUMMARY.txt`
- nsys의 `incr` / `full`이 각각 위 `opt` / `off`에 대응
- 단 nsys의 `full`은 GPU-FLUTE·GPU batch gen이 **켜져 있음** (vcost·presum만 변수) → 총량이 `off`와 다름, wall은 §2 값 사용

### 4. score

```bash
cd $REPO/run && ./evaluator $BENCH/$D.cap $BENCH/$D.net $REPO/results_$D/opt.out |& tee $REPO/results_$D/opt.eval
```

```bash
cd $REPO/run && ./evaluator $BENCH/$D.cap $BENCH/$D.net $REPO/results_$D/off.out |& tee $REPO/results_$D/off.eval
```

- **evaluator를 다시 돌리기 전에 `.out`이 덮어써지지 않았는지 확인** — 로그와 다른 런의 `.out`에 실행하면 페어가 어긋남
- 프로그램 자체 report와 evaluator는 overflow 집계가 달라 **한 표 안에서 섞지 말 것** (delta는 동일)

### 5. 표에 옮기기

```bash
grep -E "^config:|^grid:|s +[0-9.]+ %|^total|#Batches|Generation" $REPO/results_$D/off.log $REPO/results_$D/opt.log
```

- **config 확인**
  - `off` 런에 `gpu-flute=off`, `incremental-vcost=off`, `incremental-presum=off`가 찍혔는지
  - **GPU batch generation은 `config:` 줄에 없음** → `Generation (GPU)` / `Generation (CPU)` 줄로 확인 (`opt` → GPU, `off` → CPU)
  - 안 맞으면 두 열이 같은 실행
- 개선 칸 = `(off − opt) / off × 100`
- 들여쓴 행은 부모 행에 이미 포함 — 합산 금지
- 2% 미만 부모 행 / 1% 미만 자식 행은 출력에서 누락 → 두 config에서 접히는 행이 다르면 합쳐서 비교

---

## 병합 표 템플릿

```markdown
### <design>

- grid : `L=.. X=.. Y=.. cells=.. tracks=..`
- batch : `opt` S1 .. / S2 .. · `off` S1 .. / S2 ..

| 항목 | `off` | `opt` | 개선 |
| --- | ---: | ---: | ---: |
| input | | | |
| build CUDA database | | | |
| **Lshape route (S1)** | | | |
| &nbsp;&nbsp;S1: RSMT, CPU FLUTE (overlapped) | | | |
| &nbsp;&nbsp;S1: batch generation | | | |
| &nbsp;&nbsp;S1: DAG build (DFS) | | | |
| &nbsp;&nbsp;S1: GPU route batches | | | |
| **DAG/detour route (S2)** | | | |
| &nbsp;&nbsp;S2: detour generation | | | |
| &nbsp;&nbsp;S2: batch generation | | | |
| &nbsp;&nbsp;S2: host DAG prep + upload | | | |
| &nbsp;&nbsp;S2: GPU route batches | | | |
| finish nets + final output | | | |
| close output + other | | | |
| **total** | | | |
| *— GPU 커널 (nsys), 위 `GPU route batches` 내역 —* | | | |
| &nbsp;&nbsp;vcost | | | |
| &nbsp;&nbsp;presum | | | |
| &nbsp;&nbsp;나머지 커널 | | | |
| &nbsp;&nbsp;**GPU busy 합** | | | |

| 계측 | `off` | `opt` | 차이 |
| --- | ---: | ---: | ---: |
| 프로그램 자체 report | | | |
| **evaluator** | | | |
```

---

## 주의

- **연속 실행할 것**
  - 공유 서버라 간격이 벌어지면 호스트 구간이 흔들림 (같은 입력에 S2 detour generation 5.60s ↔ 19.19s, 3.4배 사례)
  - 두 런의 Stage 1 score가 자릿수까지 같으면 상태가 안정적이었다는 신호
- **`S2: detour generation`은 편차가 특히 큼** — 이 행의 증감은 1회 측정으로 판정 금지, 반복 측정 필요
- `nvidia-smi`로 GPU가 비었는지 확인 — 겹쳤으면 그 런은 폐기
- **nsys wall clock을 성능 수치로 사용 금지**
  - `mempool_group`에서는 오버헤드가 노이즈 수준 (36.77 vs 37.04s)
  - 런치가 수백만 건인 `mempool_cluster_ranking`에서는 다름
- nsys 트레이스는 **런치 수를 배치 수와 대조해 검증** — `quick_exit()`이 CUPTI flush를 건너뛰어 조용히 잘린 전례 ([profiling-nsys.md](profiling-nsys.md))
- 손으로 복사한 경우 실행 권한 소실 : `chmod +x $REPO/run/InstantGR.opt $REPO/run/evaluator $REPO/baseline/run/InstantGR $REPO/tools/nsys_profile.sh`
