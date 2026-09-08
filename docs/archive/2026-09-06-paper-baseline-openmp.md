# 논문 baseline × OpenMP × 최대 opt — 7개 디자인 (RTX 3060)

- 논문 baseline(`baseline/src`)을 `-Xcompiler -fopenmp` 유무 두 가지로 빌드 (`baseline/run/InstantGR.omp`, `.noomp`)
- `opt` = `run/InstantGR.opt`를 `-fopenmp`로 빌드, 노브 전부 on, GPU-FLUTE degree ≥ 10 (기본)
- `mempool_cluster_ranking`은 논문 OpenMP 판이 GPU 11.9 GB에서 오래 걸려 중단, 미측정
- 단위 s, 각 프로그램 자체 타이머. 논문 바이너리는 input/Lshape/DAG/total 4줄만 출력하므로 구간 경계가 `opt`의 `runtime_breakdown()`과 정확히 같지는 않음
- 배경 : [gpu-flute-all-degrees](2026-09-06-gpu-flute-all-degrees.md) — 리포의 모든 빌드 명령에 `-fopenmp`가 없었다는 발견

| design | paper 비OpenMP | paper OpenMP | opt (OpenMP) | 배율 vs 비OpenMP | 배율 vs OpenMP |
| --- | --- | --- | --- | --- | --- |
| mempool_tile | 3.02 | 2.57 | 2.22 | 1.36× | 1.16× |
| mempool_tile_rank | 3.39 | 2.85 | 2.33 | 1.45× | 1.22× |
| ariane133_51 | 3.73 | 3.28 | 2.06 | 1.81× | 1.59× |
| ariane133_68 | 3.37 | 2.98 | 2.12 | 1.59× | 1.41× |
| nvdla | 8.81 | 8.22 | 3.12 | 2.82× | 2.63× |
| bsg_chip | 25.39 | 21.92 | 10.72 | 2.37× | 2.04× |
| mempool_group | 96.26 | 86.18 | 24.83 | 3.88× | 3.47× |

구간별 (Lshape / DAG) :

| design | paper 비OpenMP | paper OpenMP | opt |
| --- | --- | --- | --- |
| nvdla | 3.98 / 3.94 | 3.37 / 3.95 | 1.03 / 1.25 |
| bsg_chip | 15.99 / 6.68 | 12.46 / 6.73 | 5.49 / 2.88 |
| mempool_group | 43.41 / 45.21 | 32.78 / 45.70 | 7.09 / 10.99 |

- 논문 baseline의 OpenMP 효과는 Lshape 구간에만 있음 (DAG·input 동일). `#pragma omp`가 Stage 1 CPU FLUTE 루프 하나뿐이라서
- 논문 baseline에 OpenMP를 붙이면 배율이 디자인마다 0.2~0.4× 내려감. `mempool_group` 3.88× → 3.47×
- 보고서에 어느 쪽을 baseline으로 쓸지 결정 필요 : 원저자 README 빌드(비OpenMP)가 "논문 그대로"지만, 심사에서 `-fopenmp` 한 줄 지적을 받을 수 있음. OpenMP 판 배율을 병기하는 게 안전
- 로그 : `results_baseline_omp/<design>.{noomp,omp,opt}.log` (`.out`은 삭제)

## CPU FLUTE(OpenMP) vs GPU-FLUTE(degree ≥ 10) — OpenMP 빌드, 나머지 노브 전부 on

`INSTANTGR_GPU_FLUTE=0`(CPU 8스레드) / `=1`만 토글. S1 RSMT는 stage 시간(=max(CPU 루프, GPU tail)).

| design | 넷 수 | CPU FLUTE S1 RSMT | GPU-FLUTE S1 RSMT | GPU wall | CPU total | GPU total | Δ total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| mempool_tile | 129k | 0.11 | 0.23 | 0.23 | 2.08 | 2.12 | +0.04 |
| mempool_tile_rank | 125k | 0.11 | 0.23 | 0.23 | 2.24 | 2.31 | +0.07 |
| ariane133_51 | 117k | 0.10 | 0.23 | 0.23 | 1.91 | 2.03 | +0.12 |
| ariane133_68 | 100k | 0.09 | 0.21 | 0.22 | 1.90 | 2.11 | +0.21 |
| nvdla | 162k | 0.16 | 0.23 | 0.23 | 3.04 | 3.11 | +0.07 |
| bsg_chip | 744k | 0.81 | 0.72 | 0.71 | 10.88 | 10.69 | −0.19 |
| mempool_group | 2,491k | 2.11 | 1.14 | 0.90 | 26.02 | 24.74 | −1.28 |

- CPU FLUTE가 8스레드가 되면 GPU-FLUTE의 이득은 `mempool_group` −1.3 s, `bsg_chip` −0.2 s뿐. 작은 디자인 5개는 오히려 +0.04~0.21 s 손해
- 원인 : GPU-FLUTE 고정 비용 ≈ 0.2 s. `mempool_tile`(GPU 넷 6,758개) 프로파일 — solve wall 0.210 s 중 device pipeline 0.203 s, H2D **131 MB**. `upload_lut()`가 호출마다 FLUTE LUT를 호스트에서 평탄화해 올리는 비용이 대부분. 넷 수와 무관한 상수라 작은 디자인에서는 CPU 루프(0.05 s)보다 GPU tail(0.18 s)이 길어짐
- 비OpenMP 기준으로 적어 둔 `rtx3060-ab.md`의 GPU-FLUTE 기여(`mempool_group` −7.57 s)는 OpenMP 빌드에서는 −1.3 s로 봐야 함
- 다음 후보 : LUT를 한 번만 올려 두고 재사용(또는 Stage 2 호출과 공유)하면 고정 비용이 사라져 작은 디자인의 손해가 없어질 것
- 로그 : `results_baseline_omp/<design>.{cpuflt,gpuflt}.log`

## LUT 상주 + 파싱과 겹치기 — 고정 비용 제거 후 재비교

변경 (`gpu_flute.hpp`, `main.cpp`) :

- `gpu_flute::resident_lut()` — FLUTE LUT를 프로세스에서 한 번만 평탄화·업로드하고 상주 (~130 MB). 호출마다 `upload_lut()` + `release()` 하던 것을 제거
- `main()`에서 `readLUT()` + GPU 업로드를 스레드로 띄워 입력 파싱·넷 분할과 겹침. Stage 1 전에는 아무도 LUT를 안 쓰므로 안전. `read FLUTE LUT`(0.07 s) 구간도 같이 사라짐
- 중간 버전(v1 : `route()` 안에서 CUDA DB 구축과만 겹침)은 작은 디자인에서 0.15 s가 그대로 노출돼 부족했음

CPU FLUTE(OpenMP 8스레드) vs GPU-FLUTE(degree ≥ 10), 나머지 노브 on. total(s), Δ = GPU − CPU :

| design | v0 호출마다 업로드 | v1 DB 구축과 겹침 | **v2 파싱과 겹침** | v2 S1 RSMT cpu / gpu |
| --- | --- | --- | --- | --- |
| mempool_tile | 2.08 / 2.12 (+0.04) | 2.03 / 2.10 (+0.07) | 1.96 / 1.87 (**−0.09**) | 0.11 / 0.05 |
| mempool_tile_rank | 2.24 / 2.31 (+0.07) | 2.23 / 2.33 (+0.10) | 2.06 / 2.08 (+0.02) | 0.10 / 0.05 |
| ariane133_51 | 1.91 / 2.03 (+0.12) | 1.93 / 1.99 (+0.06) | 1.84 / 1.79 (**−0.05**) | 0.10 / 0.05 |
| ariane133_68 | 1.90 / 2.11 (+0.21) | 1.99 / 2.09 (+0.10) | 1.92 / 1.87 (**−0.05**) | 0.09 / 0.04 |
| nvdla | 3.04 / 3.11 (+0.07) | 3.04 / 3.06 (+0.02) | 2.98 / 2.85 (**−0.13**) | 0.16 / 0.08 |
| bsg_chip | 10.88 / 10.69 (−0.19) | 10.94 / 10.66 (−0.28) | 10.98 / 10.37 (**−0.61**) | 0.81 / 0.54 |
| mempool_group | 26.02 / 24.74 (−1.28) | 25.91 / 24.88 (−1.03) | 25.73 / 25.10 (−0.63)* | 2.07 / 1.01 |
| mempool_group 재측정 | | | 25.52 / 24.40 (**−1.12**) | 2.04 / 0.99 |

\* 첫 런은 `DAG build (DFS)`가 1.94 s로 튄 것(다른 런 1.4~1.6 s). 재측정에서 1.53 s로 복귀

- GPU wall : 작은 디자인 0.23 s → 0.05 s. S1 RSMT는 7개 디자인 전부 GPU가 CPU의 절반
- 작은 디자인의 손해(+0.04~0.21 s)가 사라지고 −0.05~−0.13 s로 돌아섬. `bsg_chip` −0.61, `mempool_group` −1.1
- 검증 : `INSTANTGR_GPU_FLUTE_VALIDATE=1` mempool_group, Hanan 0 / WL 1 (기존 degree 34 건)
- 로그 : `results_baseline_omp/lut_resident/`(v1), `lut_resident2/`(v2, `.r2.log`가 재측정)

## `mempool_group` base(off) vs opt — OpenMP 빌드, 상주 LUT 포함 (보고서 양식)

`run/InstantGR.opt`를 `-Xcompiler -fopenmp`로 빌드. base = 4개 노브 전부 off, opt = 전부 on (degree 10). opt는 2회 중 2회차(1회차 26.04 s는 input 5.01·detour 2.47로 튐). 로그 : `results_mempool_group_gpuflute_all/final_omp/`

| 항목 | base | opt | 개선 |
| --- | --- | --- | --- |
| input | 4.59 | 4.67 | — |
| Lshape route (S1) | 29.81 | 6.86 | −77.0% |
| &nbsp;&nbsp;S1: RSMT (CPU+GPU 합) | 2.06 | 1.05 | −49.0% |
| &nbsp;&nbsp;S1: batch generation | 3.40 | 0.98 | −71.2% |
| &nbsp;&nbsp;S1: DAG build (DFS) | 1.89 | 1.52 | −19.6% |
| &nbsp;&nbsp;S1: GPU route batches | 21.90 | 2.88 | −86.8% |
| DAG/detour route (S2) | 41.07 | 11.09 | −73.0% |
| &nbsp;&nbsp;S2: detour generation | 2.29 | 1.75 | −23.6% |
| &nbsp;&nbsp;S2: batch generation | 3.32 | 1.29 | −61.1% |
| &nbsp;&nbsp;S2: host DAG prep + upload | 1.94 | 1.94 | 0.0% |
| &nbsp;&nbsp;S2: GPU route batches | 33.45 | 6.04 | −81.9% |
| finish nets + final output | (접힘) | 1.43 | — |
| other | 2.27 | 0.76 | −66.5% |
| total | 77.74 | 24.81 | −69.1% |

- 배율 3.13× (비OpenMP 92.47 → 28.28, 3.27×). base의 RSMT가 12.47 → 2.06으로 줄어든 것이 차이의 전부
- opt `other` = build CUDA database 0.50 + 12 minor 0.26 (비OpenMP 표의 0.82와 같은 범위)

### 감소분의 귀속 (위 base/opt 표에서, 08-23 문서 §3과 같은 방식)

| 최적화 | 해당 항목 | `mempool_group` OpenMP (3060) | 참고 : 08-23 비OpenMP (TITAN) |
| --- | --- | ---: | ---: |
| incremental vcost / presum | S1 + S2 `GPU route batches` | **−46.43 s** (87.7%) | −18.16 s (58.5%) |
| GPU-FLUTE | S1: RSMT, CPU FLUTE | **−1.01 s** (1.9%) | −7.57 s (24.4%) |
| GPU batch generation | S1 + S2 `batch generation` | **−4.45 s** (8.4%) | −5.25 s (16.9%) |
| 그 외 (증가분 포함) | | −1.04 s | −0.04 s |
| | **실측 total 델타** | **−52.93 s** | −31.02 s |

- GPU-FLUTE 몫이 24.4% → 1.9%. CPU FLUTE가 8스레드가 되면서 base의 RSMT 자체가 2.06 s뿐이라 줄일 여지가 1 s
- 같은 날 leave-one-out(`INSTANTGR_GPU_FLUTE=0`만 끔, 상주 LUT) −1.12 s와 일치
- vcost/presum 몫은 절대값(−46 s)도 비중(88%)도 커짐 — 3060에서 GPU route batches가 TITAN보다 오래 걸리고(base 55 s), OpenMP로 다른 몫이 줄어서
