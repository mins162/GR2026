# 내 작업 환경 전용 메모

- 서버 계정(`shkim`) 기준 실제 경로·복붙용 명령
- 일반 설명은 [README](../README.md)

- 셸 : tcsh
- 리포지토리 : `/home/shkim/6_Internship/mskim/InstantGR-2026`
- 벤치마크 : `/home/shkim/6_Internship/mskim/benchmarks/benchmarks`
- 벤치마크 다운로드 : [Google Drive](https://drive.google.com/drive/folders/1afrsbeS_KuSeHEVfuQOuLWPuuZqlDVlw?hl=ko)
- GPU : TITAN RTX → `-arch=sm_75`

---

## GPU 선택

```bash
nvidia-smi
```

```bash
setenv CUDA_VISIBLE_DEVICES 0
```

---

## 빌드

```bash
cd /home/shkim/6_Internship/mskim/InstantGR-2026/src && nvcc main.cpp -o ../run/InstantGR.opt -std=c++17 -x cu -O3 -arch=sm_75
```

```bash
cd /home/shkim/6_Internship/mskim/InstantGR-2026/baseline/src && nvcc main.cpp -o ../run/InstantGR -std=c++17 -x cu -O3 -arch=sm_75
```

```bash
cd /home/shkim/6_Internship/mskim/InstantGR-2026/run && g++ -O3 -std=c++17 -o evaluator evaluator.cpp && chmod +x ./evaluator
```

## 실행

```bash
cd /home/shkim/6_Internship/mskim/InstantGR-2026/run; time ./InstantGR.opt -cap ../../benchmarks/benchmarks/mempool_cluster_ranking.cap -net ../../benchmarks/benchmarks/mempool_cluster_ranking.net -out test.out |& tee test.log
```

## 평가

```bash
cd /home/shkim/6_Internship/mskim/InstantGR-2026/run && ./evaluator ../../benchmarks/benchmarks/mempool_cluster_ranking.cap ../../benchmarks/benchmarks/mempool_cluster_ranking.net test.out
```

## A/B 자동 실행

- 스크립트 기본값이 이미 위 벤치마크 경로 + `sm_75` → 인자 없이 실행 가능

```bash
cd /home/shkim/6_Internship/mskim/InstantGR-2026 && ./run_ab_no_treecenter.sh
```

- 디자인 하나만 :

```bash
cd /home/shkim/6_Internship/mskim/InstantGR-2026 && ./run_ab_no_treecenter.sh bsg_chip
```

---

## nsys 프로파일 (vcost / presum 검증)

- 브랜치 : `profile/nsys-vcost-presum`, 설명은 [profiling-nsys.md](profiling-nsys.md)
- 경로만 실제 리포지토리로 맞추면 됨
- 벤치마크 경로·`sm_75`는 스크립트 기본값이라 인자 없이 실행 가능

### 0. 파일 확인

- git 없이 손으로 복사한 경우 실행 권한 소실

```bash
chmod +x /home/shkim/6_Internship/mskim/InstantGR-2026/tools/nsys_profile.sh
```

```bash
ls -la /home/shkim/6_Internship/mskim/InstantGR-2026/tools /home/shkim/6_Internship/mskim/InstantGR-2026/src/nvtx_profile.hpp
```

- 필요한 파일 : `tools/nsys_profile.sh`, `tools/nsys_summarize.py`, `src/nvtx_profile.hpp`, NVTX 구간이 들어간 `src/main.cpp` · `src/Lshape_route.hpp` · `src/Lshape_route_detour.hpp`

### 1. nsys 확인

- 스크립트가 자동 탐색하므로 보통 생략 가능
- `$CUDA/bin/nsys`는 버전 불일치 시 거부하는 래퍼 (`Error: Nsight Systems 2024.2.3 hasn't been installed with CUDA Toolkit 12.5`) → 동작하는 바이너리를 `--version`으로 확인해가며 선택
- 설치된 것 직접 확인 :

```bash
ls -d /opt/nvidia/nsight-systems/*/target-linux-x64/nsys /opt/nvidia/nsight-systems-cli/*/target-linux-x64/nsys /usr/local/cuda*/nsight-systems-*/target-linux-x64/nsys
```

- 못 찾으면 경로 직접 지정 :

```bash
setenv NSYS /opt/nvidia/nsight-systems/2024.2.3/target-linux-x64/nsys
```

### 2. 빈 GPU 잡기

```bash
nvidia-smi
```

```bash
setenv CUDA_VISIBLE_DEVICES 0
```

### 3. 프로파일 (작은 디자인부터)

- `incr` / `full` / `paper` 3종 연속 실행, 빌드 포함

```bash
cd /home/shkim/6_Internship/mskim/InstantGR-2026 && ./tools/nsys_profile.sh -d mempool_tile_rank
```

- 결과 : `nsys_results_MMDD_HHMMSS/SUMMARY.txt`
- 여기서 결론이 나면 아래 생략 가능

### 4. 큰 디자인

- `mempool_cluster_ranking`은 커널 런치가 수백만 건 → 트레이스가 GB 단위
- `mempool_group`부터 실행할 것

```bash
cd /home/shkim/6_Internship/mskim/InstantGR-2026 && ./tools/nsys_profile.sh -d mempool_group
```

```bash
cd /home/shkim/6_Internship/mskim/InstantGR-2026 && ./tools/nsys_profile.sh -d mempool_cluster_ranking --no-build
```

### 5. CPU가 병목인지 (FLUTE가 논문보다 느린 건)

```bash
cd /home/shkim/6_Internship/mskim/InstantGR-2026 && ./tools/nsys_profile.sh -d mempool_tile_rank -c incr --cpu
```

### 6. 요약만 다시 보기

```bash
cd /home/shkim/6_Internship/mskim/InstantGR-2026 && python3 tools/nsys_summarize.py nsys_results_0823_140000
```

### 7. 타임라인 눈으로 보기 (X11 필요)

```bash
nsys-ui /home/shkim/6_Internship/mskim/InstantGR-2026/nsys_results_0823_140000/mempool_tile_rank-incr.nsys-rep
```

### 주의

- `INSTANTGR_AUGMENTED_DAG_PROFILE=1`과 동시 사용 금지 — 이벤트 sync가 파이프라인을 변형
- nsys 실행의 wall clock을 성능 수치로 사용 금지 — wall 비교는 `run_ab_no_treecenter.sh`, 구간 분해는 nsys
  - 이유는 오버헤드가 아니라 **커버리지** : 커널 표는 wall의 ~29%만 덮고 호스트 70%는 안 보임 ([measure-runtime.md](measure-runtime.md))
  - 오버헤드 자체는 `mempool_group`에서 노이즈 수준 (nsys 36.77s vs 비-nsys 37.04s), 런치가 수백만 건인 `mempool_cluster_ranking`에서는 다름
- 같은 GPU를 남이 사용 중이면 커널 시간이 부풀어 비교 무의미
- 실행 권한 없으면 `bash tools/nsys_profile.sh ...` 로도 실행 가능
