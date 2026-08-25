# 내 작업 환경 전용 메모

- 서버 계정(`mskim`) 기준 실제 경로·복붙용 명령
- 일반 설명은 [README](../README.md)

| 항목 | 값 |
| --- | --- |
| 서버 | `intern` (RHEL 8, glibc 2.28) |
| 셸 | tcsh |
| 리포지토리 | `/home/2026fall/mskim/InstantGR-2026` |
| 벤치마크 | `/home/2026fall/mskim/InstantGR-2026/benchmarks` |
| GPU | GeForce RTX 3060 12GB → **`-arch=sm_86`** |
| 드라이버 | 610.57.04 (CUDA 13.3) |
| CUDA toolkit | **`~/cuda-12.6`** (홈에 직접 설치, 시스템에는 없음) |
| 호스트 컴파일러 | `/usr/bin/gcc` 8.5.0 (기본) 또는 gcc-toolset-13 (13.3.1) |

- 벤치마크 다운로드 : [Google Drive](https://drive.google.com/drive/folders/1afrsbeS_KuSeHEVfuQOuLWPuuZqlDVlw?hl=ko)
- 기본 4종 중 `bsg_chip`은 아직 없음 (필요하면 위에서 받아 같은 디렉터리에 둘 것)

---

## 0. PATH (매 셸마다 필요)

- 시스템에 `nvcc`가 없음. 홈에 설치한 것을 PATH에 올려야 함

```bash
setenv PATH $HOME/cuda-12.6/bin:$PATH
```

- 매번 치기 싫으면 `~/.tcshrc`에 추가 :

```bash
echo 'setenv PATH $HOME/cuda-12.6/bin:$PATH' >> ~/.tcshrc
```

### toolkit 재설치 (다른 서버로 옮길 때만)

- `/home`이 NFS가 아니라 이 서버 로컬 디스크 → 서버가 바뀌면 다시 설치해야 함
- `~/.tcshrc`가 이미 공용 anaconda를 source하므로 `conda`는 그냥 잡힘

```bash
env CONDA_PKGS_DIRS=$HOME/.conda/pkgs conda create -y -p $HOME/cuda-12.6 --override-channels -c nvidia -c conda-forge cuda-toolkit=12.6.3
```

- `-c conda-forge` 필수 (`libgcc-ng`가 nvidia 채널에 없어서 빼면 solve 실패)
- Anaconda 기본 채널은 ToS 미승인 상태라 `--override-channels`로 우회
- 용량 : env 4.4G + `~/.conda/pkgs` 캐시 1.9G (`conda clean -a`로 캐시 회수 가능)
- root가 있으면 이게 더 깔끔 : `sudo dnf install cuda-toolkit-12-6` (cuda repo는 이미 붙어 있음)

---

## 1. GPU 선택

```bash
nvidia-smi
```

```bash
setenv CUDA_VISIBLE_DEVICES 0
```

---

## 2. 빌드

- `-arch=sm_86` (RTX 3060). README의 `sm_75`는 이전 TITAN RTX 환경 값

```bash
cd /home/2026fall/mskim/InstantGR-2026/src && nvcc main.cpp -o ../run/InstantGR.opt -std=c++17 -x cu -O3 -arch=sm_86
```

```bash
cd /home/2026fall/mskim/InstantGR-2026/baseline/src && nvcc main.cpp -o ../run/InstantGR -std=c++17 -x cu -O3 -arch=sm_86
```

```bash
cd /home/2026fall/mskim/InstantGR-2026/run && g++ -O3 -std=c++17 -o evaluator evaluator.cpp && chmod +x ./evaluator
```

- 기본 gcc 8.5로 빌드 확인됨. 최신 호스트 컴파일러를 쓰고 싶으면 `-ccbin` 추가 (선택) :

```bash
cd /home/2026fall/mskim/InstantGR-2026/src && nvcc main.cpp -o ../run/InstantGR.opt -std=c++17 -x cu -O3 -arch=sm_86 -ccbin /opt/rh/gcc-toolset-13/root/usr/bin/g++
```

### gcc / CUDA 호환 (host_config.h 실측)

- 하한 체크는 없음. 상한만 있음 → gcc 8.5는 어느 버전이든 통과

| CUDA | 허용 gcc |
| --- | --- |
| 12.6 | ≤ 13 |
| 12.9 | ≤ 14 |
| 13.x | ≤ 15 |

- 이 서버의 gcc : `/usr/bin/gcc` 8.5 · `/opt/rh/gcc-toolset-13/...` 13.3.1 · `/opt/rh/gcc-toolset-15/...` 15.2.1
- toolset-15(15.2.1)를 쓰려면 CUDA 13.x가 필요 → 현재 12.6과는 조합 불가

---

## 3. 실행

```bash
cd /home/2026fall/mskim/InstantGR-2026/run; time ./InstantGR.opt -cap ../benchmarks/mempool_tile_rank.cap -net ../benchmarks/mempool_tile_rank.net -out test.out |& tee test.log
```

## 4. 평가

```bash
cd /home/2026fall/mskim/InstantGR-2026/run && ./evaluator ../benchmarks/mempool_tile_rank.cap ../benchmarks/mempool_tile_rank.net test.out
```

## 5. A/B 자동 실행

- 스크립트 기본값이 이전 환경(`sm_75` + `/home/shkim/...` 경로)이라 **인자 없이는 안 됨**. 매번 override 필요

```bash
env BENCH=/home/2026fall/mskim/InstantGR-2026/benchmarks ARCH=sm_86 ./run_ab_no_treecenter.sh mempool_tile_rank
```

- 디자인 인자를 생략하면 기본 4종(`mempool_tile_rank` `mempool_group` `mempool_cluster_ranking` `bsg_chip`)을 돎 — `bsg_chip`이 로컬에 없으면 해당 디자인에서 실패

---

## nsys 프로파일

- 이 서버에는 nsys 없음 (`~/cuda-12.6`에는 `ncu`만 있음). 필요해지면 그때 설치
- 절차·주의사항은 [profiling-nsys.md](profiling-nsys.md), 측정 원칙은 [measure-runtime.md](measure-runtime.md)

---

## 이 환경의 함정

- **다른 머신에서 빌드한 바이너리는 복사해도 안 돌아감.** 이 서버는 glibc 2.28인데 이전 환경은 2.29 이상이라 `version 'GLIBC_2.29' not found`로 죽음. 하위 호환은 되지만 상위 호환은 안 됨 → 여기서 직접 빌드할 것
  - git에 커밋된 `run/evaluator` · `baseline/run/evaluator`도 여기서는 실행 불가. 위 빌드 명령으로 다시 만들 것
- CUDA 런타임은 nvcc 기본값이 `-cudart static`이라 바이너리에 정적 링크됨 → 실행 시 `LD_LIBRARY_PATH` 불필요
- `~/.tcshrc`의 `setenv LD_LIBRARY_PATH ${CONDA_PREFIX}/lib`는 기존 값을 append가 아니라 덮어씀. 다른 conda env를 activate할 때 주의
- docs에 기록된 절대 런타임 수치는 이전 환경(TITAN RTX + 다른 toolkit) 기준이라 여기서는 재현되지 않음. opt/baseline을 같은 toolkit으로 빌드하므로 A/B 상대 비교만 유효
