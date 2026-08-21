# 내 작업 환경 전용 메모

서버 계정(`shkim`) 기준 실제 경로·복붙용 명령. 일반 설명은 [README](../README.md).

- 셸 : tcsh
- 리포지토리 : `/home/shkim/6_Internship/mskim/InstantGR`
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
cd /home/shkim/6_Internship/mskim/InstantGR/src && nvcc main.cpp -o ../run/InstantGR.opt -std=c++17 -x cu -O3 -arch=sm_75
```

```bash
cd /home/shkim/6_Internship/mskim/InstantGR/baseline/src && nvcc main.cpp -o ../run/InstantGR -std=c++17 -x cu -O3 -arch=sm_75
```

```bash
cd /home/shkim/6_Internship/mskim/InstantGR/run && g++ -O3 -std=c++17 -o evaluator evaluator.cpp && chmod +x ./evaluator
```

## 실행

```bash
cd /home/shkim/6_Internship/mskim/InstantGR/run; time ./InstantGR.opt -cap ../../benchmarks/benchmarks/mempool_cluster_ranking.cap -net ../../benchmarks/benchmarks/mempool_cluster_ranking.net -out test.out |& tee test.log
```

## 평가

```bash
cd /home/shkim/6_Internship/mskim/InstantGR/run && ./evaluator ../../benchmarks/benchmarks/mempool_cluster_ranking.cap ../../benchmarks/benchmarks/mempool_cluster_ranking.net test.out
```

## A/B 자동 실행

- 스크립트 기본값이 이미 위 벤치마크 경로 + `sm_75` → 인자 없이 실행 가능

```bash
cd /home/shkim/6_Internship/mskim/InstantGR && ./run_ab_no_treecenter.sh
```

- 디자인 하나만:

```bash
cd /home/shkim/6_Internship/mskim/InstantGR && ./run_ab_no_treecenter.sh bsg_chip
```
