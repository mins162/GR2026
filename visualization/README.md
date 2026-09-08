# visualization

`docs/rtx3060-ab.md`의 수치로 보고서 그림을 그리는 스크립트 모음.
배선 그림 한 장만 예외로 라우터 출력(`.out`)을 직접 읽음.
피드백용 시안이며, 아직 확정된 그림 없음.

## 실행

```
cd visualization && python3 make_all.py
```

`figures/`에 그림당 **SVG + PNG** 두 벌 생성. SVG가 보고서용(벡터), PNG는 확인용.

## 파일

| 파일 | 그림 | 쓸 자리 |
| --- | --- | --- |
| `plot_speedup.py` | 디자인별 속도 향상 배율 | 3장 결과 요약 |
| `plot_base_vs_opt.py` | base / opt 실행 시간 나란히 | 3장 결과 요약 (위와 택일) |
| `plot_ladder.py` | 누적 사다리 폭포 — 92.47 s → 28.28 s | 3장 결과 |
| `plot_ladder_stacked.py` | 같은 내용을 막대 하나로, 기여별 색 분할 | 3장 결과 (위와 택일) |
| `plot_contributions.py` | 기여별 leave-one-out 마진 (두 디자인) | 3장 기여별 서술 |
| `plot_breakdown.py` | 단계별 시간 구성, base vs opt | 2장 구성 및 동작 |
| `plot_breakdown_pie.py` | 단계별 구성 원형(도넛), opt만 | 2장 (위와 택일) |
| `plot_overlap_timeline.py` | CPU/GPU 오버랩 타임라인 | 2장 GPU-FLUTE |
| `plot_memcpy_timeline.py` | CPU↔GPU 메모리 복사 타임라인 (nsys, `mempool_group`) | 2장 동작 / 4장 한계·향후 |
| `plot_cpu_gpu_timeline.py` | CPU 연산 / CUDA API 대기 / GPU 커널 점유율 타임라인 (같은 nsys) | 2장 동작 / 4장 한계·향후 |
| `plot_strategy_blocks.py` | CPU / 전송 / GPU 블록 타임라인, InstantGR Fig. 11 방식, 전략 위치를 주황 상자로 | 2장 전략 개요 |
| `plot_gpu_flute_overlap.py` | 같은 블록 형식으로 S1 RSMT만 확대: GPU FLUTE 스레드가 CPU FLUTE 루프 아래 숨는 모습 | 2장 GPU-FLUTE / 오버랩 |
| `plot_s2_batch.py` | 같은 블록 형식으로 Stage 2 배치 하나 확대: rip-up → vcost → presum → DP → traceback → commit | 2장 incremental vcost/commit |
| `plot_routing.py` | 배선 크롭 + 밀도 히트맵 (`mempool_tile_rank`, 180×180 GCell) | 3장 결과 — 라우팅 그림 |
| `plot_routing_layers.py` | 같은 크롭을 금속층별로 분리 (+ 배선 없는 밀도) | 3장 결과 (위와 택일) |
| `plot_die_overview.py` | 다이 전체 밀도, 위 크롭 위치 표시 | 3장 결과 — 위 그림의 맥락 |
| `data.py` | 위 전부의 입력 수치 (`plot_routing*.py`, `plot_die_overview.py`, `plot_*_timeline.py`, `plot_strategy_blocks.py`, `plot_gpu_flute_overlap.py`, `plot_s2_batch.py` 제외) | — |
| `style.py` | 공통 스타일·색·저장 | — |

## 수치 출처

`data.py`가 [../docs/rtx3060-ab.md](../docs/rtx3060-ab.md)를 **손으로 옮긴 것**.
각 상수 위에 출처 절 번호를 주석으로 달아둠. 재측정하면 `data.py`만 고치면 됨.

`runs.csv`는 이 저장소에 없음(서버 산출물). 있으면 `data.py`를 CSV 로더로 바꾸는 쪽이 안전함.

`plot_routing.py`만 `data.py`를 안 거치고 `sample/mempool_tile_rank.opt.r1.out.gz`를
직접 파싱함. 원본은 `ab_results_0826_143856/mempool_tile_rank.opt.r1.out` (33 MB → gz 5.3 MB).
4개 디자인 중 저장소에 넣을 수 있는 크기는 이것뿐임 — `mempool_group`은 884 MB.

## 색과 흑백

- 파랑 = 우리 기여, 회색 + **빗금** = ICCAD'22 논문 알고리즘
- 빗금을 넣은 이유는 흑백 인쇄에서 색만으로는 구분이 안 되기 때문
- 글자 크기는 그림 폭 대비로 맞춤. 전체 조정은 `style.py`의 `font.size`
- 라벨은 전부 영문 — 한글 폰트 문제를 피하기 위함 ([../docs/visualization-tools.md](../docs/visualization-tools.md) 5)

## 아직 안 된 것

- **SciencePlots 미설치.** 지금은 `style.py`의 자체 스타일로 그려짐.
  설치하면 `science`/`ieee` 스타일이 자동 적용되며 **그림 모양이 바뀜**
- **혼잡도 히트맵 없음.** overflow는 demand뿐 아니라 capacity(`.cap`)가 있어야 나오는데
  `.cap`은 저장소에 없음(`benchmarks/`는 gitignore). `mempool_tile_rank.cap`은 gz 30 KB라
  올리려면 올릴 수 있음
- **다이 전체 그림 없음.** `plot_routing.py`는 30×30 GCell 창만 그림. 429×581 전체는
  배선 428k개라 보고서 크기에서 검은 덩어리가 됨. 큰 디자인(`mempool_group`, 884 MB,
  3.2M net)은 서버에 있고 datashader도 미설치
- **`routing_crop` 그림 파일 없음.** 서버에 matplotlib이 없어 렌더 못 함.
  로컬에서 `make_all.py` 한 번 돌리면 `figures/`에 생김
- 흑백 출력 테스트 안 함

`plot_memcpy_timeline.py`, `plot_cpu_gpu_timeline.py`, `plot_strategy_blocks.py`, `plot_gpu_flute_overlap.py`, `plot_s2_batch.py`는 `sample/mempool_group-incr.memcpy.csv.gz`를 읽음.
그 프로파일엔 CPU 샘플링이 없어서(`--sample=none`) "CPU 연산"은 메인 스레드가 CUDA API 밖에 있던 시간으로 정의함.
진짜 CPU 점유율이 필요하면 `tools/nsys_profile.sh --cpu`로 다시 찍어야 함.
원본은 `nsys_results_0826_012934/mempool_group-incr.sqlite`(gitignore)이고,
`python3 ../tools/nsys_memcpy_csv.py <sqlite> sample/<이름>.memcpy.csv.gz 21.99 22.03`으로 다시 뽑을 수 있음
(뒤의 두 수는 `plot_s2_batch.py`가 쓰는 커널 이름 창, 초 단위).
