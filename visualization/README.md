# visualization

`docs/rtx3060-ab.md`의 수치로 보고서 그림을 그리는 스크립트 모음.
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
| `data.py` | 위 전부의 입력 수치 | — |
| `style.py` | 공통 스타일·색·저장 | — |

## 수치 출처

`data.py`가 [../docs/rtx3060-ab.md](../docs/rtx3060-ab.md)를 **손으로 옮긴 것**.
각 상수 위에 출처 절 번호를 주석으로 달아둠. 재측정하면 `data.py`만 고치면 됨.

`runs.csv`는 이 저장소에 없음(서버 산출물). 있으면 `data.py`를 CSV 로더로 바꾸는 쪽이 안전함.

## 색과 흑백

- 파랑 = 우리 기여, 회색 + **빗금** = ICCAD'22 논문 알고리즘
- 빗금을 넣은 이유는 흑백 인쇄에서 색만으로는 구분이 안 되기 때문
- 글자 크기는 그림 폭 대비로 맞춤. 전체 조정은 `style.py`의 `font.size`
- 라벨은 전부 영문 — 한글 폰트 문제를 피하기 위함 ([../docs/visualization-tools.md](../docs/visualization-tools.md) 5)

## 아직 안 된 것

- **SciencePlots 미설치.** 지금은 `style.py`의 자체 스타일로 그려짐.
  설치하면 `science`/`ieee` 스타일이 자동 적용되며 **그림 모양이 바뀜**
- **혼잡도 히트맵 없음.** `.out` 페어(2.2 GB)가 서버에 있어 여기서 못 그림
- **라우팅 오버뷰 없음.** 위와 같은 이유 + datashader 미설치
- 흑백 출력 테스트 안 함
