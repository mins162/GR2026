# 시각화 도구 선정

- 작성일 : **2026-08-29**
- 목적 : [contest-plan.md](contest-plan.md) §1 그림 V1~V6를 무엇으로 그릴지 결정
- 결론 : **`matplotlib` + `SciencePlots`로 통일.** 필요할 때만 datashader 추가, 개념도만 손으로

---

## 0. 선정 기준

보고서 그림은 아래 넷을 만족해야 한다. "예쁜가"보다 이 넷이 먼저다.

| 기준 | 이유 |
| --- | --- |
| **벡터 출력 (SVG/PDF)** | 워드 12p 문서에 넣고 인쇄. 래스터는 글자가 뭉개진다 |
| **흑백에서 읽힘** | 심사위원이 컬러로 뽑는다는 보장이 없다 |
| **숫자가 원본과 자동 일치** | `runs.csv` → 표 → 그림이 한 소스에서 나와야 한다. 손으로 옮기면 반드시 어긋난다 |
| **스타일 통일** | 그림마다 폰트·색이 다른 게 아마추어로 보이는 가장 큰 원인 |

---

## 1. 보고서 그림용 — 채택

### `matplotlib` + `SciencePlots` ← **이걸로 간다**

```
pip install SciencePlots
```

```python
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science', 'ieee'])
```

- `ieee` 스타일이 **흑백 인쇄에서 읽히도록** 설계됨 — 위 기준 2를 그대로 충족
- 폰트 크기·선 굵기·여백이 논문 기준. 기본 matplotlib의 촌스러움이 사라진다
- 벡터(SVG/PDF) 출력
- 로컬 matplotlib **3.10.9** 확인됨 (2026-08-29)
- `tools/ab_summary.py`가 이미 `runs.csv`를 읽는다 → **같은 CSV에서 그림도 뽑으면 표와 그림이 어긋날 수 없다**

### 검토했으나 채택 안 함

| 도구 | 성격 | 기각 사유 |
| --- | --- | --- |
| `seaborn` | matplotlib 위 레이어, 기본값이 예쁨 | 강점이 통계 그래프(분포·회귀). 우리에게 필요한 막대·간트·히트맵에는 SciencePlots 대비 이득이 작다 |
| `plotnine` | ggplot2 문법의 파이썬 판 | 결과물은 깔끔하나 **문법을 새로 배워야 한다**. 마감 3주 전에 낼 비용이 아니다 |

---

## 2. 발표용 — 예선 통과 후 판단

### `plotly`

- 마우스 오버로 값 표시, 확대·필터 가능
- **인쇄물에는 못 쓴다.** 보고서용 아님
- **10-09 본선 발표**에서 노트북으로 보여줄 때만 값어치가 있다 → 예선 결과(10-02) 나온 뒤 결정

---

## 3. V1 · V2 전용

### `datashader` — V1 라우팅 오버뷰를 실제로 그릴 경우에만

- `mempool_group`은 **3.2M net**. matplotlib으로 그냥 그리면 렌더링이 안 끝나거나 검은 사각형이 된다 ([contest-plan.md](contest-plan.md) §1 주의)
- datashader는 수백만 요소를 **픽셀 단위로 집계**해서 렌더링 — 정확히 이 문제를 푸는 도구
- 단, V1은 계획서에서 **△(자리 남으면)** 등급. 이 도구를 설치하는 시점 = V1을 하기로 확정한 시점

### `matplotlib.imshow` / `pcolormesh` — V2 혼잡도 히트맵

- 그리드 배열을 색으로 칠하는 것뿐이라 **별도 도구 불필요**
- 데이터는 확보됨 : `ab_results_0826_143856/*.r1.out` (base/opt 페어, 2.2 GB)

---

## 4. V5 개념도 — 코드로 그리지 말 것

batch generation 재설계 도식은 데이터 그래프가 아니라 개념도다. 손으로 그리는 게 빠르고 결과도 낫다.

| 도구 | 느낌 | 용도 |
| --- | --- | --- |
| **draw.io** | 각지고 정확 | **보고서** — 이쪽 |
| Excalidraw | 손그림 | 발표 슬라이드 |

---

## 5. 한글 폰트 — 어느 도구를 쓰든 따라오는 문제

matplotlib은 기본 폰트에 한글이 없어 라벨이 `□□□`로 나온다.

```python
plt.rcParams['font.family'] = 'AppleGothic'   # macOS
plt.rcParams['font.family'] = 'NanumGothic'   # 서버 (설치 여부 먼저 확인)
plt.rcParams['axes.unicode_minus'] = False    # 음수 부호 깨짐 방지
```

> **권장 : 그림 안 라벨은 전부 영문으로 통일하고, 캡션만 한글로 쓴다.**
> `mempool_group`, `GPU route batches` 같은 이름은 어차피 원래 영문이다.
> 이러면 폰트 문제 자체가 사라지고, 서버에서 뽑든 로컬에서 뽑든 결과가 같다.

---

## 6. 다음 할 일

- [ ] `tools/`에 플롯 스크립트 추가 — `runs.csv`를 읽어 SVG로 출력
- [ ] SciencePlots 설치 후 `science`/`ieee` 스타일 실물 확인
- [ ] 첫 그림 뽑아서 **흑백 출력 테스트** — 최종본 전에 반드시
