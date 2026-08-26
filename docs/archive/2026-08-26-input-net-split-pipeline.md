# 2026-08-26 — input 파싱 ∥ net 쪼개기 파이프라인

- 브랜치 : `pipeline-overlap`
- 배경 : [2026-08-25-pipeline-plan.md](../2026-08-25-pipeline-plan.md) 후보 표 2번 (input 파싱 ∥ CUDA DB build)
- 환경 : intern 서버 (RTX 3060 12GB, `sm_86`), opt config 4개 전부 on

## 사전 측정 — 어디를 겹칠지

`build_cuda_database()` 내부를 구간별로 재서 겹칠 가치가 있는 곳을 특정 (`mempool_group`) :

| 구간 | 시간 | 의존성 |
| --- | ---: | --- |
| **net 쪼개기 루프 (subnet 분해 + MST)** | **1.18s** | net 파싱 결과 |
| grid 업로드 (capacity 155MB) | 0.16s | cap만 |
| pin 업로드 | 0.17s | net |
| track/congestion 할당 | 0.03s | cap만 |

- cap만으로 되는 부분은 0.18s뿐 → "cap 파싱 후 grid 먼저 빌드"는 무익
- net 쪼개기 루프는 net 단위 독립 → 파서를 따라가는 소비자로 분리 가능. 이게 유일한 대상

계획 문서의 "net 단위 스트리밍 파싱 필요" 우려는 불필요했음 — 파서는 그대로 두고,
**완성된 net 개수만 atomic으로 publish**하면 소비자가 뒤따라갈 수 있음.

## 구현

- 파서(`db::read`) : net 하나 완성할 때마다 `db_parsed_net_count`를 release-store
  (`db::nets`는 사전 reserve로 재할당이 없어 소비자가 lock 없이 읽어도 안전)
- 소비자(`cudb::build_nets_from_parse`, 스레드 1개) : count를 acquire-load하며 따라가서
  subnet 분해 + `calc_hpwl` 수행. `build_cuda_database()`에는 GPU 업로드만 남음
- main : `db::read()` 직전에 스레드 launch, 직후 join (`wait for net split` 행으로 계측)

### 과정에서 잡은 것 2건

| 문제 | 증상 | 수정 |
| --- | --- | --- |
| 소비자가 따라잡은 뒤 `yield()` 스핀 | 파서 3.76 → 4.23s (+0.47s) | 200µs sleep으로 교체 → +0.18s로 축소 |
| `calc_hpwl`이 소비 루프 밖의 직렬 꼬리 | 이득 0.4s에 그침 | 루프 안으로 이동 → 이득 0.9s |

## 결과

경계 = `Stage 1 initial routing: RSMT starts` 타임스탬프 (input + DB 구축 전체).
total은 공유 서버 노이즈(±1s)가 구간 이득보다 커서 판정 지표로 쓰지 않음.

| 디자인 | pre-route base | pre-route pipe | 순이득 | 파서 경합 |
| --- | ---: | ---: | ---: | ---: |
| `mempool_tile_rank` | 0.5s | 0.4s | −0.1s | — |
| `mempool_group` (3회) | 6.0 / 6.1 / 6.2s | 5.2 / 5.5 / 5.2s | **−0.9s** | +0.2s |
| `mempool_cluster_ranking` (2회) | 19.0 / 18.8s | 17.3 / 17.1s | **−1.7s** | +0.8s |

- 순이득 = net 쪼개기 루프가 파싱 뒤로 숨은 만큼 − 파서 경합
- 품질 : `MAX PINS` / `Broken Nets` / `PIN_NUM` 전 디자인 동일.
  `mempool_tile_rank` evaluator score **바이트 동일**, `mempool_group`은 런간 노이즈 수준(±0.0002%)
- cluster는 GPU-FLUTE off로 측정 (아래 참고) — 측정 경계가 GPU-FLUTE 이전이라 영향 없음

## 참고 — cluster + GPU-FLUTE OOM (기존 문제)

`mempool_cluster_ranking` + `INSTANTGR_GPU_FLUTE=1`은 이 서버(12GB)에서
`GPU-FLUTE allocate score si scratch: out of memory`로 시작 직후 죽음.
base(HEAD 빌드)도 동일하게 죽으므로 이 브랜치와 무관 — TITAN RTX(24GB) 시절엔 문제 없던 할당.
cluster 풀런/score 대조에는 당분간 `INSTANTGR_GPU_FLUTE=0` 필요.
