# PDGNN-MolHIV + 3D 컨포머 기반 전기적 성질 확장 모델 요약

## 1. 배경 및 목적

기존 OGBG-MolHIV 그래프(노드=원자, 엣지=공유결합)는 **전기적 성질(전자 분포)**과
**3D 형태 정보**를 담고 있지 않다는 문제의식에서 출발했다. RDKit을 이용해
아래 두 가지를 추가로 계산하여 node/edge feature를 확장했다.

---

## 2. 추가된 Feature

### 2-1. Node feature: Partial Charge (9d → 10d)

- **방법**: RDKit `rdPartialCharges.ComputeGasteigerCharges`
- **특징**: 3D 구조 없이 2D 연결 정보만으로 계산 가능 (빠름, 항상 계산됨)
- **의미**: 각 원자가 전자를 얼마나 끌어당기는지(음전하) / 내주는지(양전하)를
  연속값으로 표현. 기존 OGB의 `formal charge`(정수, 대략적 근사)보다 정밀함

### 2-2. Edge feature: 3D 결합거리 + 정전기 점수 (3d → 5d)

공유결합 엣지에 한해 아래 두 값을 추가:

| feature | 계산식 | 의미 |
|---|---|---|
| 3D 결합 거리 | 컨포머 좌표 간 유클리드 거리 (Å) | 결합의 실제 공간적 길이 |
| 정전기 점수 (Coulomb term) | `q_i * q_j / distance` | 두 원자가 서로 끌어당기는지(음수) 밀어내는지(양수) |

- 3D 좌표가 있어야 계산 가능 → RDKit 컨포머 임베딩 필요
- 임베딩 실패 시 해당 엣지는 거리=0, 정전기점수=0으로 처리 (partial charge는 별도 계산이라 그대로 유지)

### 2-3. 컨포머 생성 방식: 단일 vs 멀티

| 옵션 | 방식 | 비용 |
|---|---|---|
| `--n-confs 1` (기본) | `EmbedMolecule` 1회 + MMFF(실패시 UFF) 최적화 | 빠름 (분자당 ≈0.1초) |
| `--n-confs 5` 이상 | `EmbedMultipleConfs`로 N개 생성 → 각각 MMFF/UFF 에너지 최소화 → **에너지 최저 컨포머 선택** | 느림 (분자당 ≈0.9~1초, n_confs=5 기준) |

멀티 컨포머는 "분자가 실제로 취할 법한 가장 안정적인 3D 형태"를 더 정확히
근사하려는 목적이나, 계산 비용이 컨포머 개수에 비선형적으로 증가한다.

---

## 3. 모델 구조 (원본 PDGNN-MolHIV와 동일, 입력 차원만 확장)

```
입력: x [N, 10]  (원자 9d + partial charge)
      edge_attr [E, 5]  (결합정보 3d + 3D거리 + 정전기점수)
        │
        ├─▶ LearnableFilteration ─▶ persistence image (TDA 기반 위상 정보)
        │
        └─▶ MolGINE (GINEConv × 5, edge_dim=5) ─▶ graph embedding
                        │
                        ▼
              concat(graph embedding, persistence image)
                        │
                        ▼
                   Classifier (MLP) → HIV 억제 여부 (logit)
```

- **LearnableFilteration**: node/edge feature를 각각 MLP로 스칼라 값(0~1)으로
  변환 → 이 값을 filtration 기준으로 persistence image 생성 (사람이 결합 방식을
  정하지 않고 모델이 전기적 성질의 기여도를 학습하도록 설계)
- **MolGINE**: GINEConv 기반 5-layer 인코더, edge_attr을 함께 사용해
  결합 종류 + 3D 정전기 정보를 message passing에 반영

---

## 4. 데이터 파이프라인

1. OGB `ogbg-molhiv` 로드 (41,127개 그래프)
2. **층화추출(Stratified Sampling)**: train/valid/test 각 그룹 안에서
   양성(3.5%) / 음성 비율을 유지한 채 subset 크기만큼 무작위 추출
   → subset 크기와 무관하게 항상 원래 클래스 비율 보장
3. RDKit으로 partial charge + 3D feature 계산 (분자별 캐싱, `n_confs`별로 캐시 분리)
4. PyG `Data` 객체에 feature 병합 → 학습

---

## 5. 지금까지의 실험 결과 (subset 8000 기준)

| 실험 | 추출 방식 | n_confs | Best Val AUC | Best Test AUC |
|---|---|---|---|---|
| 실험 A | 단순 무작위 | 1 (단일 컨포머) | 0.7013 | 0.5542 |
| 실험 B | 층화추출 | 5 (멀티 컨포머) | 0.8227 | 0.7036 |

> ⚠️ **주의**: 두 실험은 추출 방식(단순무작위 vs 층화추출)과 컨포머 개수(1 vs 5)가
> **동시에** 달라서, 이 차이가 "컨포머 개수 효과"인지 "추출 방식 개선 효과"인지
> 아직 분리되지 않았다. 또한 Val/Test AUC가 epoch마다 크게 진동하는 패턴이
> 관찰되는데, val/test 크기가 작아(각 800개, 양성 20~30개 수준) 생기는
> **통계적 변동(분산)일 가능성**이 있으며, 아직 train AUC를 로깅하지 않아
> 진짜 오버피팅인지 노이즈인지도 명확히 구분되지 않은 상태다.

---

## 6. 다음 단계 (남은 검증 과제)

1. **동일 조건(층화추출 8000개, n_confs=5) baseline 비교**
   `--no-3d-features` 옵션으로 정확히 같은 분자 조합에서 3D feature 유무만 비교
2. **Train AUC 로깅 추가** → 오버피팅 여부를 노이즈와 구분해서 판단
3. 필요시 여러 랜덤 시드로 반복 실행 → 평균/표준편차로 통계적 유의성 확인
4. 저비용(n_confs=1) 대비 멀티컨포머(n_confs=5)의 **성능 대비 비용** 트레이드오프 정량 비교

---

*작성 기준: `train_pdgnn_molhiv_3d.py` (PDGNN-MolHIV + RDKit 3D feature 확장 버전)*
