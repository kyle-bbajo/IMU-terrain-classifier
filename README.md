# IMU Terrain Classifier

Full-body IMU-based terrain classification using terrain-aware step segmentation, multi-branch CNN, and Hierarchical Supervised Contrastive Learning.

---

## Overview

보행 중 지면 유형을 실시간으로 분류하는 것은 하지 보조기, 의족, 재활 로봇 등 웨어러블 의료기기의 핵심 문제입니다. 기존 연구는 단일 센서 또는 단순 분류기에 의존하지만, 이 프로젝트는 **전신 IMU(관성측정장치) 신호에서 생체역학적 특징을 추출**하고 **계층적 분류 구조**를 통해 6가지 지형을 고정밀로 분류합니다.

### 분류 대상 지형

| 클래스 | 지형 |
|--------|------|
| C1 | 미끄러운 지면 (Slippery) |
| C2 | 오르막 (Uphill) |
| C3 | 내리막 (Downhill) |
| C4 | 흙길 (Dirt) |
| C5 | 잔디 (Grass) |
| C6 | 평지 (Flat) |

### 핵심 접근법

- **Heel-Strike 기반 스텝 분절**: 발 충격 시점을 검출하여 보행 주기 단위로 신호 분절
- **Multi-Branch CNN**: 신체 부위(Foot, Shank, Thigh 등)별 IMU 신호를 독립 브랜치로 처리
- **Hierarchical Classification**: 경사(오르막/내리막) 먼저 분류 → 평탄 지면 4종 세분류
- **Supervised Contrastive Learning**: 같은 지형 클래스 임베딩을 가깝게, 다른 클래스는 멀게 학습
- **Biomechanical Features**: 충격 피크, 감쇠율, 진동 등 생체역학적 피처를 CNN과 결합

---

## Data

- **출처**: Noraxon MyoMotion 전신 IMU 시스템
- **피험자**: 40명
- **센서 위치**: Foot, Shank, Thigh, Pelvis, Hand (양측, 총 54채널)
- **샘플링**: 200 Hz

원본 CSV → HDF5 변환은 `step_segmentation.py`가 담당합니다.

---

## Project Structure

```
project/
├── data/
│   ├── raw_csv/                  # 원본 CSV (Noraxon MyoMotion)
│   └── processed/
│       └── batches/
│           └── dataset.h5        # 전처리 완료 HDF5
│
└── repo/
    ├── src/                      # 공용 모듈
    │   ├── config.py             # 전역 설정 (경로, 하이퍼파라미터)
    │   ├── models.py             # M1~M6 모델 정의
    │   ├── train_common.py       # 학습 공용 유틸 (K-Fold / LOSO)
    │   ├── channel_groups.py     # IMU 채널 그룹 정의
    │   └── step_segmentation.py  # Heel-Strike 기반 스텝 분절 및 HDF5 생성
    │
    ├── experiments/
    │   ├── baseline/
    │   │   ├── train_kfold.py    # M1~M6 K-Fold 교차검증
    │   │   └── train_loso.py     # M1~M6 LOSO (Leave-One-Subject-Out)
    │   ├── supcon/
    │   │   └── train_supcon.py   # Supervised Contrastive Learning
    │   └── hierarchical/
    │       └── train_hierarchical.py  # Hierarchical SupCon (권장)
    │
    ├── logs/                     # 학습 로그 (git 추적 제외)
    ├── out_N40/                  # 실험 결과 (git 추적 제외)
    └── tests.py
```

---

## Requirements

```bash
pip install torch numpy h5py scikit-learn scipy matplotlib
```

Python 3.10+ / PyTorch 2.x 권장. GPU 환경에서 자동으로 AMP(혼합 정밀도) 적용됩니다.

---

## Usage

### Step 1. 데이터 전처리 (최초 1회)

```bash
cd ~/project/repo
python src/step_segmentation.py
```

`data/raw_csv/` 의 CSV 파일을 읽어 `data/processed/batches/dataset.h5` 를 생성합니다.

---

### Step 2. 학습 실행

#### Hierarchical SupCon (권장)

가장 높은 성능을 목표로 하는 메인 실험입니다.
Stage 1에서 경사 여부를 먼저 분류하고, Stage 2에서 평탄 지형 4종을 세분류합니다.

```bash
cd ~/project/repo/experiments/hierarchical
python train_hierarchical.py
```

백그라운드 실행 + 실시간 로그:
```bash
nohup python train_hierarchical.py > ../../logs/hierarchical.log 2>&1 &
tail -f ../../logs/hierarchical.log
```

#### Baseline — K-Fold

```bash
cd ~/project/repo
python experiments/baseline/train_kfold.py
```

옵션:
```bash
python experiments/baseline/train_kfold.py \
    --n_subjects 40 \
    --epochs 50 \
    --no-focal \
    --no-tta
```

#### Baseline — LOSO

피험자 독립성 평가 (Leave-One-Subject-Out). 논문 제출 기준 평가에 사용합니다.

```bash
python experiments/baseline/train_loso.py
```

중단 후 재시작 시 체크포인트에서 자동 재개됩니다.

#### SupCon

```bash
python experiments/supcon/train_supcon.py
```

---

## Configuration

모든 설정은 `src/config.py` 에서 관리합니다. 경로는 파일 위치 기준으로 자동 계산되므로 별도 수정이 필요 없습니다.

주요 파라미터:

| 파라미터 | 설명 |
|----------|------|
| `N_SUBJECTS` | 피험자 수 |
| `KFOLD` | K-Fold 수 |
| `SEED` | 재현성을 위한 랜덤 시드 |
| `BATCH` | GPU 메모리 기준 자동 설정 |
| `USE_AMP` | Mixed Precision 자동 감지 |
| `USE_BALANCED_SAMPLER` | 클래스 불균형 대응 샘플링 |
| `USE_TTA` | Test Time Augmentation |

---

## Validation Protocol

- **K-Fold**: StratifiedGroupKFold — 피험자 단위 분할 (subject leakage 방지)
- **LOSO**: Leave-One-Subject-Out — 피험자 완전 독립 평가
- **Inner val split**: 각 fold 내 15% subject를 val로 사용 (early stopping 기준)
- **결과 보고**: Oracle accuracy (정답 라우팅) / Pipeline accuracy (실제 예측 기반) 분리

---

## Output

결과는 `repo/out_N{N}/` 에 저장됩니다:

```
out_N40/
└── kfold/
    └── hierarchical/
        ├── summary.json          # 전체 결과 요약
        ├── config_snapshot.json  # 실험 당시 설정 스냅샷
        └── confusion_matrix.png  # 혼동 행렬
```

---

## References

- Niswander et al. (2021) — Foot IMU biomechanics
- Ordóñez & Roggen (2016) IEEE TNNLS — Hierarchical HAR
- Khosla et al. (2020) NeurIPS — Supervised Contrastive Learning
- Lin et al. (2017) ICCV — Focal Loss