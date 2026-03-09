# IMU 지형 분류 실험 파이프라인

> 9센서 216-피처 도메인 특화 확장 | v2.0 | 2026.03

---

## 목차
1. [프로젝트 개요](#1-프로젝트-개요)
2. [피처 추출](#2-피처-추출-featurespy)
3. [모델 구성](#3-모델-구성)
4. [실험 파이프라인](#4-실험-파이프라인)
5. [주요 수정 이력](#5-주요-수정-이력-v20)
6. [실험 결과 출력 구조](#6-실험-결과-출력-구조)

---

## 1. 프로젝트 개요

9개 IMU 센서(가속도계 + 자이로스코프)로부터 수집한 신호를 기반으로 보행 지형(6종)을 분류하는 딥러닝 실험 파이프라인입니다.

### 1.1 센서 구성

총 9개 센서 × 6채널(Accel 3축 + Gyro 3축) = **54채널**

| 센서 위치 | 채널 인덱스 | 채널 수 | 설명 |
|-----------|------------|--------|------|
| Pelvis (골반) | 0 – 5 | 6ch | 몸통 안정성, tilt |
| Hand LT (왼손) | 6 – 11 | 6ch | 팔 스윙 리듬 |
| Thigh LT (왼허벅지) | 12 – 17 | 6ch | 고관절 ROM |
| Shank LT (왼정강이) | 18 – 23 | 6ch | 무릎 충격 |
| Foot LT (왼발) | 24 – 29 | 6ch | 착지 패턴 |
| Hand RT (오른손) | 30 – 35 | 6ch | 팔 스윙 리듬 |
| Thigh RT (오른허벅지) | 36 – 41 | 6ch | 고관절 ROM |
| Shank RT (오른정강이) | 42 – 47 | 6ch | 무릎 충격 |
| Foot RT (오른발) | 48 – 53 | 6ch | 착지 패턴 |

### 1.2 데이터셋

| 항목 | 값 |
|------|----|
| 피험자 | 50명 |
| 클래스 | 6종 지형 (C1~C6) |
| 샘플링 레이트 | 200Hz |
| 윈도우 크기 | 256 포인트 (1.28초) |
| 총 샘플 | 65,331개 |
| 포맷 | HDF5 — `subjects/S{id}/X`, `y` |

---

## 2. 피처 추출 (features.py)

기존 44개(Foot Accel 전용) → **9센서 전체 216개** 도메인 특화 피처로 확장

### 2.1 피처 구성 요약

| 센서 그룹 | 피처 수 | 핵심 피처 |
|-----------|--------|----------|
| Pelvis | 22개 | jerk RMS, tilt 불안정성, 지배 주파수, 벡터 크기 |
| Hand LT + RT | 38개 | 팔 스윙 리듬, 좌우 대칭 상관계수 |
| Thigh LT + RT | 36개 | 고관절 ROM(max-min), 평균 각속도 |
| Shank LT + RT | 40개 | 무릎 충격 피크 통계, 정강이 진동 |
| Foot LT + RT | 80개 | 착지 패턴, 보행 주파수, heel-strike 피크 |
| **합계** | **216개** | 기존 44개 대비 약 5배 확장 |

### 2.2 센서별 피처 상세

#### Pelvis (22개) — 몸통 안정성
- time stats (mean, RMS) × 6축 = 12개
- 벡터 크기 (accel / gyro) = 6개
- jerk RMS (accel Z) = 1개
- tilt 불안정성 (gyro std 평균) = 1개
- 지배 주파수 (accel Z, gyro Y) = 2개

#### Hand LT/RT (38개) — 팔 스윙 리듬
- time stats (mean, std, RMS) × 3ax × 2side = 18개
- 벡터 크기 × 2side = 6개
- 지배 주파수, 스펙트럼 엔트로피 × 2side = 4개
- gyro RMS × 3ax × 2side = 6개
- 좌우 대칭 상관계수 (accel 3ax + gyro Y) = 4개

#### Thigh LT/RT (36개) — 고관절
- accel time stats (mean, RMS) × 3ax × 2side = 12개
- accel 벡터 크기 × 2side = 6개
- gyro ROM (max-min) × 3ax × 2side = 6개
- gyro 평균 각속도 × 3ax × 2side = 6개
- 지배 주파수 (accel Z) × 2side = 2개
- 좌우 대칭 상관계수 = 4개

#### Shank LT/RT (40개) — 무릎 충격
- accel 피크 통계 (개수, 높이, 간격) × 3ax × 2side = 18개
- gyro time stats (mean, std, RMS) × 3ax × 2side = 18개
- 좌우 대칭 상관계수 = 4개

#### Foot LT/RT (80개) — 착지 패턴
- accel time stats × 3ax × 2side = 42개
- accel 교차축 (SMA, var_ratio, peak_ratio) × 2side = 6개
- 지배 주파수 × 3ax × 2side = 6개
- 주파수 대역 파워 (band_low/mid/high/hf) × 2side = 8개
- gyro RMS × 3ax × 2side = 6개
- gyro Y 지배 주파수 (시상면 리듬) × 2side = 2개
- heel-strike 피크 통계 (개수, 높이, 간격) × 2side = 6개
- 좌우 대칭 상관계수 = 4개

---

## 3. 모델 구성

7개 모델을 K-Fold로 비교합니다. Hybrid 모델(M7, Hierarchical)은 CNN/ResNetTCN과 216-feat FeatureMLP를 융합합니다.

| 모델 | 유형 | 특징 |
|------|------|------|
| M2 | CNN | 경량 2-block CNN |
| M4 | CNN | 4-block CNN + SE |
| M6 | CNN | 6-block CNN + CBAM |
| ResNet1D | ResNet | 1D Residual Network |
| CNNTCN | CNN+TCN | CNN + Temporal Convolutional Network |
| ResNetTCN | ResNet+TCN | ResNet + TCN 결합 |
| M7 (Hybrid) | CNN+Feature | M6 + 216-feat FeatureMLP 융합 (`IS_HYBRID=True`) |

### 3.1 FeatureMLP 구조

```
BatchNorm1d(216)
→ Linear(216, 128) → ReLU → Dropout(0.3)
→ Linear(128, feat_dim) → ReLU → Dropout(0.3)
```

---

## 4. 실험 파이프라인

### 4.1 전체 실행

```bash
chmod +x run_all_experiments.sh
mkdir -p experiments/logs

# 전체 실행
nohup ./run_all_experiments.sh all > experiments/logs/run.log 2>&1 &

# 개별 실행
./run_all_experiments.sh kfold
./run_all_experiments.sh loso
./run_all_experiments.sh hierarchical
```

| 단계 | 스크립트 | 내용 |
|------|---------|------|
| 1/3 | `train_kfold.py` | 7개 모델 × 5-Fold StratifiedGroupKFold 비교 |
| 2/3 | `train_loso.py` | 상위 4개 모델 × LOSO (피험자 독립 검증) |
| 3/3 | `train_hierarchical.py` | HierarchicalFusionNet 최종 실험 |

### 4.2 K-Fold 설정

- 분할: `StratifiedGroupKFold` (n_splits=5)
- 그룹 기준: 피험자 ID (피험자 간 데이터 누출 방지)
- 클래스 불균형: `BalancedSampler` + `FocalLoss(γ=2.0)`
- 정규화: `LabelSmoothing=0.1`, `Mixup=0.2`
- 조기 종료: `patience=7`

### 4.3 학습 설정

| 하이퍼파라미터 | 값 |
|--------------|-----|
| 배치 크기 | 512 |
| 에폭 | 50 |
| 학습률 | 0.001 → 1e-6 (CosineAnnealing) |
| Weight Decay | 0.001 |
| Dropout (clf/feat) | 0.5 / 0.3 |
| AMP | BF16 (NVIDIA L4) |
| TTA | ×5 |

### 4.4 모니터링

```bash
# 로그 실시간 확인
tail -f experiments/logs/kfold_run.log

# 프로세스 확인
ps aux | grep train_kfold | grep -v grep

# 프로세스 종료
kill <PID>
```

---

## 5. 주요 수정 이력 (v2.0)

| 파일 | 수정 내용 |
|------|----------|
| `features.py` | 44개(Foot 전용) → 216개(9센서 도메인 특화)로 전면 확장 |
| `datasets.py` | HDF5 v9 구조(`subjects/S{id}/X,y`) 대응 파싱 수정 |
| `train_kfold.py` | `feat44` → `feat`(N_FEATURES 동적 참조), config 속성명 수정 |
| `train_common.py` | `ensure_dir`, `save_json`, `Timer`, `save_summary_table`, `fit_model` alias 추가 |
| `channel_groups.py` | `get_foot_accel_idx` 함수 추가 |
| `config.py` | `models_kfold`, `result_tables` 미존재 → 하드코딩/경로 수정 |

---

## 6. 실험 결과 출력 구조

```
experiments/
├── kfold/
│   ├── M2/
│   ├── M4/
│   ├── M6/
│   ├── ResNet1D/
│   ├── CNNTCN/
│   ├── ResNetTCN/
│   ├── M7/
│   │   ├── fold1_best.pt
│   │   ├── ...
│   │   ├── fold5_best.pt
│   │   └── kfold_results.json   ← fold별 acc, f1, confusion matrix
│   └── tables/
│       └── comparison_table.csv
├── loso/
│   └── (동일 구조)
├── hierarchical/
└── logs/
    ├── kfold_YYYYMMDD_HHMMSS.log
    ├── loso_YYYYMMDD_HHMMSS.log
    └── hierarchical_YYYYMMDD_HHMMSS.log
```

---

*IMU 지형 분류 파이프라인 v2.0*