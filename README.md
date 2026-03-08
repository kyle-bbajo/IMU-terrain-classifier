# IMU Terrain Classifier

보행 중 지면 유형을 IMU(관성측정장치) 신호만으로 실시간 분류하는 시스템.

하지 보조기·의족·재활 로봇의 보행 제어를 위한 **downstream task 전제**: 지형 인식 → 보조기 강성/댐핑 자동 조절 → 낙상 예방 및 에너지 효율 개선.

---

## Background & Motivation

하지 절단 환자가 착용하는 로봇 보조기는 지면 상태(평지/오르막/계단/잔디 등)에 따라 능동적으로 제어 파라미터를 바꿔야 한다. 기존 시스템은 visual sensor(카메라)와 IMU를 함께 사용하지만, **IMU 단독으로 지형 분류가 가능한지** 검증하는 것이 이 프로젝트의 핵심 질문이다.

- 참고: Niswander et al. (2021) — IMU 기반 발 충격 생체역학 분석
- 참고: Ordóñez & Roggen (2016) — 계층적 HAR (Hierarchical Activity Recognition)

---

## 분류 대상 지형 (6 Classes)

| 레이블 | 지형 | 특징 |
|--------|------|------|
| C1 | 미끄러운 지면 | 보행 불안정, Foot 신호 변동성↑ |
| C2 | 오르막 | 경사각 양수, Shank 전방 기울기↑ |
| C3 | 내리막 | 경사각 음수, 충격 피크↑ |
| C4 | 흙길 | 불규칙 진동, Shank 고주파↑ |
| C5 | 잔디 | 충격 감쇠 빠름 |
| C6 | 평지 | 기준 클래스 |

---

## Dataset

| 항목 | 내용 |
|------|------|
| 수집 장비 | Noraxon MyoMotion 전신 IMU |
| 피험자 | 40명 |
| 센서 위치 | Foot, Shank, Thigh, Pelvis, Hand (양측, 54채널) |
| 샘플링 | 200 Hz |
| 총 스텝 수 | ~45,073 스텝 |
| 클래스 불균형 | C2(오르막), C3(내리막) 소수 → balanced sampler + class weight 대응 |

### 데이터 전처리 파이프라인

```
Raw CSV (200Hz)
    │
    ▼
[1] Heel-Strike 검출
    · Foot Z-축 가속도 피크 기반 보행 이벤트 검출
    · 스텝 경계 추출 → 가변 길이 시퀀스를 고정 길이(TS=256)로 정규화
    │
    ▼
[2] BSC (Batch Standardization & Calibration)
    · 피험자별 센서 바이어스 보정
    · train fold 기준 StandardScaler fit → val/test에 동일 적용
    · subject leakage 방지
    │
    ▼
[3] HDF5 저장
    · dataset.h5: (N, 54, 256) 형태
    · subject ID / label 별도 저장
    │
    ▼
[4] Branch 분리
    · Foot, Shank, Thigh, Pelvis, Hand 5개 그룹으로 채널 분리
    · 부위별 독립 CNN branch 입력
```

---

## Project Structure

```
project/
├── data/
│   ├── raw_csv/                    # 원본 CSV
│   └── processed/batches/
│       └── dataset.h5              # 전처리 완료 HDF5
│
└── repo/
    ├── src/                        # 공용 모듈
    │   ├── config.py               # 전역 설정 (경로, 하이퍼파라미터)
    │   ├── models.py               # M1~M6 모델 정의
    │   ├── train_common.py         # 공통 학습 유틸
    │   ├── channel_groups.py       # IMU 채널 그룹 정의
    │   └── step_segmentation.py    # Heel-Strike 분절 및 HDF5 생성
    │
    ├── experiments/
    │   ├── baseline/
    │   │   ├── train_kfold.py      # M1~M6 K-Fold 비교
    │   │   └── train_loso.py       # LOSO (Leave-One-Subject-Out)
    │   ├── supcon/
    │   │   └── train_supcon.py     # Supervised Contrastive Learning
    │   └── hierarchical/
    │       └── train_hierarchical.py  # Hierarchical SupCon ← 권장
    │
    ├── logs/                       # 학습 로그
    ├── out_N40/                    # 실험 결과
    │   └── kfold/hierarchical/
    │       ├── summary_v88.json
    │       ├── report_v88.txt
    │       ├── confusion_matrix.png
    │       └── curves/             # 훈련 곡선 (loss/acc PNG + JSON)
    └── tests.py
```

---

## Requirements

```bash
pip install -r requirements.txt
```

- Python 3.10+
- PyTorch 2.1.0 (CUDA 11.8+ 권장)
- 전체 패키지 버전은 `requirements.txt` 참조

---

## Usage

### Step 1. 데이터 전처리 (최초 1회)

```bash
python src/step_segmentation.py
```

`data/raw_csv/` → `data/processed/batches/dataset.h5` 생성.

### Step 2. 학습

#### Hierarchical SupCon (권장)

```bash
cd experiments/hierarchical
python train_hierarchical.py
```

**CLI 옵션으로 하이퍼파라미터 오버라이드:**

```bash
# 기본 실행
python train_hierarchical.py

# 에포크 조정
python train_hierarchical.py --s1_epochs 80 --s2_warmup 80

# temperature 실험
python train_hierarchical.py --temperature 0.05

# majority vote 비활성
python train_hierarchical.py --vote_window 0

# 전체 옵션 확인
python train_hierarchical.py --help
```

**백그라운드 실행:**

```bash
nohup python train_hierarchical.py > ~/project/repo/logs/hierarchical.log 2>&1 &
tail -f ~/project/repo/logs/hierarchical.log
```

#### Baseline 비교

```bash
python experiments/baseline/train_kfold.py   # K-Fold
python experiments/baseline/train_loso.py    # LOSO (논문 제출 기준)
```

#### SupCon

```bash
python experiments/supcon/train_supcon.py
```

---

## Model Architecture

### Hierarchical Hard-Routing (v8.8)

```
IMU 신호 (54ch, 256 samples)
    │
    ├─ [Stage 1] 3cls CE
    │   Backbone: M6 (Branch CNN + CBAM + Cross-Attention)
    │   Output: 평탄(0) / 오르막(1) / 내리막(2)
    │   Acc: ~99%
    │
    └─ [Stage 2] 4cls (평탄 클래스만)
        Backbone: M6 (독립 학습, S2_INIT_FROM_S1=False)
        + BioMech 16-dim 피처 결합
        학습 4단계:
          Step1: CE Warmup   (60ep, LR=1e-4)
          Step2: SupCon      (100ep, T=0.07, balanced sampler)
          Step3: Focal Loss  (100ep, γ=2.0, fc 레이어 부분 해제)
          Step4: CE 마무리   (50ep)
        Output: C1 / C4 / C5 / C6
    │
    └─ Hard Routing 결합 + Majority Vote (window=5)
```

### BioMech 16 Features (생체역학 피처)

Heel-Strike 기반 스텝 분절에서 추출하는 16개 도메인 특징:

| 피처 | 설명 | 관련 클래스 |
|------|------|-------------|
| 0~3 | Foot/Shank LT/RT 충격 피크값 | 전체 |
| 4~5 | Foot/Shank 충격비 | 지면 흡수량 |
| 6~7 | 상대 고주파 에너지 비율 (FFT) | C4 흙길 |
| 8~9 | Foot 신호 표준편차 | C1 미끄러운 |
| 10~11 | 피크 후 감쇠율 | C5 잔디 |
| 12~13 | Shank 진동 (∣diff∣.mean) | C4 흙길 |
| 14~15 | Foot/Shank 분산비 | C1 불안정 |

---

## Validation Protocol

| 항목 | 내용 |
|------|------|
| K-Fold | StratifiedGroupKFold (K=5), 피험자 단위 분할 |
| LOSO | Leave-One-Subject-Out (논문 기준) |
| Inner val | 각 fold 내 15% subject → early stopping |
| Subject leakage | 완전 방지 (train/val/test 피험자 분리) |
| 보고 방식 | Oracle acc (정답 라우팅) / Pipeline acc (실제 예측) 분리 |

---

## Experiment Results

| 모델 | Acc | 비고 |
|------|-----|------|
| M6 Branch+CBAM+Cross+Aug | 75.6% | Baseline |
| SupCon v2 | 80.6% | Contrastive |
| Hierarchical v8.5 (Stage1) | 83.2% | 최초 계층 구조 |
| Hierarchical v8.8 | 진행중 | 4단계 학습 + MajorityVote |

### Ablation (v8.7 → v8.8)

| 변경 | 이유 |
|------|------|
| S2_INIT_FROM_S1: True → False | Stage1은 slope 최적화 → flat 4cls에 역효과 확인 |
| S2_LR: 3e-5 → 1e-4 | warmup 56.6% 수렴 부족 |
| S2_WARMUP_EP: 30 → 60 | 워밍업 기간 연장 |
| T: 0.12 → 0.07 | SupCon 수렴 안정화 |
| Focal patience: 7 → 25 | ep9 조기종료 방지 |
| S2_WEIGHTS: 수동 → auto | compute_class_weight, fold별 자동 최적화 |
| Majority Vote 추가 | 보행 연속성 기반 노이즈 제거 |

---

## Output Files

```
out_N40/kfold/hierarchical/
├── summary_v88.json          # 전체 결과 + fold별 메타
├── report_v88.txt            # per-class precision/recall/F1
├── confusion_matrix.png      # 혼동 행렬
└── curves/
    ├── curve_S1_F1Hier.png   # Stage1 훈련 곡선
    ├── curve_S2W_F1Hier.png  # Stage2 Warmup 곡선
    ├── curve_S2FT_F1Hier.png # Stage2 Finetune 곡선
    └── *.json                # 곡선 원본 데이터
```

---

## References

1. Niswander et al. (2021) — Foot IMU biomechanics & impact analysis
2. Ordóñez & Roggen (2016) IEEE TNNLS — Hierarchical HAR
3. Khosla et al. (2020) NeurIPS — Supervised Contrastive Learning
4. Lin et al. (2017) ICCV — Focal Loss for Dense Object Detection