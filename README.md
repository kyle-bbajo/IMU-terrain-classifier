# IMU Terrain Classifier

Full-body IMU 기반 지형 분류 파이프라인.  
Noraxon MyoMotion 관성 센서 데이터로부터 힐스트라이크 스텝을 자동 분할하고, Multi-branch CNN으로 6종 지형을 분류합니다.

## 파이프라인 구조

```
raw_csv/  →  step_segmentation.py  →  dataset.h5  →  train_kfold.py / train_loso.py  →  결과
(Noraxon CSV)   (힐스트라이크 분할)     (HDF5)         (M1~M6 학습)                     (report/cm/json)
```

## 모델 구조 (M1–M6)

| 모델 | 구조 | 설명 |
|------|------|------|
| M1 | PCA → Flat CNN | Baseline. 305ch → 64ch PCA 후 단일 CNN |
| M2 | 7-Branch CNN | 신체 부위별 7개 브랜치 |
| M3 | M2 + SE | Squeeze-and-Excitation 채널 어텐션 |
| M4 | M2 + CBAM | Convolutional Block Attention Module |
| M5 | M4 + Cross-Attn | 브랜치 간 Multi-head Self-Attention |
| M6 | M5 + Augmentation | 온라인 데이터 증강 |

## 데이터 구조

### 입력 CSV (Noraxon MR3)
```
data/raw_csv/
├── 20230101_S01C1T1.csv    # S{피험자}C{지형}T{시행}
└── ...
```
- 포맷: Noraxon MyoMotion CSV (skiprows=2), ~320채널, 200Hz
- 지형 조건: C1~C6 (6종)

### HDF5 스키마 (`batches/dataset.h5`)
```
X          : (N_steps, 256, 305)   float32
y          : (N_steps,)            int64     — 라벨 (0~5)
subject_id : (N_steps,)            int64
channels   : (305,)                bytes
```

## 설치

```bash
git clone https://github.com/kyle-bbajo/IMU-terrain-classifier.git
cd IMU-terrain-classifier
pip install -r requirements.txt
```

## 실행

```bash
# 전체 파이프라인
bash run.sh 40 all

# 개별 실행 (argparse)
python3 step_segmentation.py --n_subjects 40
python3 train_kfold.py --n_subjects 40
python3 train_loso.py --n_subjects 40

# 증분 스텝 분할 (N=40 → N=100 확장 시 기존 40명 스킵)
python3 step_segmentation.py --n_subjects 100

# 테스트
python3 tests.py
```

## 출력 결과

```
out_N40/
├── kfold/
│   ├── summary.json, config_snapshot.json
│   ├── report_KF5_M1_CNN.txt, cm_KF5_M1_CNN.png
│   └── history_M1_CNN.png
├── loso/
│   ├── summary_loso.json, config_snapshot.json
│   ├── per_subject_accuracy.csv, per_subject_heatmap.png
│   └── checkpoint.json (중단/재개용)
└── batches/
    ├── dataset.h5, terrain_params.json, step_log.json
```

## 파일 설명

| 파일 | 역할 |
|------|------|
| `config.py` | 전역 설정 (하드웨어 감지, apply_overrides, snapshot) |
| `channel_groups.py` | 305채널 → 7그룹 매핑 |
| `models.py` | M1~M6 CNN 아키텍처 |
| `step_segmentation.py` | 3-패스 힐스트라이크 분할 (증분 지원) |
| `train_common.py` | 공용 유틸 (HDF5, PCA, Dataset, 학습루프) |
| `train_kfold.py` | 5-Fold StratifiedGroupKFold |
| `train_loso.py` | LOSO (체크포인트 지원) |
| `tests.py` | 단위 테스트 |
| `run.sh` | 파이프라인 실행 스크립트 |

## 주요 설계 결정

- **피험자 누수 방지**: StratifiedGroupKFold + overlap 검증
- **Dual-mode Dataset**: RAM ≥24GB → Preload / <24GB → OTF 자동 전환
- **증분 세그먼테이션**: 기존 HDF5의 피험자 스킵, 신규만 추가
- **LOSO 체크포인트**: Spot 인스턴스 중단 후 재개 가능
- **Config 스냅샷**: 매 실험에 config + git commit hash 자동 저장
