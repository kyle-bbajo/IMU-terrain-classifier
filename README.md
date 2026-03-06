# IMU Terrain Classifier v8.0

Full-body IMU 기반 지형 분류 딥러닝 파이프라인.
Noraxon MyoMotion 관성 센서 데이터로부터 힐스트라이크 스텝을 자동 분할하고, Multi-branch CNN(M1–M6)으로 지형을 분류합니다.

## 파이프라인 구조

```
raw_csv/  →  step_segmentation.py  →  dataset.h5  →  train_kfold.py / train_loso.py  →  결과
(Noraxon CSV)   (힐스트라이크 분할)    (Subject-group    (M1~M6 학습 + PCA 캐시)       (report/cm/json)
                 3-pass + 증분)         HDF5 v8)
```

## 모델 구조 (M1–M6)

| 모델 | 구조 | 핵심 | 논문 의의 |
|------|------|------|-----------|
| M1 | PCA(64ch) → 1D-CNN | 차원축소 baseline | 전통적 접근법 비교 기준 |
| M2 | 5-Branch CNN | 신체 부위별 독립 인코딩 | 해부학적 사전지식 반영 |
| M3 | M2 + SE block | 채널 중요도 가중치 | Squeeze-and-Excitation |
| M4 | M2 + CBAM | 채널 + 시간축 동시 어텐션 | 힐스트라이크 시점 학습 |
| M5 | M4 + Cross-Attention | 부위 간 상호작용 학습 | 발-무릎-골반 연쇄 반응 |
| M6 | M5 + Online Augmentation | 과적합 방지 (노이즈, 스케일, 마스크) | 소규모 데이터 대응 |

## v8.0 주요 변경사항

### 1. Subject-group HDF5 (step_segmentation.py)
- **이전**: 증분 시 기존 HDF5 전체를 `.tmp.h5`로 복사 후 append → N=100이면 매번 수십 GB I/O
- **현재**: `/subjects/S{sid}/X`, `/subjects/S{sid}/y` 형식으로 피험자별 저장 → `h5py.File("a")` append만 수행, 전체 복사 제거
- v7 flat 형식 자동 감지 (하위 호환)

### 2. PCA/BSC 디스크 캐시 (train_common.py)
- **이전**: fold마다 + 모델마다 sc.transform → pca.transform 반복 (CPU 병목)
- **현재**: fold 시작 시 변환 결과를 `.npy` memmap(fp16)으로 캐시 → M2가 branch 캐시 생성하면 M3~M6은 캐시 히트 (변환 1회만)
- fold 끝나면 `clear_fold_cache()` 자동 정리
- 누수 방지: train subject로만 fit한 scaler/PCA로 변환

### 3. Augmentation 벡터화 (models.py)
- **이전**: `for b in range(B)` batch 루프로 채널 마스킹
- **현재**: `scatter_` 연산으로 한 번에 처리 → 배치 커질수록 효과

### 4. 오류/메타 로깅 강화
- `train_model()` 반환값에 `meta` 딕셔너리 포함: OOM 횟수, NaN 이벤트, early stop 정보, 학습 시간, 오류 목록
- `summary.json`에 fold별 클래스 분포, 오류 집계, 모델별 메타 구조적 기록
- 오류 발생 시 경고 로그 자동 출력

### 5. LOSO 진행률 정확화
- **이전**: 체크포인트 복원 후 `fi`가 1부터 시작 → "LOSO 3/40" 표시되지만 실제 13번째 처리 중
- **현재**: skip-aware 카운터로 `[13/40]` 정확 표시, 남은 시간도 실제 미처리 수 기준 계산

### 6. 버전 통일
- 전 파일 v8.0 태그 일관화

## 데이터 구조

### 입력 CSV (Noraxon MR3)
```
data/raw_csv/
├── 20230101_S01C1T1.csv    # S{피험자}C{지형}T{시행}
└── ...
```
- 포맷: Noraxon MyoMotion CSV (skiprows=2), ~320채널 중 Accel+Gyro 54채널 사용, 200Hz
- 지형 조건: C1~C8 (최대 8종)
- 유연한 컬럼 매칭: 3-tier (exact → regex → normalized) 자동 정규화

### HDF5 스키마 v8 (batches/dataset.h5)
```
subjects/
├── S0001/
│   ├── X    : (n_steps, 256, 54)  float32
│   └── y    : (n_steps,)           int64
├── S0002/
│   ├── X, y
│   └── ...
channels  : (54,)  bytes
attrs: format="subject_group_v8", sample_rate, target_ts, n_classes
```

### 캐시 디렉토리 (cache/)
```
cache/
├── F1_train_flat.npy       # M1 PCA 변환 캐시 (fp16 memmap)
├── F1_test_flat.npy
├── F1_train_br_Pelvis.npy  # M2-M6 branch별 캐시
├── F1_train_y.npy          # 라벨
└── ...                     # fold 완료 시 자동 삭제
```

## 설치 및 실행

### 설치
```bash
pip install -r requirements.txt
```

### 실행
```bash
# 전체 파이프라인 (seg → kfold → loso)
bash run.sh 40 all

# 백그라운드 (SSH 끊어도 유지)
bash run.sh 40 all bg

# 개별 실행
python3 step_segmentation.py --n_subjects 40
python3 train_kfold.py --n_subjects 40
python3 train_loso.py --n_subjects 40

# N=40 → N=100 확장 (기존 40명 자동 스킵, 신규 60명만 추가)
python3 step_segmentation.py --n_subjects 100

# 전체 재생성 (기존 무시)
python3 step_segmentation.py --n_subjects 40 --force

# 테스트
python3 tests.py
```

### tmux 세션 관리
```bash
tmux attach -t train          # 실행 화면 확인
# Ctrl+B → D                  # 화면에서 나오기 (학습 계속)
tmux kill-session -t train     # 세션 종료
```

## 출력 결과

```
out_N40/
├── kfold/
│   ├── summary.json              # 결과 + fold별 메타 + 오류 집계
│   ├── config_snapshot.json      # 실험 재현용 config
│   ├── report_KF5_M1_CNN.txt     # classification report
│   ├── cm_KF5_M1_CNN.png         # confusion matrix
│   └── history_M1_CNN.png        # loss/acc 곡선
├── loso/
│   ├── summary_loso.json         # 결과 + per_subject + fold_meta
│   ├── per_subject_accuracy.csv  # 피험자별 정확도 표
│   ├── per_subject_heatmap.png   # 히트맵
│   └── checkpoint.json           # 중단/재개용 (완료 시 자동 삭제)
└── batches/
    ├── dataset.h5                # Subject-group HDF5
    ├── terrain_params.json       # 지면별 통계 파라미터
    └── step_log.json             # 스텝 검출 로그
```

## 파일 구조

| 파일 | 역할 |
|------|------|
| `config.py` | 전역 설정. 하드웨어 자동감지(GPU/RAM/vCPU), 배치/워커 자동 스케일링, argparse 오버라이드, config 스냅샷 |
| `channel_groups.py` | 54채널 Raw IMU → 5그룹(Pelvis/Hand/Thigh/Shank/Foot) 매핑 |
| `models.py` | M1~M6 CNN 아키텍처. SE/CBAM/Cross-Attention, 벡터화 augmentation |
| `step_segmentation.py` | 3-pass 힐스트라이크 분할. Subject-group HDF5, 유연한 컬럼 매칭, 증분 처리 |
| `train_common.py` | 공용 엔진. H5Data(v7/v8 호환), PCA 디스크 캐시, Preload/OTF/Cache 3모드 Dataset, 학습루프, 메타 로깅 |
| `train_kfold.py` | 5-Fold StratifiedGroupKFold. 피험자 누수 검증, fold별 메타 수집 |
| `train_loso.py` | Leave-One-Subject-Out. 체크포인트 재개, skip-aware 진행률, per-subject 히트맵 |
| `tests.py` | 단위 테스트 (모델/config/채널그룹/증강/유연매칭/멀티GPU) |
| `run.sh` | 파이프라인 실행 (S3 동기화, tmux 백그라운드, 환경 출력) |
| `requirements.txt` | 의존성 |

## 핵심 설계 결정

### 데이터 무결성
- **피험자 누수 방지**: StratifiedGroupKFold + overlap 검증 (위반 시 즉시 raise)
- **PCA 누수 방지**: train subject로만 scaler/PCA fit → test는 transform만
- **증분 일관성**: 채널 순서 고정 (첫 실행 시 결정, 이후 재사용)

### 성능 최적화
- **3모드 Dataset**: Cache(memmap) → Preload(RAM) → OTF(디스크) 자동 선택
- **AMP**: torch.cuda.amp.autocast + GradScaler 전체 적용
- **TF32**: Ada Lovelace/Ampere GPU에서 자동 활성화
- **하드웨어 적응**: GPU VRAM 기반 배치 자동 조정, vCPU 기반 워커 스케일링

### 안정성
- **CUDA OOM 복구**: 학습 중 OOM → cache 정리 + break (전체 파이프라인 중단 방지)
- **NaN 감지**: 손실값 NaN/Inf 시 로그 기록
- **LOSO 체크포인트**: fold별 저장, 중단 후 정확히 이어서 재개
- **오류 메타**: fold별 OOM 횟수, 오류 메시지, 학습 시간 구조적 기록

### 재현성
- **seed_everything()**: Python/NumPy/PyTorch/CUDA 시드 고정
- **config.snapshot()**: 매 실험에 전체 하이퍼파라미터 JSON 저장
- **argparse 오버라이드**: `--n_subjects`, `--seed`, `--batch`, `--epochs` 런타임 변경

## summary.json 구조 (v8.0)

```json
{
  "experiment": "kfold",
  "version": "v8.0",
  "config": { ... },
  "total_minutes": 45.2,
  "total_errors": 0,
  "total_oom_events": 0,
  "results": {
    "M1_CNN": { "acc": 0.8523, "f1": 0.8401 },
    "M6_CNN": { "acc": 0.9012, "f1": 0.8934 }
  },
  "fold_meta": [
    {
      "fold": 1,
      "train_subjects": [1,2,3,...],
      "test_subjects": [8,9],
      "train_samples": 3200,
      "test_samples": 800,
      "train_class_dist": {"0": 400, "1": 400, ...},
      "test_class_dist": {"0": 100, "1": 100, ...},
      "fold_time_min": 9.1,
      "errors": [],
      "models": {
        "M1_CNN": {
          "tag": "[F1][M1]",
          "oom_events": 0,
          "nan_events": 0,
          "early_stopped": true,
          "early_stop_epoch": 35,
          "total_epochs": 35,
          "best_val_loss": 0.4231,
          "train_time_sec": 120.5,
          "errors": []
        }
      }
    }
  ]
}
```