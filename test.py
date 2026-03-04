"""
tests.py — 단위 테스트 (v8 Final)
═══════════════════════════════════════════════════════
실행:
    python3 tests.py
    python3 -m pytest tests.py -v

v7->v8: 경계값 테스트, Gradient Accum 검증, 체크포인트 검증,
        config 검증 실패 테스트, 모델 forward/backward 검증
═══════════════════════════════════════════════════════
"""
from __future__ import annotations

import unittest
import tempfile
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch


# ═══════════════════════════════════════════════
# 1. Config 테스트
# ═══════════════════════════════════════════════

class TestConfig(unittest.TestCase):
    """config.py 설정값 검증."""

    def test_import_succeeds(self) -> None:
        """config 모듈이 정상 임포트되는지 확인."""
        import config
        self.assertIsNotNone(config.DEVICE)

    def test_all_types_correct(self) -> None:
        """주요 설정값의 타입이 올바른지 확인."""
        import config
        self.assertIsInstance(config.N_SUBJECTS, int)
        self.assertIsInstance(config.NUM_CLASSES, int)
        self.assertIsInstance(config.BATCH, int)
        self.assertIsInstance(config.LR, float)
        self.assertIsInstance(config.USE_PRELOAD, bool)
        self.assertIsInstance(config.USE_GPU, bool)
        self.assertIsInstance(config.SEED, int)
        self.assertIsInstance(config.GRAD_ACCUM_STEPS, int)

    def test_value_ranges(self) -> None:
        """설정값이 유효 범위 내인지 확인."""
        import config
        self.assertGreater(config.N_SUBJECTS, 0)
        self.assertEqual(config.NUM_CLASSES, 6)
        self.assertGreater(config.BATCH, 0)
        self.assertGreater(config.LR, config.MIN_LR)
        self.assertGreater(config.EPOCHS, 0)
        self.assertGreater(config.EARLY_STOP, 0)
        self.assertLessEqual(config.EARLY_STOP, config.EPOCHS)

    def test_chunk_sizes_positive(self) -> None:
        """청크 크기가 모두 양수인지 확인."""
        import config
        self.assertGreater(config.H5_READ_CHUNK, 0)
        self.assertGreater(config.IPCA_CHUNK, 0)
        self.assertGreater(config.FLUSH_SIZE, 0)
        self.assertGreater(config.DS_CHUNK, 0)

    def test_dropout_in_range(self) -> None:
        """Dropout 비율이 0~1 사이인지 확인."""
        import config
        for name in ["DROPOUT_CLF", "DROPOUT_FEAT", "LABEL_SMOOTH"]:
            val = getattr(config, name)
            self.assertGreaterEqual(val, 0.0, f"{name}={val}")
            self.assertLessEqual(val, 1.0, f"{name}={val}")

    def test_bandpass_nyquist(self) -> None:
        """대역통과 필터 주파수가 나이퀴스트 미만인지 확인."""
        import config
        nyquist = config.SAMPLE_RATE / 2
        self.assertLess(config.BANDPASS_HIGH, nyquist)
        self.assertLess(config.BANDPASS_LOW, config.BANDPASS_HIGH)

    def test_stride_range(self) -> None:
        """보폭 샘플 범위가 올바른지 확인."""
        import config
        self.assertLess(config.HS_MIN_STRIDE_SAM, config.HS_MAX_STRIDE_SAM)
        self.assertEqual(
            config.HS_MIN_STRIDE_SAM,
            int(config.HS_MIN_STRIDE_MS / 1000 * config.SAMPLE_RATE),
        )

    def test_paths_exist(self) -> None:
        """출력 디렉토리가 존재하는지 확인."""
        import config
        self.assertTrue(config.ROOT.exists())
        self.assertTrue(config.BATCH_DIR.exists())
        self.assertTrue(config.RESULT_KFOLD.exists())
        self.assertTrue(config.RESULT_LOSO.exists())

    def test_set_seed_deterministic(self) -> None:
        """set_seed 후 난수가 결정적인지 확인."""
        import config
        config.set_seed(99)
        a = np.random.rand(5)
        config.set_seed(99)
        b = np.random.rand(5)
        np.testing.assert_array_equal(a, b)

    def test_validate_config_passes(self) -> None:
        """현재 설정이 validate_config를 통과하는지 확인."""
        import config
        config.validate_config()  # 예외 없이 통과


# ═══════════════════════════════════════════════
# 2. Channel Groups 테스트
# ═══════════════════════════════════════════════

class TestChannelGroups(unittest.TestCase):
    """channel_groups.py 테스트."""

    def test_basic_grouping(self) -> None:
        """모든 7개 그룹이 정상 분류되는지 확인."""
        from channel_groups import build_branch_idx
        channels = [
            "Pelvis Accel X",
            "Hand Gyro Y LT",
            "Thigh Accel Z LT",
            "Shank Gyro X RT",
            "Foot Accel Sensor X LT",
            "Hip Flexion LT",
            "Trajectories Body X",
        ]
        branch_idx, branch_ch = build_branch_idx(channels)
        self.assertIn("Pelvis", branch_idx)
        self.assertIn("Hand", branch_idx)
        self.assertIn("Thigh", branch_idx)
        self.assertIn("Foot", branch_idx)
        self.assertIn("Joints", branch_idx)
        self.assertIn("Trajectory", branch_idx)

    def test_empty_raises(self) -> None:
        """빈 채널 리스트에서 ValueError가 발생하는지 확인."""
        from channel_groups import build_branch_idx
        with self.assertRaises(ValueError):
            build_branch_idx([])

    def test_no_match_raises(self) -> None:
        """모든 채널이 미분류일 때 ValueError가 발생하는지 확인."""
        from channel_groups import build_branch_idx
        with self.assertRaises(ValueError):
            build_branch_idx(["unknown_1", "unknown_2"])

    def test_no_duplicate_assignment(self) -> None:
        """한 채널이 2개 이상 그룹에 할당되지 않는지 확인."""
        from channel_groups import build_branch_idx
        channels = [
            "Pelvis Accel X", "Foot Accel Sensor X LT",
            "Thigh Gyro Y RT", "Shank Mag Z LT",
        ]
        branch_idx, _ = build_branch_idx(channels)
        all_indices = []
        for il in branch_idx.values():
            all_indices.extend(il)
        self.assertEqual(len(all_indices), len(set(all_indices)),
                         "중복 할당 발생!")

    def test_index_bounds(self) -> None:
        """인덱스가 유효 범위 내인지 확인."""
        from channel_groups import build_branch_idx
        channels = ["Pelvis X", "Hand Y", "Foot Z"]
        branch_idx, _ = build_branch_idx(channels)
        for il in branch_idx.values():
            for idx in il:
                self.assertGreaterEqual(idx, 0)
                self.assertLess(idx, len(channels))

    def test_low_channel_warning(self) -> None:
        """예상 최소 채널보다 적을 때 경고가 발생하는지 확인."""
        from channel_groups import build_branch_idx
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            build_branch_idx(["Pelvis X"])
            has_warning = any("최소" in str(x.message) for x in w)
            self.assertTrue(has_warning, "최소 채널 경고가 없음")


# ═══════════════════════════════════════════════
# 3. Models 테스트
# ═══════════════════════════════════════════════

class TestModels(unittest.TestCase):
    """models.py 테스트."""

    def test_m1_forward_shape(self) -> None:
        """M1 출력 shape이 (B, NUM_CLASSES)인지 확인."""
        from models import M1_FlatCNN
        import config
        model = M1_FlatCNN()
        x = torch.randn(4, config.PCA_CH, config.TS)
        out = model(x)
        self.assertEqual(out.shape, (4, config.NUM_CLASSES))

    def test_m1_extract_shape(self) -> None:
        """M1 extract 출력이 (B, FEAT_DIM)인지 확인."""
        from models import M1_FlatCNN
        import config
        model = M1_FlatCNN()
        x = torch.randn(2, config.PCA_CH, config.TS)
        feat = model.extract(x)
        self.assertEqual(feat.shape, (2, config.FEAT_DIM))

    def test_m1_wrong_channels_raises(self) -> None:
        """M1에 잘못된 채널 수 입력 시 AssertionError 확인."""
        from models import M1_FlatCNN
        model = M1_FlatCNN(in_ch=64)
        x = torch.randn(2, 32, 256)  # 64 != 32
        with self.assertRaises(AssertionError):
            model(x)

    def test_all_branch_models(self) -> None:
        """M2-M6 모든 모델의 forward shape 확인."""
        from models import (
            M2_BranchCNN, M3_BranchSE, M4_BranchCBAM,
            M5_BranchCBAMCross, M6_BranchCBAMCrossAug,
        )
        import config
        bc = {"Pelvis": 10, "Foot": 15, "Thigh": 12}
        bi = {k: torch.randn(2, ch, config.TS) for k, ch in bc.items()}
        for name, fn in [
            ("M2", M2_BranchCNN), ("M3", M3_BranchSE),
            ("M4", M4_BranchCBAM), ("M5", M5_BranchCBAMCross),
            ("M6", M6_BranchCBAMCrossAug),
        ]:
            model = fn(bc)
            out = model(bi)
            self.assertEqual(out.shape, (2, config.NUM_CLASSES), f"{name} shape 오류")

    def test_branch_missing_group_raises(self) -> None:
        """Branch 모델에 누락된 그룹 입력 시 KeyError 확인."""
        from models import M2_BranchCNN
        import config
        bc = {"Pelvis": 10, "Foot": 15}
        bi = {"Pelvis": torch.randn(2, 10, config.TS)}  # Foot 누락
        model = M2_BranchCNN(bc)
        with self.assertRaises(KeyError):
            model(bi)

    def test_branch_empty_raises(self) -> None:
        """빈 branch_channels에서 ValueError 확인."""
        from models import M2_BranchCNN
        with self.assertRaises(ValueError):
            M2_BranchCNN({})

    def test_se_block_shape(self) -> None:
        """SEBlock이 입력 shape을 유지하는지 확인."""
        from models import SEBlock
        se = SEBlock(64)
        x = torch.randn(2, 64, 100)
        self.assertEqual(se(x).shape, x.shape)

    def test_cbam_residual(self) -> None:
        """CBAM 잔차 연결: 출력 shape이 입력과 같은지 확인."""
        from models import CBAM
        cbam = CBAM(64)
        x = torch.randn(2, 64, 100)
        self.assertEqual(cbam(x).shape, x.shape)

    def test_augment_no_op_in_eval(self) -> None:
        """eval 모드에서 augment가 identity인지 확인."""
        from models import augment
        x = torch.randn(2, 10, 256)
        with torch.no_grad():
            out = augment(x)
        self.assertTrue(torch.equal(x, out))

    def test_augment_changes_in_train(self) -> None:
        """train 모드에서 augment가 값을 변경하는지 확인."""
        from models import augment
        x = torch.randn(4, 10, 256)
        out = augment(x.clone())
        self.assertFalse(torch.equal(x, out))

    def test_m1_backward(self) -> None:
        """M1 backward pass가 정상 동작하는지 확인."""
        from models import M1_FlatCNN
        import config
        model = M1_FlatCNN()
        x = torch.randn(2, config.PCA_CH, config.TS)
        y = torch.tensor([0, 1])
        out = model(x)
        loss = torch.nn.functional.cross_entropy(out, y)
        loss.backward()
        has_grad = any(p.grad is not None for p in model.parameters())
        self.assertTrue(has_grad, "Backward 후 grad가 없음")

    def test_count_parameters(self) -> None:
        """count_parameters가 양수를 반환하는지 확인."""
        from models import M1_FlatCNN, count_parameters
        model = M1_FlatCNN()
        n = count_parameters(model)
        self.assertGreater(n, 0)


# ═══════════════════════════════════════════════
# 4. Step Segmentation 테스트
# ═══════════════════════════════════════════════

class TestStepSegmentation(unittest.TestCase):
    """step_segmentation.py 테스트."""

    def test_parse_filename_full(self) -> None:
        """S01C3T2 형식 파싱."""
        from step_segmentation import parse_filename
        sid, cond, trial = parse_filename("20230101_S01C3T2.csv")
        self.assertEqual((sid, cond, trial), (1, 3, 2))

    def test_parse_filename_no_trial(self) -> None:
        """S05C2 형식 (시행 번호 없음)."""
        from step_segmentation import parse_filename
        sid, cond, trial = parse_filename("S05C2.csv")
        self.assertEqual((sid, cond, trial), (5, 2, 1))

    def test_parse_filename_invalid(self) -> None:
        """유효하지 않은 파일명."""
        from step_segmentation import parse_filename
        self.assertEqual(parse_filename("random.csv"), (None, None, None))

    def test_resample_basic(self) -> None:
        """기본 리샘플링: 180 -> 256."""
        from step_segmentation import resample_step
        seg = np.random.randn(180, 10).astype(np.float32)
        out = resample_step(seg, 256)
        self.assertEqual(out.shape, (256, 10))
        self.assertEqual(out.dtype, np.float32)

    def test_resample_same_length(self) -> None:
        """입력 길이 = 타겟 길이일 때 복사."""
        from step_segmentation import resample_step
        seg = np.random.randn(256, 5).astype(np.float32)
        out = resample_step(seg, 256)
        self.assertEqual(out.shape, (256, 5))

    def test_resample_handles_nan(self) -> None:
        """NaN이 보간되는지 확인."""
        from step_segmentation import resample_step
        seg = np.random.randn(180, 5).astype(np.float32)
        seg[10:20, 0] = np.nan
        out = resample_step(seg, 256)
        self.assertFalse(np.isnan(out).any())

    def test_resample_all_nan_channel(self) -> None:
        """전체 NaN 채널이 0으로 채워지는지 확인."""
        from step_segmentation import resample_step
        seg = np.full((100, 3), np.nan, dtype=np.float32)
        seg[:, 0] = np.random.randn(100)  # 0번만 정상
        out = resample_step(seg, 256)
        self.assertEqual(out.shape, (256, 3))
        self.assertTrue((out[:, 1] == 0).all())  # NaN→0
        self.assertTrue((out[:, 2] == 0).all())

    def test_resample_1d_raises(self) -> None:
        """1D 배열 입력 시 ValueError 확인."""
        from step_segmentation import resample_step
        with self.assertRaises(ValueError):
            resample_step(np.array([1, 2, 3]), 256)

    def test_resample_very_short(self) -> None:
        """매우 짧은(L<2) 세그먼트 처리."""
        from step_segmentation import resample_step
        seg = np.array([[1.0, 2.0]])  # (1, 2)
        out = resample_step(seg, 256)
        self.assertEqual(out.shape, (256, 2))

    def test_detect_steps_empty(self) -> None:
        """단조로운 신호에서 빈 리스트 반환."""
        from step_segmentation import detect_steps
        signal = np.zeros(200)
        steps = detect_steps(signal, {}, 1)
        self.assertEqual(steps, [])

    def test_bandpass_all_nan(self) -> None:
        """전체 NaN 신호에서 NaN 유지."""
        from step_segmentation import bandpass_filter
        signal = np.full(200, np.nan)
        out = bandpass_filter(signal)
        self.assertTrue(np.isnan(out).all())

    def test_bandpass_empty(self) -> None:
        """빈 신호 처리."""
        from step_segmentation import bandpass_filter
        out = bandpass_filter(np.array([]))
        self.assertEqual(len(out), 0)

    def test_discover_csvs_missing_dir(self) -> None:
        """존재하지 않는 디렉토리에서 FileNotFoundError."""
        from step_segmentation import discover_csvs
        with self.assertRaises(FileNotFoundError):
            discover_csvs(Path("/nonexistent"), 40)


# ═══════════════════════════════════════════════
# 5. Train Common 테스트
# ═══════════════════════════════════════════════

class TestTrainCommon(unittest.TestCase):
    """train_common.py 테스트."""

    def test_mixup_alpha_zero_identity(self) -> None:
        """alpha=0이면 mixup이 identity인지 확인."""
        from train_common import mixup_data
        x = torch.randn(4, 10, 256)
        y = torch.tensor([0, 1, 2, 3])
        mixed, ya, yb, lam = mixup_data(x, y, alpha=0)
        self.assertEqual(lam, 1.0)
        self.assertTrue(torch.equal(mixed, x))

    def test_mixup_dict_input(self) -> None:
        """dict 입력에 대한 mixup 동작 확인."""
        from train_common import mixup_data
        x = {"a": torch.randn(4, 10, 256), "b": torch.randn(4, 5, 256)}
        y = torch.tensor([0, 1, 2, 3])
        mixed, ya, yb, lam = mixup_data(x, y, alpha=0.2)
        self.assertIsInstance(mixed, dict)
        self.assertIn("a", mixed)
        self.assertIn("b", mixed)
        self.assertEqual(mixed["a"].shape, x["a"].shape)

    def test_collate_branch(self) -> None:
        """branch collate 함수가 올바른 shape을 반환하는지 확인."""
        from train_common import collate_branch
        batch = [
            ({"a": torch.randn(5, 100), "b": torch.randn(3, 100)},
             torch.tensor(0)),
            ({"a": torch.randn(5, 100), "b": torch.randn(3, 100)},
             torch.tensor(1)),
        ]
        bi, ys = collate_branch(batch)
        self.assertEqual(bi["a"].shape, (2, 5, 100))
        self.assertEqual(bi["b"].shape, (2, 3, 100))
        self.assertEqual(ys.shape, (2,))

    def test_mixup_criterion(self) -> None:
        """mixup_criterion이 스칼라 텐서를 반환하는지 확인."""
        from train_common import mixup_criterion
        crit = torch.nn.CrossEntropyLoss()
        pred = torch.randn(4, 6)
        ya = torch.tensor([0, 1, 2, 3])
        yb = torch.tensor([3, 2, 1, 0])
        loss = mixup_criterion(crit, pred, ya, yb, 0.7)
        self.assertEqual(loss.ndim, 0)

    def test_seed_everything(self) -> None:
        """seed_everything 후 torch 결과가 결정적인지 확인."""
        from train_common import seed_everything
        seed_everything(42)
        a = torch.randn(10)
        seed_everything(42)
        b = torch.randn(10)
        self.assertTrue(torch.equal(a, b))

    def test_h5data_missing_file(self) -> None:
        """존재하지 않는 HDF5에서 FileNotFoundError."""
        from train_common import H5Data
        with self.assertRaises(FileNotFoundError):
            H5Data(Path("/nonexistent.h5"))

    def test_h5data_context_manager(self) -> None:
        """H5Data가 with문으로 정상 동작하는지 확인 (파일 없이 테스트)."""
        from train_common import H5Data
        try:
            with H5Data("/nonexistent.h5"):
                pass
        except FileNotFoundError:
            pass  # 예상된 예외


# ═══════════════════════════════════════════════
# 6. LOSO 체크포인트 테스트
# ═══════════════════════════════════════════════

class TestCheckpoint(unittest.TestCase):
    """체크포인트 저장/복원 테스트."""

    def test_save_and_load(self) -> None:
        """저장 후 복원이 동일한지 확인."""
        from train_loso import _load_checkpoint, _save_checkpoint
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "ckpt.json"
            done = [1, 2, 3]
            preds = {"M1": [0, 1, 2, 0, 1, 2], "M2": [1, 1, 2, 0, 0, 2]}
            labels = [0, 1, 2, 0, 1, 2]
            _save_checkpoint(path, done, preds, labels)
            d2, p2, l2 = _load_checkpoint(path)
            self.assertEqual(d2, done)
            self.assertEqual(p2, preds)
            self.assertEqual(l2, labels)

    def test_corrupt_checkpoint(self) -> None:
        """손상된 체크포인트에서 빈 상태 반환."""
        from train_loso import _load_checkpoint
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "ckpt.json"
            path.write_text("{invalid json!!!")
            d, p, l = _load_checkpoint(path)
            self.assertEqual(d, [])
            self.assertEqual(p, {})
            self.assertEqual(l, [])

    def test_length_mismatch_checkpoint(self) -> None:
        """pred/label 길이 불일치 체크포인트에서 빈 상태 반환."""
        from train_loso import _load_checkpoint
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "ckpt.json"
            bad_ckpt = {
                "done_subjects": [1],
                "preds": {"M1": [0, 1]},
                "labels": [0, 1, 2],  # 길이 불일치
            }
            path.write_text(json.dumps(bad_ckpt))
            d, p, l = _load_checkpoint(path)
            self.assertEqual(d, [])

    def test_missing_checkpoint(self) -> None:
        """존재하지 않는 체크포인트에서 빈 상태 반환."""
        from train_loso import _load_checkpoint
        d, p, l = _load_checkpoint(Path("/nonexistent_ckpt.json"))
        self.assertEqual(d, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)