"""
tests.py — 단위 테스트 (v8.0)
═══════════════════════════════════════════════════════
피드백 항목 5번: 테스트 코드 부재 해결

실행:
    python3 tests.py
    python3 -m pytest tests.py -v  (pytest 설치 시)
═══════════════════════════════════════════════════════
"""
from __future__ import annotations

import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch


class TestConfig(unittest.TestCase):
    """config.py 설정값 검증."""

    def test_import(self) -> None:
        import config
        self.assertIsNotNone(config.DEVICE)

    def test_types(self) -> None:
        import config
        self.assertIsInstance(config.N_SUBJECTS, int)
        self.assertIsInstance(config.NUM_CLASSES, int)
        self.assertIsInstance(config.BATCH, int)
        self.assertIsInstance(config.LR, float)
        self.assertIsInstance(config.USE_PRELOAD, bool)

    def test_ranges(self) -> None:
        import config
        self.assertGreater(config.N_SUBJECTS, 0)
        self.assertEqual(config.NUM_CLASSES, 6)
        self.assertGreater(config.BATCH, 0)
        self.assertGreater(config.LR, 0)
        self.assertGreater(config.EPOCHS, 0)
        self.assertGreater(config.TS, 0)

    def test_chunk_sizes(self) -> None:
        import config
        self.assertGreater(config.H5_READ_CHUNK, 0)
        self.assertGreater(config.IPCA_CHUNK, 0)
        self.assertGreater(config.FLUSH_SIZE, 0)

    def test_paths_exist(self) -> None:
        import config
        self.assertTrue(config.ROOT.exists())
        self.assertTrue(config.BATCH_DIR.exists())


class TestChannelGroups(unittest.TestCase):
    """channel_groups.py 채널 분류 검증."""

    def test_basic_grouping(self) -> None:
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

    def test_empty_channels_raises(self) -> None:
        from channel_groups import build_branch_idx
        with self.assertRaises(ValueError):
            build_branch_idx([])

    def test_no_match_raises(self) -> None:
        from channel_groups import build_branch_idx
        with self.assertRaises(ValueError):
            build_branch_idx(["unknown_col_1", "unknown_col_2"])

    def test_index_integrity(self) -> None:
        from channel_groups import build_branch_idx
        channels = ["Pelvis X", "Hand Y", "Thigh Z", "Foot W"]
        branch_idx, branch_ch = build_branch_idx(channels)
        all_indices = []
        for indices in branch_idx.values():
            all_indices.extend(indices)
        # 모든 인덱스가 유효 범위
        for idx in all_indices:
            self.assertGreaterEqual(idx, 0)
            self.assertLess(idx, len(channels))


class TestModels(unittest.TestCase):
    """models.py 모델 forward pass 검증."""

    def test_m1_forward(self) -> None:
        from models import M1_FlatCNN
        import config
        model = M1_FlatCNN(in_ch=config.PCA_CH)
        x = torch.randn(2, config.PCA_CH, config.TS)
        out = model(x)
        self.assertEqual(out.shape, (2, config.NUM_CLASSES))

    def test_m1_extract(self) -> None:
        from models import M1_FlatCNN
        import config
        model = M1_FlatCNN()
        x = torch.randn(2, config.PCA_CH, config.TS)
        feat = model.extract(x)
        self.assertEqual(feat.shape, (2, config.FEAT_DIM))

    def test_branch_models(self) -> None:
        from models import (M2_BranchCNN, M3_BranchSE, M4_BranchCBAM,
                            M5_BranchCBAMCross, M6_BranchCBAMCrossAug)
        import config
        bc = {"Pelvis": 10, "Foot": 15, "Thigh": 12}
        bi = {k: torch.randn(2, ch, config.TS) for k, ch in bc.items()}
        for name, fn in [("M2", M2_BranchCNN), ("M3", M3_BranchSE),
                         ("M4", M4_BranchCBAM), ("M5", M5_BranchCBAMCross),
                         ("M6", M6_BranchCBAMCrossAug)]:
            model = fn(bc)
            out = model(bi)
            self.assertEqual(out.shape, (2, config.NUM_CLASSES), f"{name} 출력 shape 오류")

    def test_se_block(self) -> None:
        from models import SEBlock
        se = SEBlock(64)
        x = torch.randn(2, 64, 100)
        out = se(x)
        self.assertEqual(out.shape, x.shape)

    def test_cbam_residual(self) -> None:
        from models import CBAM
        cbam = CBAM(64)
        x = torch.randn(2, 64, 100)
        out = cbam(x)
        self.assertEqual(out.shape, x.shape)

    def test_augment_eval_mode(self) -> None:
        from models import augment
        x = torch.randn(2, 10, 256)
        out = augment(x, training=False)
        self.assertTrue(torch.equal(x, out), "training=False에서 augment는 원본 반환해야 함")

    def test_augment_train_mode(self) -> None:
        from models import augment
        x = torch.randn(2, 10, 256)
        out = augment(x, training=True)
        self.assertFalse(torch.equal(x, out), "training=True에서 augment는 변환해야 함")


class TestStepSegmentation(unittest.TestCase):
    """step_segmentation.py 검증."""

    def test_parse_filename(self) -> None:
        from step_segmentation import parse_filename
        sid, cond, trial = parse_filename("20230101_S01C3T2.csv")
        self.assertEqual(sid, 1)
        self.assertEqual(cond, 3)
        self.assertEqual(trial, 2)

    def test_parse_filename_no_trial(self) -> None:
        from step_segmentation import parse_filename
        sid, cond, trial = parse_filename("20230101_S05C2.csv")
        self.assertEqual(sid, 5)
        self.assertEqual(cond, 2)
        self.assertEqual(trial, 1)

    def test_parse_filename_invalid(self) -> None:
        from step_segmentation import parse_filename
        sid, cond, trial = parse_filename("random_file.csv")
        self.assertIsNone(sid)

    def test_resample_step(self) -> None:
        from step_segmentation import resample_step
        seg = np.random.randn(180, 10).astype(np.float32)
        out = resample_step(seg, 256)
        self.assertEqual(out.shape, (256, 10))
        self.assertEqual(out.dtype, np.float32)

    def test_resample_step_same_length(self) -> None:
        from step_segmentation import resample_step
        seg = np.random.randn(256, 5).astype(np.float32)
        out = resample_step(seg, 256)
        self.assertEqual(out.shape, (256, 5))

    def test_resample_step_with_nans(self) -> None:
        from step_segmentation import resample_step
        seg = np.random.randn(180, 5).astype(np.float32)
        seg[10:20, 0] = np.nan
        out = resample_step(seg, 256)
        self.assertFalse(np.isnan(out).any(), "리샘플 후 NaN이 남으면 안 됨")

    def test_detect_steps_empty(self) -> None:
        from step_segmentation import detect_steps
        signal = np.zeros(100)
        steps = detect_steps(signal, {}, 1)
        self.assertEqual(steps, [])

    def test_bandpass_filter_all_nan(self) -> None:
        from step_segmentation import bandpass_filter
        signal = np.full(200, np.nan)
        out = bandpass_filter(signal)
        self.assertTrue(np.isnan(out).all())


class TestTrainCommon(unittest.TestCase):
    """train_common.py 유틸 검증."""

    def test_mixup_alpha_zero(self) -> None:
        from train_common import mixup_data
        x = torch.randn(4, 10, 256)
        y = torch.tensor([0, 1, 2, 3])
        mixed, ya, yb, lam = mixup_data(x, y, alpha=0)
        self.assertEqual(lam, 1.0)
        self.assertTrue(torch.equal(mixed, x))

    def test_mixup_dict(self) -> None:
        from train_common import mixup_data
        x = {"a": torch.randn(4, 10, 256), "b": torch.randn(4, 5, 256)}
        y = torch.tensor([0, 1, 2, 3])
        mixed, ya, yb, lam = mixup_data(x, y, alpha=0.2)
        self.assertIsInstance(mixed, dict)
        self.assertIn("a", mixed)
        self.assertIn("b", mixed)

    def test_collate_branch(self) -> None:
        from train_common import collate_branch
        batch = [
            ({"a": torch.randn(5, 100), "b": torch.randn(3, 100)}, torch.tensor(0)),
            ({"a": torch.randn(5, 100), "b": torch.randn(3, 100)}, torch.tensor(1)),
        ]
        bi, ys = collate_branch(batch)
        self.assertEqual(bi["a"].shape, (2, 5, 100))
        self.assertEqual(ys.shape, (2,))


class TestColumnMatching(unittest.TestCase):
    """컬럼명 유연 매칭 테스트."""

    def test_exact_match(self) -> None:
        """정확한 이름이면 그대로 반환."""
        cols = ["Foot Accel Sensor X LT (mG)", "time", "Activity"]
        result = config.resolve_column(cols, "Foot Accel Sensor X LT (mG)")
        self.assertEqual(result, "Foot Accel Sensor X LT (mG)")

    def test_pattern_match(self) -> None:
        """정확 매칭 실패 시 패턴으로 찾기."""
        cols = ["Foot Acceleration X LT (mG)", "time"]
        import re
        pat = re.compile(r"(?i)foot.*accel.*x.*lt")
        result = config.resolve_column(cols, "Foot Accel Sensor X LT (mG)", pat)
        self.assertEqual(result, "Foot Acceleration X LT (mG)")

    def test_normalized_match(self) -> None:
        """공백/특수문자 무시 매칭."""
        cols = ["FootAccelSensorXLT(mG)", "time"]
        result = config.resolve_column(cols, "Foot Accel Sensor X LT (mG)")
        self.assertEqual(result, "FootAccelSensorXLT(mG)")

    def test_no_match_raises(self) -> None:
        """매칭 실패 시 KeyError."""
        cols = ["completely_different_column"]
        with self.assertRaises(KeyError):
            config.resolve_column(cols, "Foot Accel Sensor X LT (mG)")

    def test_resolve_drop_cols(self) -> None:
        """드롭 컬럼 유연 매칭."""
        cols = ["Time", "ACTIVITY", "marker_1", "Pelvis X"]
        drops = config.resolve_drop_cols(cols)
        self.assertIn("Time", drops)
        self.assertIn("ACTIVITY", drops)
        self.assertIn("marker_1", drops)
        self.assertNotIn("Pelvis X", drops)

    def test_resolve_foot_acc_exact(self) -> None:
        """발 가속도 정확 매칭."""
        cols = [
            "Foot Accel Sensor X LT (mG)",
            "Foot Accel Sensor Y LT (mG)",
            "Foot Accel Sensor Z LT (mG)",
        ]
        result = config.resolve_foot_acc_cols(cols, "LT")
        self.assertEqual(result["x"], "Foot Accel Sensor X LT (mG)")
        self.assertEqual(result["y"], "Foot Accel Sensor Y LT (mG)")
        self.assertEqual(result["z"], "Foot Accel Sensor Z LT (mG)")


class TestChannelGroupsCaseInsensitive(unittest.TestCase):
    """대소문자 무관 채널 그룹핑."""

    def test_lowercase_channels(self) -> None:
        """소문자 컬럼도 올바르게 분류."""
        from channel_groups import build_branch_idx
        cols = [
            "pelvis x", "pelvis y", "pelvis z",
            "pelvis rx", "pelvis ry", "pelvis rz",
            "thigh x lt", "thigh y lt", "thigh z lt",
            "thigh rx lt", "thigh ry lt", "thigh rz lt",
            "thigh x rt", "thigh y rt", "thigh z rt",
            "thigh rx rt", "thigh ry rt", "thigh rz rt",
        ]
        idx, ch = build_branch_idx(cols)
        self.assertIn("Pelvis", idx)
        self.assertIn("Thigh", idx)
        self.assertEqual(ch["Pelvis"], 6)
        self.assertEqual(ch["Thigh"], 12)


class TestHardwareScaling(unittest.TestCase):
    """하드웨어 스케일링 설정."""

    def test_n_gpu_exists(self) -> None:
        """N_GPU 설정 존재."""
        self.assertIsInstance(config.N_GPU, int)
        self.assertGreaterEqual(config.N_GPU, 0)

    def test_gpu_total_mem(self) -> None:
        """GPU 전체 메모리 합산."""
        self.assertIsInstance(config.GPU_TOTAL_MEM_GB, float)
        if config.N_GPU > 0:
            self.assertAlmostEqual(
                config.GPU_TOTAL_MEM_GB,
                config.GPU_MEM_GB * config.N_GPU,
                places=1,
            )

    def test_batch_scales_with_gpu(self) -> None:
        """배치 크기가 GPU 수에 비례."""
        if config.N_GPU > 1:
            # multi-GPU: batch = base × N_GPU
            self.assertTrue(config.BATCH >= 64 * config.N_GPU)

    def test_worker_scaling(self) -> None:
        """워커 수가 CPU 코어에 비례."""
        self.assertGreaterEqual(config.LOADER_WORKERS, 0)
        if config.USE_PRELOAD:
            self.assertGreater(config.LOADER_WORKERS, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)