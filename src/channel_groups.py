"""
channel_groups.py — 9센서 Raw IMU 채널 그룹핑 (v8.1)
═══════════════════════════════════════════════════════
v8.0→v8.1: 305ch 전체 → 54ch Raw (Accel+Gyro) 전환
            7그룹 → 5그룹 (해부학적 영역)
            9센서: 골반, 양손, 양허벅지, 양정강이, 양발

센서 데이터:
    Accelerometer X/Y/Z (mG)  — 3축 가속도
    Gyroscope X/Y/Z (deg/s)   — 3축 각속도
    = 6ch/센서 × 9센서 = 54ch

그룹:
    Pelvis : 골반 (6ch)
    Hand   : 양손 (12ch = LT 6 + RT 6)
    Thigh  : 양허벅지 (12ch)
    Shank  : 양정강이 (12ch)
    Foot   : 양발 (12ch)
═══════════════════════════════════════════════════════
"""
from __future__ import annotations


# ─────────────────────────────────────────────
# 센서-그룹 정의
# ─────────────────────────────────────────────

GROUPS: dict[str, list[str]] = {
    "Pelvis": ["pelvis"],
    "Hand":   ["hand lt", "hand rt"],
    "Thigh":  ["thigh lt", "thigh rt"],
    "Shank":  ["shank lt", "shank rt"],
    "Foot":   ["foot lt", "foot rt"],
}

# 채널 타입 키워드 (이 키워드를 포함하는 채널만 사용)
# "Accel Sensor" = 짧은 이름, "Acceleration" = Noraxon 긴 이름
RAW_CHANNEL_KEYWORDS: list[str] = ["accel sensor", "acceleration", "gyroscope"]


def is_raw_imu_channel(col: str) -> bool:
    """채널명이 Raw IMU (Accel/Gyro) 채널인지 판단한다."""
    cl = col.lower()
    return any(kw in cl for kw in RAW_CHANNEL_KEYWORDS)


def get_sensor_part(col: str) -> str | None:
    """채널명에서 센서 부위를 추출한다.

    "Hand Accel Sensor X LT (mG)" 처럼 부위명과 좌우가 떨어져 있어도 매칭.
    """
    cl = col.lower()

    # 부위 키워드
    body_parts = ["hand", "thigh", "shank", "foot"]
    sides = ["lt", "rt"]

    for part in body_parts:
        if part in cl:
            for side in sides:
                # "hand ... lt" 또는 "hand-lt" 모두 매칭
                if side in cl:
                    return f"{part} {side}"
            # 좌우 구분 없이 부위만 있으면 (예외 케이스)
            return None

    if "pelvis" in cl:
        return "pelvis"
    return None


def filter_raw_channels(all_channels: list[str]) -> list[str]:
    """전체 채널 목록에서 Raw IMU 채널만 필터링한다.

    중복 제거 및 순서 보존. 같은 센서의 Accel Sensor / Acceleration
    형태가 모두 있을 경우 먼저 나오는 것만 유지.
    """
    raw_channels: list[str] = []
    seen_parts: set[str] = set()  # "pelvis_accel_x" 형태로 중복 체크

    for c in all_channels:
        if not is_raw_imu_channel(c):
            continue
        part = get_sensor_part(c)
        if part is None:
            continue

        # 축(x/y/z) + 타입(accel/gyro) 키 생성
        cl = c.lower()
        axis = "x" if "x" in cl.split("(")[0].split("-")[-1] else (
               "y" if "y" in cl.split("(")[0].split("-")[-1] else "z")
        ch_type = "accel" if ("accel" in cl or "acceleration" in cl) else "gyro"
        dedup_key = f"{part}_{ch_type}_{axis}"

        if dedup_key not in seen_parts:
            seen_parts.add(dedup_key)
            raw_channels.append(c)

    return raw_channels


def build_branch_idx(
    channels: list[str],
) -> tuple[dict[str, list[int]], dict[str, int]]:
    """채널 목록에서 5개 그룹의 인덱스 매핑을 생성한다."""
    branch_idx: dict[str, list[int]] = {nm: [] for nm in GROUPS}

    for i, col in enumerate(channels):
        part = get_sensor_part(col)
        if part is None:
            continue
        for grp_name, keywords in GROUPS.items():
            if part in keywords:
                branch_idx[grp_name].append(i)
                break

    branch_idx = {k: sorted(v) for k, v in branch_idx.items() if v}
    branch_ch  = {k: len(v) for k, v in branch_idx.items()}

    total = sum(branch_ch.values())
    print(f"  Channel Groups ({len(channels)} total -> {total} assigned):")
    for nm in GROUPS:
        if nm in branch_ch:
            print(f"    {nm:<12}: {branch_ch[nm]:3d}ch")

    return branch_idx, branch_ch
def get_foot_accel_idx(channels: list[str]) -> list[int]:
    """Foot 센서의 Accelerometer 채널 인덱스만 반환한다."""
    idx = []
    for i, col in enumerate(channels):
        part = get_sensor_part(col)
        if part in ("foot lt", "foot rt"):
            cl = col.lower()
            if "accel" in cl or "acceleration" in cl:
                idx.append(i)
    return sorted(idx)
