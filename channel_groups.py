"""
channel_groups.py — Noraxon MyoMotion 채널 그룹핑 (v8.0)
═══════════════════════════════════════════════════════
v7→v8: 중복 할당 방지, 그룹별 최소 채널 경고, 정렬 순서 보장

그룹 (7개):
    Pelvis, Hand, Thigh, Shank, Foot, Joints, Trajectory
═══════════════════════════════════════════════════════
"""
from __future__ import annotations

import warnings
from typing import Callable

_JOINT_KEYWORDS: tuple[str, ...] = (
    "Hip Flexion", "Hip Abduction", "Hip Rotation",
    "Knee Flexion", "Knee Abduction", "Knee Rotation",
    "Ankle Dorsiflexion", "Ankle Abduction", "Ankle Inversion",
)

# 대소문자 무관 매칭 (Noraxon 버전/내보내기 설정 차이 대응)
def _ci(keyword: str) -> Callable[[str], bool]:
    """Case-insensitive substring match."""
    kw_lower = keyword.lower()
    return lambda c: kw_lower in c.lower()

GROUP_RULES: list[tuple[str, Callable[[str], bool]]] = [
    ("Pelvis",     _ci("Pelvis")),
    ("Hand",       _ci("Hand")),
    ("Thigh",      _ci("Thigh")),
    ("Shank",      _ci("Shank")),
    ("Foot",       _ci("Foot")),
    ("Joints",     lambda c: any(k.lower() in c.lower() for k in _JOINT_KEYWORDS)),
    ("Trajectory", lambda c: any(
        k.lower() in c.lower()
        for k in ("Trajectories", "Body Orientation", "Body center")
    )),
]

_MIN_EXPECTED: dict[str, int] = {
    "Pelvis": 5, "Hand": 5, "Thigh": 10, "Shank": 10,
    "Foot": 10, "Joints": 5, "Trajectory": 10,
}


def build_branch_idx(
    channels: list[str],
) -> tuple[dict[str, list[int]], dict[str, int]]:
    """채널 리스트를 신체 부위 그룹별 인덱스로 분류한다.

    각 채널은 GROUP_RULES 순서대로 **첫 매칭 그룹에만** 할당된다.

    Parameters
    ----------
    channels : list[str]
        CSV 컬럼명 리스트 (time/Activity/Marker 제외 상태).

    Returns
    -------
    branch_idx : dict[str, list[int]]
        그룹명 -> 채널 인덱스 리스트.
    branch_ch : dict[str, int]
        그룹명 -> 채널 수.

    Raises
    ------
    ValueError
        channels가 비어 있거나 모든 채널이 미분류일 때.
    """
    if not channels:
        raise ValueError("channels 리스트가 비어 있습니다.")

    groups: dict[str, list[int]] = {name: [] for name, _ in GROUP_RULES}
    unassigned: list[tuple[int, str]] = []

    for i, ch in enumerate(channels):
        matched = False
        for gname, rule in GROUP_RULES:
            if rule(ch):
                groups[gname].append(i)
                matched = True
                break
        if not matched:
            unassigned.append((i, ch))

    # 빈 그룹 제거 (GROUP_RULES 순서 보존)
    branch_idx: dict[str, list[int]] = {}
    branch_ch: dict[str, int] = {}
    for name, _ in GROUP_RULES:
        if groups[name]:
            branch_idx[name] = groups[name]
            branch_ch[name]  = len(groups[name])

    if not branch_idx:
        raise ValueError(
            f"분류된 채널이 없습니다. 전체 {len(channels)}개 모두 미분류.\n"
            f"첫 5개: {channels[:5]}"
        )

    for gname, min_ch in _MIN_EXPECTED.items():
        actual = branch_ch.get(gname, 0)
        if 0 < actual < min_ch:
            warnings.warn(
                f"채널 그룹 '{gname}': {actual}ch (최소 {min_ch}ch 예상)",
                stacklevel=2,
            )

    total_assigned = sum(branch_ch.values())
    print(f"  Channel Groups ({len(channels)} total -> {total_assigned} assigned):")
    for nm in branch_idx:
        print(f"    {nm:<12}: {branch_ch[nm]:>3}ch")
    if unassigned:
        print(f"    (미분류     : {len(unassigned):>3}ch)")

    return branch_idx, branch_ch