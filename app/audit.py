"""综合判定逻辑：三模型任一命中 → NSFW。"""

from __future__ import annotations

import os
from dataclasses import dataclass

NSFW_THRESHOLD = float(os.environ.get("NSFW_THRESHOLD", "0.5"))
NUDENET_DETECT_THRESHOLD = float(os.environ.get("NUDENET_DETECT_THRESHOLD", "0.3"))

NUDENET_EXPOSED_CLASSES = {
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "FEMALE_BREAST_EXPOSED",
    "BUTTOCKS_EXPOSED",
    "ANUS_EXPOSED",
}


@dataclass
class AuditResult:
    verdict: str  # "NSFW" | "SAFE"
    reason: str
    falconsai_nsfw: float
    marqo_nsfw: float
    nudenet_top_score: float
    nudenet_labels: list[str]


def filter_nudenet(detections: list[dict]) -> tuple[float, list[str]]:
    """从 NudeNet 原始检测中筛出命中的暴露部位。"""
    exposed = [
        (d["class"], d["score"])
        for d in detections
        if d["class"] in NUDENET_EXPOSED_CLASSES and d["score"] >= NUDENET_DETECT_THRESHOLD
    ]
    if not exposed:
        return 0.0, []
    top_score = max(s for _, s in exposed)
    labels = sorted({c for c, _ in exposed})
    return top_score, labels


def combine(
    falc_nsfw: float,
    marqo_nsfw: float,
    nude_top_score: float,
    nude_labels: list[str],
) -> AuditResult:
    reasons: list[str] = []
    if falc_nsfw >= NSFW_THRESHOLD:
        reasons.append(f"Falconsai={falc_nsfw:.2f}")
    if marqo_nsfw >= NSFW_THRESHOLD:
        reasons.append(f"Marqo={marqo_nsfw:.2f}")
    if nude_labels:
        reasons.append(f"NudeNet={'+'.join(nude_labels)}")

    verdict = "NSFW" if reasons else "SAFE"
    reason = " | ".join(reasons) if reasons else "-"
    return AuditResult(
        verdict=verdict,
        reason=reason,
        falconsai_nsfw=falc_nsfw,
        marqo_nsfw=marqo_nsfw,
        nudenet_top_score=nude_top_score,
        nudenet_labels=nude_labels,
    )
