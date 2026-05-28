"""综合判定逻辑：四个模型投票，≥2 票 → NSFW。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

NSFW_THRESHOLD = float(os.environ.get("NSFW_THRESHOLD", "0.5"))
NUDENET_DETECT_THRESHOLD = float(os.environ.get("NUDENET_DETECT_THRESHOLD", "0.5"))
LUKE_THRESHOLD = float(os.environ.get("LUKE_THRESHOLD", "0.5"))

NUDENET_EXPOSED_CLASSES = {
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "FEMALE_BREAST_EXPOSED",
    "BUTTOCKS_EXPOSED",
    "ANUS_EXPOSED",
}

LUKE_NSFW_LABELS = ("hentai", "porn", "sexy")


@dataclass
class AuditResult:
    verdict: str  # "NSFW" | "SAFE"
    reason: str
    falconsai_nsfw: float
    marqo_nsfw: float
    nudenet_top_score: float
    nudenet_labels: list[str]
    luke_nsfw_score: float = 0.0
    luke_scores: dict[str, float] = field(default_factory=dict)


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
    luke_scores: dict[str, float],
) -> AuditResult:
    luke_nsfw = sum(luke_scores.get(label, 0.0) for label in LUKE_NSFW_LABELS)

    votes: list[str] = []
    if falc_nsfw >= NSFW_THRESHOLD:
        votes.append(f"Falconsai={falc_nsfw:.2f}")
    if marqo_nsfw >= NSFW_THRESHOLD:
        votes.append(f"Marqo={marqo_nsfw:.2f}")
    if nude_labels:
        votes.append(f"NudeNet={'+'.join(nude_labels)}")
    if luke_nsfw >= LUKE_THRESHOLD:
        top = max(LUKE_NSFW_LABELS, key=lambda l: luke_scores.get(l, 0.0))
        votes.append(f"Luke={luke_nsfw:.2f}({top}={luke_scores.get(top, 0.0):.2f})")

    verdict = "NSFW" if len(votes) >= 2 else "SAFE"
    if not votes:
        reason = "-"
    elif verdict == "NSFW":
        reason = " | ".join(votes)
    else:
        reason = f"单票忽略: {' | '.join(votes)}"
    return AuditResult(
        verdict=verdict,
        reason=reason,
        falconsai_nsfw=falc_nsfw,
        marqo_nsfw=marqo_nsfw,
        nudenet_top_score=nude_top_score,
        nudenet_labels=nude_labels,
        luke_nsfw_score=luke_nsfw,
        luke_scores=luke_scores,
    )
