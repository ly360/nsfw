"""API 响应结构。"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Scores(BaseModel):
    falconsai_nsfw: float = Field(..., description="Falconsai 模型的 NSFW 概率")
    marqo_nsfw: float = Field(..., description="Marqo 模型的 NSFW 概率")
    nudenet_top_score: float = Field(..., description="NudeNet 命中暴露部位的最高置信度")
    nudenet_labels: list[str] = Field(default_factory=list, description="NudeNet 命中的暴露类别")
    luke_nsfw_score: float = Field(..., description="LukeJacob2023 模型的 hentai+porn+sexy 之和")
    luke_scores: dict[str, float] = Field(default_factory=dict, description="LukeJacob2023 模型的 5 类概率明细")


class AuditResponse(BaseModel):
    verdict: str = Field(..., description="NSFW 或 SAFE")
    reason: str = Field(..., description="触发判定的依据；SAFE 时为 '-'")
    scores: Scores
    elapsed_ms: int = Field(..., description="服务端推理总耗时（毫秒）")


class HealthResponse(BaseModel):
    status: str
    device: str
    onnx_providers: list[str]
