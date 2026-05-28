"""FastAPI 入口：图片内容审计 API。"""

from __future__ import annotations

import io
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from .audit import combine, filter_nudenet
from .models import ModelBundle
from .schemas import AuditResponse, HealthResponse, Scores

logger = logging.getLogger("nsfw-audit")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("加载模型 ...")
    t0 = time.perf_counter()
    bundle = ModelBundle.load()
    logger.info(
        "模型加载完成，设备=%s，ONNX providers=%s，耗时 %.1fs",
        bundle.device, bundle.onnx_providers, time.perf_counter() - t0,
    )
    app.state.bundle = bundle
    yield


app = FastAPI(title="NSFW Image Audit API", version="0.1.0", lifespan=lifespan)

TEST_HTML_PATH = Path(__file__).resolve().parent.parent / "test.html"


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(TEST_HTML_PATH, media_type="text/html")


@app.get("/healthz", response_model=HealthResponse)
def healthz(request: Request) -> HealthResponse:
    bundle: ModelBundle = request.app.state.bundle
    return HealthResponse(
        status="ok",
        device=str(bundle.device),
        onnx_providers=bundle.onnx_providers,
    )


@app.post("/audit/image", response_model=AuditResponse)
async def audit_image(request: Request, file: UploadFile = File(...)) -> AuditResponse:
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"不支持的 content-type: {file.content_type}，仅支持 {sorted(ALLOWED_CONTENT_TYPES)}",
        )

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="文件为空")
    if len(image_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"文件超出大小限制（{MAX_FILE_SIZE} bytes）",
        )

    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.load()
        image = image.convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, ValueError) as exc:
        logger.warning("图片解析失败 filename=%s err=%s", file.filename, exc)
        raise HTTPException(status_code=400, detail=f"图片损坏或格式无法识别: {exc}")

    bundle: ModelBundle = request.app.state.bundle
    t0 = time.perf_counter()

    try:
        falc = bundle.falconsai_classify(image)
        marqo = bundle.marqo_classify(image)
        nude_detections = bundle.nudenet_detect(image_bytes)
    except Exception as exc:
        logger.warning("模型推理失败 filename=%s err=%s", file.filename, exc)
        raise HTTPException(status_code=400, detail=f"图片无法被模型解析: {exc}")
    nude_top, nude_labels = filter_nudenet(nude_detections)

    result = combine(
        falc_nsfw=falc.get("nsfw", 0.0),
        marqo_nsfw=marqo.get("nsfw", 0.0),
        nude_top_score=nude_top,
        nude_labels=nude_labels,
    )

    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    logger.info(
        "audit filename=%s verdict=%s falc=%.3f marqo=%.3f nude=%.3f labels=%s elapsed=%dms",
        file.filename, result.verdict, result.falconsai_nsfw, result.marqo_nsfw,
        result.nudenet_top_score, result.nudenet_labels, elapsed_ms,
    )

    return AuditResponse(
        verdict=result.verdict,
        reason=result.reason,
        scores=Scores(
            falconsai_nsfw=result.falconsai_nsfw,
            marqo_nsfw=result.marqo_nsfw,
            nudenet_top_score=result.nudenet_top_score,
            nudenet_labels=result.nudenet_labels,
        ),
        elapsed_ms=elapsed_ms,
    )
