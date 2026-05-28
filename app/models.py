"""三个 NSFW 模型的加载和推理封装。

从 test_nsfw.py 抽取，调整为接受内存中的图片（PIL.Image / bytes）而非文件路径，
方便在 API 请求中直接处理 UploadFile。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from PIL import Image
from nudenet import NudeDetector
from timm.data import create_transform, resolve_data_config
from transformers import AutoImageProcessor, AutoModelForImageClassification, ViTImageProcessor
import timm

FALCONSAI_MODEL = "Falconsai/nsfw_image_detection"
MARQO_MODEL = "hf_hub:Marqo/nsfw-image-detection-384"
MARQO_LABELS = ["nsfw", "sfw"]
LUKE_MODEL = "LukeJacob2023/nsfw-image-detector"


def pick_torch_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pick_onnx_providers() -> list[str]:
    import onnxruntime as ort
    available = ort.get_available_providers()
    preferred = ["CUDAExecutionProvider", "CoreMLExecutionProvider", "CPUExecutionProvider"]
    return [p for p in preferred if p in available]


@dataclass
class ModelBundle:
    """常驻内存的模型集合。"""

    device: torch.device
    onnx_providers: list[str]
    falc_processor: object
    falc_model: object
    marqo_model: object
    marqo_transform: object
    luke_processor: object
    luke_model: object
    nude_detector: NudeDetector

    @classmethod
    def load(cls) -> "ModelBundle":
        device = pick_torch_device()
        onnx_providers = pick_onnx_providers()

        falc_processor = AutoImageProcessor.from_pretrained(FALCONSAI_MODEL)
        falc_model = AutoModelForImageClassification.from_pretrained(FALCONSAI_MODEL).to(device)
        falc_model.eval()

        marqo_model = timm.create_model(MARQO_MODEL, pretrained=True).eval().to(device)
        cfg = resolve_data_config({}, model=marqo_model)
        marqo_transform = create_transform(**cfg)

        luke_processor = ViTImageProcessor.from_pretrained(LUKE_MODEL)
        luke_model = AutoModelForImageClassification.from_pretrained(LUKE_MODEL).to(device)
        luke_model.eval()

        nude_detector = NudeDetector(providers=onnx_providers)

        return cls(
            device=device,
            onnx_providers=onnx_providers,
            falc_processor=falc_processor,
            falc_model=falc_model,
            marqo_model=marqo_model,
            marqo_transform=marqo_transform,
            luke_processor=luke_processor,
            luke_model=luke_model,
            nude_detector=nude_detector,
        )

    def falconsai_classify(self, image: Image.Image) -> dict[str, float]:
        inputs = self.falc_processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.falc_model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0].cpu()
        return {self.falc_model.config.id2label[i].lower(): float(p) for i, p in enumerate(probs)}

    def marqo_classify(self, image: Image.Image) -> dict[str, float]:
        x = self.marqo_transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.marqo_model(x)
        probs = torch.softmax(logits, dim=-1)[0].cpu()
        return {MARQO_LABELS[i]: float(probs[i]) for i in range(len(MARQO_LABELS))}

    def luke_classify(self, image: Image.Image) -> dict[str, float]:
        inputs = self.luke_processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.luke_model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0].cpu()
        return {self.luke_model.config.id2label[i].lower(): float(p) for i, p in enumerate(probs)}

    def nudenet_detect(self, image_bytes: bytes) -> list[dict]:
        return self.nude_detector.detect(image_bytes)
