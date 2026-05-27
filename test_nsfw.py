"""三个 NSFW 模型对比测试：Falconsai / Marqo / NudeNet。"""

from pathlib import Path

from PIL import Image
import timm
import torch
from timm.data import create_transform, resolve_data_config
from transformers import AutoImageProcessor, AutoModelForImageClassification
from nudenet import NudeDetector

PIC_DIR = Path(__file__).parent / "pic"
NSFW_THRESHOLD = 0.5

FALCONSAI_MODEL = "Falconsai/nsfw_image_detection"
MARQO_MODEL = "hf_hub:Marqo/nsfw-image-detection-384"
# Marqo 模型的标签顺序（来自模型卡）
MARQO_LABELS = ["nsfw", "sfw"]

# NudeNet 中表示"露点/露关键部位"的类别，命中任一即视为 NSFW
NUDENET_EXPOSED_CLASSES = {
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "FEMALE_BREAST_EXPOSED",
    "BUTTOCKS_EXPOSED",
    "ANUS_EXPOSED",
}
NUDENET_DETECT_THRESHOLD = 0.3


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


def load_hf_classifier(model_name: str, device: torch.device):
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(model_name).to(device)
    model.eval()
    return processor, model


def hf_classify(processor, model, image_path: Path, device: torch.device) -> dict[str, float]:
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0].cpu()
    return {model.config.id2label[i].lower(): float(p) for i, p in enumerate(probs)}


def load_marqo(device: torch.device):
    model = timm.create_model(MARQO_MODEL, pretrained=True).eval().to(device)
    cfg = resolve_data_config({}, model=model)
    transform = create_transform(**cfg)
    return model, transform


def marqo_classify(model, transform, image_path: Path, device: torch.device) -> dict[str, float]:
    image = Image.open(image_path).convert("RGB")
    x = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
    probs = torch.softmax(logits, dim=-1)[0].cpu()
    return {MARQO_LABELS[i]: float(probs[i]) for i in range(len(MARQO_LABELS))}


def nudenet_classify(detector: NudeDetector, image_path: Path) -> tuple[float, list[str]]:
    """返回 (最高暴露置信度, 命中的暴露类别列表)。"""
    detections = detector.detect(str(image_path))
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


def verdict(score: float) -> str:
    return "NSFW" if score >= NSFW_THRESHOLD else "SAFE"


def combined_verdict(falc_nsfw: float, marqo_nsfw: float, nude_labels: list[str]) -> tuple[str, str]:
    """综合判定：任一分类器超阈值 或 NudeNet 命中暴露部位 → NSFW。返回 (判定, 触发原因)。"""
    reasons = []
    if falc_nsfw >= NSFW_THRESHOLD:
        reasons.append(f"Falconsai={falc_nsfw:.2f}")
    if marqo_nsfw >= NSFW_THRESHOLD:
        reasons.append(f"Marqo={marqo_nsfw:.2f}")
    if nude_labels:
        reasons.append(f"NudeNet={'+'.join(nude_labels)}")
    if reasons:
        return "NSFW", " | ".join(reasons)
    return "SAFE", "-"


def main():
    images = sorted(
        p for p in PIC_DIR.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    if not images:
        print(f"没有在 {PIC_DIR} 找到图片")
        return

    device = pick_torch_device()
    onnx_providers = pick_onnx_providers()
    print(f"PyTorch 设备：{device}")
    print(f"ONNX Runtime providers：{onnx_providers}")
    print()

    print("加载 Falconsai ...")
    falc_proc, falc_model = load_hf_classifier(FALCONSAI_MODEL, device)
    print("加载 Marqo ...")
    marqo_model, marqo_transform = load_marqo(device)
    print("加载 NudeNet ...")
    nude_detector = NudeDetector(providers=onnx_providers)
    print()

    header = (
        f"{'文件名':<42} | {'Falconsai':>14} | {'Marqo':>14} | {'NudeNet':>14}"
        f" | {'综合':>6} | 触发原因"
    )
    print(header)
    print("-" * len(header) + "-" * 30)

    for img_path in images:
        falc = hf_classify(falc_proc, falc_model, img_path, device)
        marqo = marqo_classify(marqo_model, marqo_transform, img_path, device)
        nude_score, nude_labels = nudenet_classify(nude_detector, img_path)

        falc_nsfw = falc.get("nsfw", 0.0)
        marqo_nsfw = marqo.get("nsfw", 0.0)

        falc_cell = f"{falc_nsfw:.3f} {verdict(falc_nsfw)}"
        marqo_cell = f"{marqo_nsfw:.3f} {verdict(marqo_nsfw)}"
        nude_cell = f"{nude_score:.3f} {verdict(nude_score)}"
        final, reason = combined_verdict(falc_nsfw, marqo_nsfw, nude_labels)

        print(
            f"{img_path.name:<42} | {falc_cell:>14} | {marqo_cell:>14} | {nude_cell:>14}"
            f" | {final:>6} | {reason}"
        )


if __name__ == "__main__":
    main()