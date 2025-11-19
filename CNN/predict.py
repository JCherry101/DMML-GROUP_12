"""Single-image inference script for manufacturer, body type, color, and model prediction."""
import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from src.model import DualHeadCarNet, upgrade_multi_head_state_dict


def resolve_device(explicit: str | None) -> str:
    if explicit:
        return explicit
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_checkpoint(checkpoint_path: Path, device: str):
    ckpt = torch.load(checkpoint_path, map_location=device)
    model_state = ckpt["model_state"]
    maker_to_idx = ckpt["maker_to_idx"]
    body_to_idx = ckpt["body_to_idx"]
    color_to_idx = ckpt.get("color_to_idx")
    model_to_idx = ckpt.get("model_to_idx")
    idx_to_maker = {idx: maker for maker, idx in maker_to_idx.items()}
    idx_to_body = {idx: body for body, idx in body_to_idx.items()}
    idx_to_color = {idx: color for color, idx in color_to_idx.items()} if color_to_idx else None
    idx_to_model = {idx: model for model, idx in model_to_idx.items()} if model_to_idx else None
    num_colors = len(color_to_idx) if color_to_idx else None
    num_models = len(model_to_idx) if model_to_idx else None
    model = DualHeadCarNet(len(maker_to_idx), len(body_to_idx), num_colors, num_models, pretrained=False)
    upgraded_state = upgrade_multi_head_state_dict(model_state)
    model.load_state_dict(upgraded_state, strict=False)
    model.to(device)
    model.eval()
    return model, idx_to_maker, idx_to_body, idx_to_color, idx_to_model


def build_transform(image_size: int = 224):
    return transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def predict(image_path: Path, model, transform, device: str, idx_to_maker, idx_to_body, idx_to_color, idx_to_model, topk: int = 3):
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        maker_logits, body_logits, color_logits, model_logits = model(tensor)
        maker_probs = torch.softmax(maker_logits, dim=1)
        body_probs = torch.softmax(body_logits, dim=1)
        color_probs = torch.softmax(color_logits, dim=1) if color_logits is not None else None
        model_probs = torch.softmax(model_logits, dim=1) if model_logits is not None else None
    topk = max(1, topk)
    maker_values, maker_indices = torch.topk(maker_probs[0], k=min(topk, maker_probs.shape[1]))
    body_values, body_indices = torch.topk(body_probs[0], k=min(topk, body_probs.shape[1]))
    maker_preds = [(idx_to_maker[idx.item()], maker_values[i].item()) for i, idx in enumerate(maker_indices)]
    body_preds = [(idx_to_body[idx.item()], body_values[i].item()) for i, idx in enumerate(body_indices)]
    color_preds = []
    if color_probs is not None and idx_to_color is not None:
        color_values, color_indices = torch.topk(color_probs[0], k=min(topk, color_probs.shape[1]))
        color_preds = [(idx_to_color[idx.item()], color_values[i].item()) for i, idx in enumerate(color_indices)]
    model_preds = []
    if model_probs is not None and idx_to_model is not None:
        model_values, model_indices = torch.topk(model_probs[0], k=min(topk, model_probs.shape[1]))
        model_preds = [(idx_to_model[idx.item()], model_values[i].item()) for i, idx in enumerate(model_indices)]
    return maker_preds, body_preds, color_preds, model_preds


def main():
    parser = argparse.ArgumentParser(description="Predict manufacturer and body type for an image")
    parser.add_argument("--image", required=True, help="Path to the input image")
    parser.add_argument("--checkpoint", default="checkpoint.pt", help="Model checkpoint path")
    parser.add_argument("--device", default="", help="cpu | mps | cuda (auto if blank)")
    parser.add_argument("--topk", type=int, default=3, help="How many predictions to show per head")
    parser.add_argument("--image_size", type=int, default=224, help="Resize/Crop size")
    args = parser.parse_args()

    device = resolve_device(args.device or None)
    model, idx_to_maker, idx_to_body, idx_to_color, idx_to_model = load_checkpoint(Path(args.checkpoint), device)
    transform = build_transform(args.image_size)
    maker_preds, body_preds, color_preds, model_preds = predict(
        Path(args.image),
        model,
        transform,
        device,
        idx_to_maker,
        idx_to_body,
        idx_to_color,
        idx_to_model,
        args.topk,
    )

    print("Manufacturer predictions:")
    for name, prob in maker_preds:
        print(f"  {name:25s} prob={prob:.3f}")
    print("Body type predictions:")
    for name, prob in body_preds:
        print(f"  {name:25s} prob={prob:.3f}")
    if color_preds:
        print("Color predictions:")
        for name, prob in color_preds:
            print(f"  {name:25s} prob={prob:.3f}")
    if model_preds:
        print("Model predictions:")
        for name, prob in model_preds:
            print(f"  {name:25s} prob={prob:.3f}")


if __name__ == "__main__":
    main()
