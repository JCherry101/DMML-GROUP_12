import argparse
import random

import torch
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from src.dataset import CarDataset
from src.model import DualHeadCarNet, upgrade_multi_head_state_dict


def compute_metrics(y_true, y_pred, label_type: str):
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    return {
        f'{label_type}_accuracy': acc,
        f'{label_type}_precision': precision,
        f'{label_type}_recall': recall,
        f'{label_type}_f1': f1,
    }


def resolve_device(explicit: str | None) -> str:
    if explicit:
        return explicit
    if torch.backends.mps.is_available():
        return 'mps'
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', default='resized_DVM')
    parser.add_argument('--csv_path', default='Adv_table.csv')
    parser.add_argument('--checkpoint', default='checkpoint.pt')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', default='', help='cpu | mps | cuda (auto-detect if blank)')
    parser.add_argument('--max_test_samples', type=int, default=0, help='Debug: limit number of test samples (0 = all)')
    args = parser.parse_args()

    args.device = resolve_device(args.device or None)
    print(f'Using device: {args.device}')

    ckpt = torch.load(args.checkpoint, map_location=args.device)
    maker_to_idx = ckpt.get('maker_to_idx')
    body_to_idx = ckpt.get('body_to_idx')
    color_to_idx = ckpt.get('color_to_idx')
    model_to_idx = ckpt.get('model_to_idx')

    test_ds = CarDataset(
        args.root_dir,
        args.csv_path,
        split='test',
        maker_to_idx=maker_to_idx,
        body_to_idx=body_to_idx,
        color_to_idx=color_to_idx,
        model_to_idx=model_to_idx,
    )
    if args.max_test_samples > 0:
        idxs = list(range(len(test_ds)))
        random.shuffle(idxs)
        test_ds = Subset(test_ds, idxs[:args.max_test_samples])
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    base_ds = test_ds.dataset if isinstance(test_ds, Subset) else test_ds
    num_makers = base_ds.num_makers
    num_bodytypes = base_ds.num_bodytypes
    num_colors = base_ds.num_colors if base_ds.num_colors > 0 else None
    num_models = base_ds.num_models if hasattr(base_ds, 'num_models') and base_ds.num_models > 0 else None

    model = DualHeadCarNet(num_makers, num_bodytypes, num_colors, num_models, pretrained=False)
    state_dict = upgrade_multi_head_state_dict(ckpt['model_state'])
    load_result = model.load_state_dict(state_dict, strict=False)
    missing = load_result.missing_keys
    unexpected = load_result.unexpected_keys
    if missing:
        print(f"[WARN] Missing checkpoint keys: {missing}")
    if unexpected:
        print(f"[WARN] Unexpected checkpoint keys: {unexpected}")
    model.to(args.device)
    model.eval()

    maker_true, maker_pred = [], []
    body_true, body_pred = [], []
    color_true, color_pred = [], []
    model_true, model_pred = [], []
    with torch.no_grad():
        for batch in loader:
            imgs, maker_labels, body_labels, color_labels, model_labels = batch
            imgs = imgs.to(args.device)
            maker_logits, body_logits, color_logits, model_logits = model(imgs)
            maker_pred.extend(maker_logits.argmax(dim=1).cpu().tolist())
            maker_true.extend(maker_labels.tolist())
            body_pred.extend(body_logits.argmax(dim=1).cpu().tolist())
            body_true.extend(body_labels.tolist())
            if color_logits is not None and color_labels is not None:
                color_pred.extend(color_logits.argmax(dim=1).cpu().tolist())
                color_true.extend(color_labels.tolist())
            if model_logits is not None and model_labels is not None:
                model_pred.extend(model_logits.argmax(dim=1).cpu().tolist())
                model_true.extend(model_labels.tolist())

    metrics = {}
    metrics.update(compute_metrics(maker_true, maker_pred, 'maker'))
    metrics.update(compute_metrics(body_true, body_pred, 'body'))
    if color_pred and color_true:
        metrics.update(compute_metrics(color_true, color_pred, 'color'))
    if model_pred and model_true:
        metrics.update(compute_metrics(model_true, model_pred, 'model'))

    for k,v in metrics.items():
        print(f'{k}: {v:.4f}')

if __name__ == '__main__':
    main()
