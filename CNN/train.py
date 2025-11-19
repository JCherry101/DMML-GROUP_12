import argparse
import random
from typing import Optional

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm

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


def unpack_batch(batch):
    imgs = batch[0]
    maker_labels = batch[1]
    body_labels = batch[2]
    color_labels = batch[3] if len(batch) > 3 else None
    model_labels = batch[4] if len(batch) > 4 else None
    return imgs, maker_labels, body_labels, color_labels, model_labels


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc='Train', leave=False):
        imgs, maker_labels, body_labels, color_labels, model_labels = unpack_batch(batch)
        imgs = imgs.to(device, non_blocking=True)
        maker_labels = maker_labels.to(device, non_blocking=True)
        body_labels = body_labels.to(device, non_blocking=True)
        color_labels = color_labels.to(device, non_blocking=True) if color_labels is not None else None
        model_labels = model_labels.to(device, non_blocking=True) if model_labels is not None else None
        optimizer.zero_grad()
        maker_logits, body_logits, color_logits, model_logits = model(imgs)
        loss = criterion(maker_logits, maker_labels) + criterion(body_logits, body_labels)
        if color_logits is not None and color_labels is not None:
            loss += criterion(color_logits, color_labels)
        if model_logits is not None and model_labels is not None:
            loss += criterion(model_logits, model_labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
    return total_loss / len(loader.dataset)


def eval_epoch(model, loader, device):
    model.eval()
    maker_true, maker_pred = [], []
    body_true, body_pred = [], []
    color_true, color_pred = [], []
    model_true, model_pred = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc='Eval', leave=False):
            imgs, maker_labels, body_labels, color_labels, model_labels = unpack_batch(batch)
            imgs = imgs.to(device, non_blocking=True)
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
    if color_true:
        metrics.update(compute_metrics(color_true, color_pred, 'color'))
    if model_true:
        metrics.update(compute_metrics(model_true, model_pred, 'model'))
    return metrics


def resolve_device(explicit: Optional[str] = None) -> str:
    if explicit:
        return explicit
    if torch.backends.mps.is_available():  # Apple Silicon GPU
        return 'mps'
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


def subset_if_needed(dataset, max_samples: int):
    if max_samples <= 0 or max_samples >= len(dataset):
        return dataset
    idxs = list(range(len(dataset)))
    random.shuffle(idxs)
    return Subset(dataset, idxs[:max_samples])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', default='resized_DVM')
    parser.add_argument('--csv_path', default='Adv_table.csv')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--output', default='checkpoint.pt')
    parser.add_argument('--device', default='', help='cpu | mps | cuda (auto-detect if blank)')
    parser.add_argument('--no-pretrained', action='store_true', help='Disable pretrained backbone weights')
    parser.add_argument('--max_train_samples', type=int, default=0, help='Debug: limit number of training samples (0 = all)')
    parser.add_argument('--max_test_samples', type=int, default=0, help='Debug: limit number of test samples (0 = all)')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers for faster throughput')
    parser.add_argument('--pin_memory', action='store_true', help='Pin host memory to speed up transfers')
    parser.add_argument('--freeze_backbone', action='store_true', help='Freeze EfficientNet backbone for faster fine-tuning')
    parser.add_argument('--resume_from', default='', help='Path to existing checkpoint to continue fine-tuning')
    args = parser.parse_args()

    args.device = resolve_device(args.device or None)
    print(f'Using device: {args.device}')
    if args.device == 'mps':
        try:
            torch.set_float32_matmul_precision('medium')
        except Exception:
            pass

    resume_state = None
    resume_maker = resume_body = resume_color = resume_model = None
    if args.resume_from:
        print(f'Resuming from {args.resume_from}')
        ckpt = torch.load(args.resume_from, map_location='cpu')
        resume_state = upgrade_multi_head_state_dict(ckpt['model_state'])
        resume_maker = ckpt.get('maker_to_idx')
        resume_body = ckpt.get('body_to_idx')
        resume_color = ckpt.get('color_to_idx')
        resume_model = ckpt.get('model_to_idx')

    train_full = CarDataset(
        args.root_dir,
        args.csv_path,
        split='train',
        maker_to_idx=resume_maker,
        body_to_idx=resume_body,
        color_to_idx=resume_color,
        model_to_idx=resume_model,
    )
    test_full = CarDataset(
        args.root_dir,
        args.csv_path,
        split='test',
        maker_to_idx=train_full.maker_to_idx,
        body_to_idx=train_full.body_to_idx,
        color_to_idx=train_full.color_to_idx,
        model_to_idx=train_full.model_to_idx,
    )

    train_ds = subset_if_needed(train_full, args.max_train_samples)
    test_ds = subset_if_needed(test_full, args.max_test_samples)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(1, args.num_workers // 2) if args.num_workers else 0,
        pin_memory=args.pin_memory,
    )

    base_train = train_ds.dataset if isinstance(train_ds, Subset) else train_ds

    model = DualHeadCarNet(
        base_train.num_makers,
        base_train.num_bodytypes,
        base_train.num_colors if base_train.num_colors > 0 else None,
        base_train.num_models if base_train.num_models > 0 else None,
        pretrained=not args.no_pretrained,
    ).to(args.device)

    if resume_state:
        load_result = model.load_state_dict(resume_state, strict=False)
        if load_result.missing_keys:
            print(f"[WARN] Missing keys when loading checkpoint: {load_result.missing_keys}")
        if load_result.unexpected_keys:
            print(f"[WARN] Unexpected keys when loading checkpoint: {load_result.unexpected_keys}")

    if args.freeze_backbone:
        for param in model.backbone.parameters():
            param.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    if args.epochs <= 0:
        print('Skipping training epochs (epochs <= 0). Performing evaluation only.')

    for epoch in range(1, max(1, args.epochs) + 1):
        if args.epochs > 0:
            loss = train_epoch(model, train_loader, criterion, optimizer, args.device)
            print(f'Epoch {epoch}: loss={loss:.4f}')
        metrics = eval_epoch(model, test_loader, args.device)
        print('Validation ' + ' '.join([f'{k}={v:.4f}' for k, v in metrics.items()]))

    torch.save({
        'model_state': model.state_dict(),
        'maker_to_idx': base_train.maker_to_idx,
        'body_to_idx': base_train.body_to_idx,
        'color_to_idx': base_train.color_to_idx,
        'model_to_idx': base_train.model_to_idx,
    }, args.output)
    print(f'Saved checkpoint to {args.output}')


if __name__ == '__main__':
    main()
