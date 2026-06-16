#!/usr/bin/env python
"""Train a LipFD model with only the Region Awareness ResNet branch replaced."""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def build_loader(real_path, fake_path, batch_size, workers, data_label, shuffle):
    from data.datasets import AVLip

    opt = argparse.Namespace(
        data_label=data_label,
        real_list_path=real_path,
        fake_list_path=fake_path,
    )
    dataset = AVLip(opt)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )
    return dataset, loader


@torch.inference_mode()
def evaluate(model, loader, device, max_batches=None, dataset=None, save_scores_path=None):
    import numpy as np
    from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix

    model.eval()
    y_true, y_score, sample_paths = [], [], []
    for batch_idx, (img, crops, label) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        img = img.to(device, non_blocking=True)
        crops = [[t.to(device, non_blocking=True) for t in sublist] for sublist in crops]
        score = model(crops, model.get_features(img))[0].sigmoid().flatten()
        y_score.extend(score.detach().cpu().tolist())
        y_true.extend(label.flatten().tolist())
        if dataset is not None and save_scores_path is not None:
            start = batch_idx * loader.batch_size
            end = start + int(label.numel())
            sample_paths.extend(dataset.total_list[start:end])

    y_true = np.array(y_true)
    y_score = np.array(y_score)
    y_pred = (y_score >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics = {
        "acc": float(accuracy_score(y_true, y_pred)),
        "ap": float(average_precision_score(y_true, y_score)),
        "fpr": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "fnr": float(fn / (fn + tp)) if (fn + tp) else 0.0,
    }
    if save_scores_path is not None:
        save_scores_path = Path(save_scores_path)
        save_scores_path.parent.mkdir(parents=True, exist_ok=True)
        with save_scores_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["path", "label", "score"])
            writer.writerows(zip(sample_paths, y_true.tolist(), y_score.tolist()))
    return metrics


def save_checkpoint(path, model, optimizer, epoch, total_steps, args, metrics=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "total_steps": total_steps,
            "args": vars(args),
            "metrics": metrics or {},
        },
        path,
    )


def append_loss_row(path, row):
    new_file = not Path(path).exists()
    with Path(path).open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "step",
                "epoch",
                "batch",
                "loss",
                "cls_loss",
                "ra_loss",
                "weighted_ra_loss",
                "ra_loss_weight",
                "lr",
            ],
        )
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def plot_loss_curve(csv_path, output_path):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skip plotting loss curve because matplotlib is unavailable: {exc}")
        return

    rows = []
    with Path(csv_path).open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    if not rows:
        return

    steps = [int(r["step"]) for r in rows]
    loss = [float(r["loss"]) for r in rows]
    cls_loss = [float(r["cls_loss"]) for r in rows]
    ra_loss = [float(r["ra_loss"]) for r in rows]
    weighted_ra_loss = [float(r["weighted_ra_loss"]) for r in rows if "weighted_ra_loss" in r and r["weighted_ra_loss"]]

    plt.figure(figsize=(10, 6))
    plt.plot(steps, loss, label="total loss", linewidth=1.8)
    plt.plot(steps, cls_loss, label="classification loss", linewidth=1.2)
    plt.plot(steps, ra_loss, label="region awareness loss", linewidth=1.2)
    if len(weighted_ra_loss) == len(steps):
        plt.plot(steps, weighted_ra_loss, label="weighted RA loss", linewidth=1.2)
    plt.xlabel("Training step")
    plt.ylabel("Loss")
    plt.title("Region-light LipFD Training Loss")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_validation_metrics(history, output_path):
    metric_history = [h for h in history if h.get("metrics")]
    if not metric_history:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skip plotting validation metrics because matplotlib is unavailable: {exc}")
        return

    epochs = [h["epoch"] for h in metric_history]
    plt.figure(figsize=(10, 6))
    for key, label in [
        ("acc", "Accuracy"),
        ("ap", "AP"),
        ("fpr", "FPR"),
        ("fnr", "FNR"),
    ]:
        values = [h["metrics"][key] for h in metric_history]
        plt.plot(epochs, values, marker="o", linewidth=2, label=label)

    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.title("Region-light LipFD Validation Metrics")
    plt.ylim(0, 1.02)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()


def region_awareness_loss(weights_max, weights_org):
    loss = 0.0
    for max_weight, org_weight in zip(weights_max, weights_org):
        loss = loss + (10.0 / torch.exp(max_weight - org_weight)).mean()
    return loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip_name", default="ViT-L/14", choices=["ViT-L/14", "ViT-B/16", "ViT-B/32"])
    parser.add_argument("--backbone", default="resnet18", choices=["resnet18", "resnet34"])
    parser.add_argument("--teacher_ckpt", default="./checkpoints/ckpt.pth")
    parser.add_argument("--real_list_path", default="./datasets/AVLips/0_real")
    parser.add_argument("--fake_list_path", default="./datasets/AVLips/1_fake")
    parser.add_argument("--val_real_list_path", default="./datasets/val/0_real")
    parser.add_argument("--val_fake_list_path", default="./datasets/val/1_fake")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument(
        "--ra_loss_weight",
        type=float,
        default=1.0,
        help="Weight applied to the region awareness loss: total_loss = cls_loss + ra_loss_weight * ra_loss.",
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--loss_freq", type=int, default=50)
    parser.add_argument("--log_loss_every", type=int, default=10)
    parser.add_argument("--val_freq", type=int, default=1)
    parser.add_argument("--max_train_batches", type=int, default=-1)
    parser.add_argument("--max_val_batches", type=int, default=-1)
    parser.add_argument("--output_dir", default="./lightweight/results/checkpoints")
    parser.add_argument("--name", default=None)
    parser.add_argument(
        "--save_val_scores",
        action="store_true",
        help="Save validation path,label,score CSV once per epoch. By default only latest and best are kept.",
    )
    parser.add_argument(
        "--keep_all_val_scores",
        action="store_true",
        help="Keep val_scores_epoch_N.csv for every epoch. Otherwise only latest and best are kept.",
    )
    args = parser.parse_args()

    if args.max_train_batches < 0:
        args.max_train_batches = None
    if args.max_val_batches < 0:
        args.max_val_batches = None
    if args.name is None:
        args.name = f"region_{args.backbone}"

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"RA loss weight: {args.ra_loss_weight}")

    from lightweight.models import LipFDRegionLight
    from lightweight.models.lipfd_region_light import load_global_weights
    model = LipFDRegionLight(clip_name=args.clip_name, backbone=args.backbone)
    print(f"CLIP: {args.clip_name}, global feature dim: {model.global_feature_dim}")
    info = load_global_weights(model, args.teacher_ckpt)
    print(f"Loaded global weights: {info['loaded_keys']} keys")
    if info.get("skipped_shape_keys"):
        print(f"Skipped shape-mismatched global weights: {len(info['skipped_shape_keys'])} keys")
    model.freeze_global_encoder()
    model.to(device)

    train_dataset, train_loader = build_loader(
        args.real_list_path,
        args.fake_list_path,
        args.batch_size,
        args.num_workers,
        "train",
        True,
    )
    val_dataset, val_loader = build_loader(
        args.val_real_list_path,
        args.val_fake_list_path,
        args.batch_size,
        args.num_workers,
        "val",
        False,
    )
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion_cls = nn.BCEWithLogitsLoss()

    total_steps = 0
    best_acc = -1.0
    history = []
    output_dir = Path(args.output_dir) / args.name
    output_dir.mkdir(parents=True, exist_ok=True)
    loss_csv = output_dir / "loss_history.csv"
    loss_png = output_dir / "loss_curve.png"
    val_metrics_png = output_dir / "val_metrics_curve.png"

    for epoch in range(1, args.epochs + 1):
        model.train()
        start = time.time()
        running_loss = 0.0
        epoch_losses = []
        for batch_idx, (img, crops, label) in enumerate(tqdm(train_loader, desc=f"epoch {epoch}")):
            if args.max_train_batches is not None and batch_idx >= args.max_train_batches:
                break
            img = img.to(device, non_blocking=True)
            crops = [[t.to(device, non_blocking=True) for t in sublist] for sublist in crops]
            label = label.to(device, non_blocking=True).float()

            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                features = model.get_features(img)
            output, weights_max, weights_org = model(crops, features)
            logits = output.flatten()
            cls_loss = criterion_cls(logits, label)
            ra_loss = region_awareness_loss(weights_max, weights_org)
            weighted_ra_loss = args.ra_loss_weight * ra_loss
            loss = cls_loss + weighted_ra_loss
            loss.backward()
            optimizer.step()

            total_steps += 1
            loss_value = float(loss.detach().cpu())
            cls_loss_value = float(cls_loss.detach().cpu())
            ra_loss_value = float(ra_loss.detach().cpu())
            weighted_ra_loss_value = float(weighted_ra_loss.detach().cpu())
            running_loss += loss_value
            epoch_losses.append(loss_value)
            if args.log_loss_every and total_steps % args.log_loss_every == 0:
                append_loss_row(
                    loss_csv,
                    {
                        "step": total_steps,
                        "epoch": epoch,
                        "batch": batch_idx + 1,
                        "loss": loss_value,
                        "cls_loss": cls_loss_value,
                        "ra_loss": ra_loss_value,
                        "weighted_ra_loss": weighted_ra_loss_value,
                        "ra_loss_weight": args.ra_loss_weight,
                        "lr": optimizer.param_groups[0]["lr"],
                    },
                )
            if args.loss_freq and total_steps % args.loss_freq == 0:
                print(f"step={total_steps} loss={running_loss / args.loss_freq:.6f}")
                running_loss = 0.0

        metrics = {}
        if args.val_freq and epoch % args.val_freq == 0:
            latest_scores_path = output_dir / "val_scores_latest.csv" if args.save_val_scores else None
            metrics = evaluate(
                model,
                val_loader,
                device,
                args.max_val_batches,
                dataset=val_dataset,
                save_scores_path=latest_scores_path,
            )
            print(
                f"val epoch={epoch} acc={metrics['acc']:.4f} ap={metrics['ap']:.4f} "
                f"fpr={metrics['fpr']:.4f} fnr={metrics['fnr']:.4f}"
            )
            if args.save_val_scores and args.keep_all_val_scores:
                epoch_scores_path = output_dir / f"val_scores_epoch_{epoch}.csv"
                latest_scores_path.replace(epoch_scores_path)
                latest_scores_path = epoch_scores_path
            if metrics["acc"] > best_acc:
                best_acc = metrics["acc"]
                save_checkpoint(output_dir / "best.pth", model, optimizer, epoch, total_steps, args, metrics)
                if args.save_val_scores and latest_scores_path is not None:
                    best_scores_path = output_dir / "val_scores_best.csv"
                    best_scores_path.write_bytes(Path(latest_scores_path).read_bytes())

        save_checkpoint(output_dir / f"epoch_{epoch}.pth", model, optimizer, epoch, total_steps, args, metrics)
        if loss_csv.exists():
            plot_loss_curve(loss_csv, loss_png)
        epoch_info = {
            "epoch": epoch,
            "seconds": time.time() - start,
            "train_loss_mean": sum(epoch_losses) / len(epoch_losses) if epoch_losses else None,
            "metrics": metrics,
        }
        history.append(epoch_info)
        with (output_dir / "history.json").open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
        plot_validation_metrics(history, val_metrics_png)

    print(f"Training done. Outputs: {output_dir}")
    if loss_png.exists():
        print(f"Loss curve saved: {loss_png}")
    if val_metrics_png.exists():
        print(f"Validation metrics curve saved: {val_metrics_png}")


if __name__ == "__main__":
    main()
