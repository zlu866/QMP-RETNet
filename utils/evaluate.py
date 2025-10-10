
import logging
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    cohen_kappa_score,
)

def evaluate_multi_label(model, val_loader, device):
    model.eval()
    all_targets, all_outputs, all_probs = [], [], []

    with torch.no_grad():
        for batch in tqdm(val_loader, total=len(val_loader), desc='Validation', unit='batch', leave=False):
            images, targets = batch['image'].to(device), batch['label'].to(device)
            outputs, _, _ = model(images)
            probs = torch.sigmoid(outputs).cpu()
            preds = (probs > 0.5).long()

            all_targets.append(targets.cpu().numpy())
            all_outputs.append(preds.numpy())
            all_probs.append(probs.numpy())

    all_targets = np.concatenate(all_targets)
    all_outputs = np.concatenate(all_outputs)
    all_probs = np.concatenate(all_probs)

    acc = (all_outputs == all_targets).mean() * 100.0
    precision = (all_outputs * all_targets).sum(0) / (
        (all_outputs * all_targets).sum(0) + (all_outputs * (1 - all_targets)).sum(0) + 1e-9)
    recall = (all_outputs * all_targets).sum(0) / (
        (all_outputs * all_targets).sum(0) + ((1 - all_outputs) * all_targets).sum(0) + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    mean_precision = precision.mean() * 100.0
    mean_f1 = f1.mean() * 100.0
    mean_ap = average_precision_score(all_targets, all_probs, average='macro')

    return acc, mean_f1, mean_precision, mean_ap


def evaluate_dr(model, val_loader, device):
    model.eval()
    all_targets, all_outputs, all_probs = [], [], []

    with torch.no_grad():
        for batch in tqdm(val_loader, total=len(val_loader), desc='Validation', unit='batch', leave=False):
            images, targets = batch['image'].to(device), batch['label'].to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1).cpu()
            preds = torch.argmax(probs, dim=1)

            all_targets.append(targets.cpu().numpy())
            all_outputs.append(preds.numpy())
            all_probs.append(probs.numpy())

    all_targets = np.concatenate(all_targets)
    all_outputs = np.concatenate(all_outputs)
    all_probs = np.concatenate(all_probs)

    acc = accuracy_score(all_targets, all_outputs)
    precision = precision_score(all_targets, all_outputs, average='weighted', zero_division=0)
    recall = recall_score(all_targets, all_outputs, average='weighted', zero_division=0)
    kappa = cohen_kappa_score(all_targets, all_outputs, weights='quadratic')
    auc = roc_auc_score(all_targets, all_probs, multi_class='ovr')

    logging.info(
        f"[Validation] Acc={acc:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, Kappa={kappa:.4f}, AUC={auc:.4f}"
    )

    return acc, precision, kappa, auc
