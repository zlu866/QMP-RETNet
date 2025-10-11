import argparse
import logging
import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataloaders import dataset, multi_label
from models.CPAN import CPAN
from utils.losses import AsymmetricLoss
from utils.evaluate import evaluate_dr, evaluate_multi_label
from models.DRCR import Loss_train
from models.pretrain_FFA import GANLoss


def get_dataset(args):
    add_noise_transform = multi_label.AddNoiseWithSNR(snr_ratio=0.10)
    scale_size, crop_size = 640, 512
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    data_transform = transforms.Compose([
        transforms.Resize((scale_size, scale_size)),
        transforms.RandomChoice([
            transforms.RandomCrop(640),
            transforms.RandomCrop(576),
            transforms.RandomCrop(512),
            transforms.RandomCrop(448),
            transforms.RandomCrop(384),
            transforms.RandomCrop(320)
        ]),
        transforms.Resize((crop_size, crop_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        add_noise_transform,
        normalize
    ])

    if args.dataset_name == 'MuReD':
        full_dataset = multi_label.MyDataset(args.data_root, mode='train', transform=data_transform)
    elif args.dataset_name == 'DDR':
        full_dataset = dataset.DDRDataset(args.data_root, mode='train', transform=data_transform)
    elif args.dataset_name == 'Eye':
        full_dataset = dataset.EyePACS(args.data_root, mode='train', transform=data_transform)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset_name}")
    return full_dataset


def train_one_fold(model, train_loader, val_loader, device, args, fold):
    save_dir = Path(f'./checkpoints/{args.dataset_name}_fold{fold}')
    save_dir.mkdir(parents=True, exist_ok=True)

    criterion = AsymmetricLoss(gamma_neg=1.5, gamma_pos=6, clip=0.01)
    loss_pix = GANLoss('lsgan').to(device)
    loss_drcr = Loss_train().to(device)

    optimizer_gan1 = torch.optim.Adam(model.pix2pix1.parameters(), lr=args.lr)
    optimizer_gan2 = torch.optim.Adam(model.pix2pix2.parameters(), lr=args.lr)
    optimizer_cls = torch.optim.Adam(model.cls.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer_cls, T_max=args.epochs, eta_min=1e-8)

    best_metric = 0
    best_path = save_dir / "best_model.pth"

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        with tqdm(total=len(train_loader.dataset), desc=f'Fold {fold} | Epoch {epoch}/{args.epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, labels = batch['image'].to(device), batch['label'].to(device)
                masks_pred, fake_A, fake_B = model(images)

                loss_cls = criterion(masks_pred, labels)
                loss_pix1 = 0.1 * loss_pix(fake_A, True) + loss_cls
                loss_pix2 = 0.1 * loss_pix(fake_B, True) + loss_cls

                optimizer_gan1.zero_grad()
                loss_pix1.backward(retain_graph=True)
                optimizer_gan1.step()

                optimizer_gan2.zero_grad()
                loss_pix2.backward(retain_graph=True)
                optimizer_gan2.step()

                optimizer_cls.zero_grad()
                loss_cls.backward()
                optimizer_cls.step()

                epoch_loss += loss_cls.item()
                pbar.update(images.size(0))
                pbar.set_postfix(loss=loss_cls.item())

        model.eval()
        if args.dataset_name == 'MuReD':
            acc, f1, precision, mean_ap = evaluate_multi_label(model, val_loader, device)
            metric = precision
        else:
            acc, precision, kappa, auc = evaluate_dr(model, val_loader, device)
            metric = precision
        scheduler.step()

        if metric > best_metric:
            best_metric = metric
            torch.save(model.state_dict(), best_path)
        logging.info(f"Fold {fold} Epoch {epoch}: Val Metric={metric:.4f}")

    logging.info(f"Fold {fold} Best Val Metric = {best_metric:.4f}")
    return best_metric

def k_fold_training(args):
    device = torch.device(f'cuda:{args.gpu_ids}' if torch.cuda.is_available() else 'cpu')
    full_dataset = get_dataset(args)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(full_dataset)):
        logging.info(f"\n========== Fold {fold+1}/5 ==========")
        train_subset = Subset(full_dataset, train_idx)
        val_subset = Subset(full_dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_subset, batch_size=16, shuffle=False, num_workers=4)

        model = CPAN(
            in_channels=args.in_channels,
            outputs=args.classes,
            num_features=args.num_features,
            device=device
        )
        model = model.to(device)

        fold_metric = train_one_fold(model, train_loader, val_loader, device, args, fold+1)
        fold_metrics.append(fold_metric)


    mean_metric = np.mean(fold_metrics)
    std_metric = np.std(fold_metrics)
    logging.info(f"\n===== 5-Fold Cross Validation Results =====")
    logging.info(f"Metrics: {fold_metrics}")
    logging.info(f"Mean = {mean_metric:.4f}, Std = {std_metric:.4f}")

    return mean_metric, std_metric


def get_args():
    parser = argparse.ArgumentParser(description="Train CPAN with 5-Fold Cross Validation")
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=24)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gpu_ids', default='0')
    parser.add_argument('--data_root', default='./data')
    parser.add_argument('--in_channels', type=int, default=3)
    parser.add_argument('--classes', type=int, default=20)
    parser.add_argument('--num_features', type=int, default=3)
    parser.add_argument('--dataset_name', default='MuReD')
    return parser.parse_args()


def main():
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    k_fold_training(args)


if __name__ == '__main__':
    main()
