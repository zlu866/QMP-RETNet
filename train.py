import argparse
import logging
import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import average_precision_score, roc_auc_score

from dataloaders import dataset, multi_label
from models.CPAN import CPAN
from utils.losses import AsymmetricLoss, GANLoss, Loss_train
# from helper_functions import mAP, AverageMeter


def get_dataloaders(args):
    add_noise_transform = multi_label.AddNoiseWithSNR(snr_ratio=0.10)
    scale_size, crop_size = 640, 512
    normTransform = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

    data_transform = {
        "train": transforms.Compose([
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
            normTransform
        ]),
        "val": transforms.Compose([
            transforms.Resize((scale_size, scale_size)),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            add_noise_transform,
            normTransform
        ])
    }

    if args.dataset_name == 'MuReD':
        train_dataset = multi_label.MyDataset(args.data, args.label_path, transform=data_transform["train"])
        val_dataset = multi_label.MyDataset(args.val_path, args.val_label, transform=data_transform["val"])
    elif args.dataset_name == 'ODIR-5K':
        train_dataset = multi_label.ODIR_Dateset(args.data, args.label_path, transform=data_transform["train"])
        val_dataset = multi_label.ODIR_Dateset(args.val_path, args.val_label, transform=data_transform["val"])
    elif args.dataset_name == 'DDR':
        train_dataset = dataset.BaseDataset(args.data, args.label_path, transform=data_transform["train"])
        val_dataset = dataset.BaseDataset(args.val_path, args.val_label, transform=data_transform["val"])
    elif args.dataset_name == 'Eye':
        train_dataset = dataset.EyePACS(args.data, args.label_path, transform=data_transform["train"])
        val_dataset = dataset.EyePACS(args.val_path, args.val_label, transform=data_transform["val"])
    else:
        raise ValueError(f"Unknown dataset: {args.dataset_name}")

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, num_workers=0, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=16, num_workers=0, shuffle=True
    )

    return train_loader, val_loader


def evaluate(model, val_loader, device):
    model.eval()
    all_targets, all_outputs, all_probs = [], [], []
    with torch.no_grad():
        for batch in tqdm(val_loader, total=len(val_loader), desc='Validation', unit='batch', leave=False):
            image, target = batch['image'].to(device), batch['label'].to(device)

            output, _, _ = model(image)
            prob = torch.sigmoid(output).cpu()
            pred = prob.data.gt(0.5).long()

            all_targets.append(target.cpu().numpy())
            all_outputs.append(pred.cpu().numpy())
            all_probs.append(prob.cpu().numpy())

    all_targets = np.concatenate(all_targets)
    all_outputs = np.concatenate(all_outputs)
    all_probs = np.concatenate(all_probs)

    acc = (all_outputs == all_targets).mean() * 100.0
    precision = (all_outputs * all_targets).sum(0) / ((all_outputs * all_targets).sum(0) +
                                                     (all_outputs * (1 - all_targets)).sum(0) + 1e-9)
    recall = (all_outputs * all_targets).sum(0) / ((all_outputs * all_targets).sum(0) +
                                                   ((1 - all_outputs) * all_targets).sum(0) + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    mean_precision = precision.mean() * 100.0
    mean_f1 = f1.mean() * 100.0
    meanAP = average_precision_score(all_targets, all_probs, average='macro')

    return acc, mean_f1, mean_precision, meanAP

def train_model(model, device, train_loader, val_loader, args):
    dir_checkpoint = Path(f'./checkpoints/{args.dataset_name}_{args.classes}cls_{args.num_features}')
    criterion = AsymmetricLoss(gamma_neg=1.5, gamma_pos=6, clip=0.01)
    loss_pix = GANLoss('lsgan').to(device)
    loss_drcr = Loss_train().to(device)

    optimizer_gan1 = torch.optim.Adam(model.pix2pix1.parameters(), lr=args.lr)
    optimizer_gan2 = torch.optim.Adam(model.pix2pix2.parameters(), lr=args.lr)
    optimizer_cls = torch.optim.Adam(model.cls.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer_cls, T_max=args.epochs, eta_min=1e-8)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=len(train_loader.dataset), desc=f'Epoch {epoch}/{args.epochs}', unit='img') as pbar:
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

        val_acc, val_f1, val_precision, MAP = evaluate(model, val_loader, device)
        scheduler.step()
        logging.info(f"Epoch {epoch}: Val Acc={val_acc:.2f}, F1={val_f1:.2f}, Precision={val_precision:.2f}, mAP={MAP:.4f}")

        if args.save_checkpoint:
            dir_checkpoint.mkdir(parents=True, exist_ok=True)
            ckpt_path = dir_checkpoint / f'epoch{epoch}_map{MAP:.4f}.pth'
            torch.save(model.state_dict(), ckpt_path)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=24)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gpu_ids', default='0')
    parser.add_argument('--data_root', default='')
    parser.add_argument('--in_channels', type=int, default=3)
    parser.add_argument('--classes', type=int, default=20)
    parser.add_argument('--num_features', type=int, default=3)
    parser.add_argument('--dataset_name', default='MuReD')
    parser.add_argument('--save_checkpoint', action='store_true', default=True)
    return parser.parse_args()


def main():
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device(f'cuda:{args.gpu_ids}' if torch.cuda.is_available() else 'cpu')

    model = CPAN(in_channels=args.in_channels, outputs=args.classes, num_features=args.num_features, device=device)

    train_loader, val_loader = get_dataloaders(args)
    train_model(model, device, train_loader, val_loader, args)


if __name__ == '__main__':
    main()
