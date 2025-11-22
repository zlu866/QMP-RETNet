import argparse
import logging
import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from dataloaders import dataset, multi_label
from models.CPAN import CPAN
from utils.losses import AsymmetricLoss
from utils.evaluate import evaluate_dr,evaluate_multi_label
from models.DRCR import Loss_train
from models.pretrain_FFA import GANLoss
# from utils.visualization import confusion, ROC

def get_dataloaders(args):

    add_noise_transform = multi_label.AddNoiseWithSNR(snr_ratio=0.10)
    scale_size, crop_size = 640, 512
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
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
            normalize
        ]),
        "val": transforms.Compose([
            transforms.Resize((scale_size, scale_size)),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            add_noise_transform,
            normalize
        ])
    }

    if args.dataset_name == 'MuReD':
        train_dataset = multi_label.MyDataset(args.data_root, mode='train', transform=data_transform["train"])
        val_dataset = multi_label.MyDataset(args.data_root, mode='val', transform=data_transform["val"])
    elif args.dataset_name == 'DDR':
        train_dataset = dataset.DDRDataset(args.data_root, mode='train', transform=data_transform["train"])
        val_dataset = dataset.DDRDataset(args.data_root, mode='val', transform=data_transform["val"])
    elif args.dataset_name == 'Eye':
        train_dataset = dataset.EyePACS(args.data_root, mode='train', transform=data_transform["train"])
        val_dataset = dataset.EyePACS(args.data_root, mode='val', transform=data_transform["val"])
    else:
        raise ValueError(f"Unknown dataset: {args.dataset_name}")

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, num_workers=4, shuffle=True, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=16, num_workers=4, shuffle=False, pin_memory=True
    )

    return train_loader, val_loader


def train_model(model, device, train_loader, val_loader, args):
    save_dir = Path(f'./checkpoints/{args.dataset_name}_{args.classes}cls_{args.num_features}')
    save_dir.mkdir(parents=True, exist_ok=True)

    criterion = AsymmetricLoss(gamma_neg=1.5, gamma_pos=6, clip=0.01)
    loss_pix = GANLoss('lsgan').to(device)
    loss_drcr = Loss_train().to(device)

    optimizer_gan1 = torch.optim.Adam(model.pix2pix1.parameters(), lr=args.lr)
    optimizer_gan2 = torch.optim.Adam(model.pix2pix2.parameters(), lr=args.lr)
    optimizer_cls = torch.optim.Adam(model.cls.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer_cls, T_max=args.epochs, eta_min=1e-8)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0

        with tqdm(total=len(train_loader.dataset), desc=f'Epoch {epoch}/{args.epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, labels = batch['image'].to(device), batch['label'].to(device)

                masks_pred, fake_A, fake_B = model(images)
                loss_cls = criterion(masks_pred, labels)
                loss_pix1 = 0.2 * loss_pix(fake_A, True) + loss_cls
                loss_pix2 = 0.2 * loss_pix(fake_B, True) + loss_cls

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

        if args.dataset_name == 'MuReD':
            acc, f1, precision, mean_ap = evaluate_multi_label(model, val_loader, device)
            scheduler.step()
            logging.info(f"Epoch {epoch}: Val Acc={acc:.2f}, F1={f1:.2f}, "
                         f"Precision={precision:.2f}, mAP={mean_ap:.4f}")

            torch.save(model.state_dict(), save_dir / f'epoch{epoch}_map{mean_ap:.4f}.pth')
        elif args.dataset_name == 'DDR':
            acc, precision, kappa, auc = evaluate_dr(model, val_loader, device)
            scheduler.step()
            logging.info(f"Epoch {epoch}: Val Acc={acc:.2f}, F1={precision:.2f}, "
                         f"Precision={kappa:.2f}, mAP={auc:.4f}")

            torch.save(model.state_dict(), save_dir / f'epoch{epoch}_map{kappa:.2f}.pth')


def get_args():
    parser = argparse.ArgumentParser(description="Train CPAN model for retinal disease classification")
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=24)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gpu_ids', default='0')
    parser.add_argument('--data_root', default='./data')
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

    model = CPAN(in_channels=args.in_channels,
                 outputs=args.classes,
                 num_features=args.num_features,
                 device=device)

    train_loader, val_loader = get_dataloaders(args)
    train_model(model, device, train_loader, val_loader, args)


if __name__ == '__main__':
    main()

