import argparse
import logging
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
from utils.evaluate import *

from dataloaders import dataset, multi_label
from models.CPAN import CPAN


def get_dataloader(args):
    import torchvision.transforms as transforms
    from dataloaders import multi_label, dataset

    scale_size, crop_size = 640, 512
    normTransform = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

    test_transform = transforms.Compose([
        transforms.Resize((scale_size, scale_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normTransform
    ])

    if args.dataset_name == 'MuReD':
        test_dataset = multi_label.MyDataset(args.root, mode='test', transform=test_transform)
    elif args.dataset_name == 'DDR':
        test_dataset = dataset.DDRDataset(args.root, mode='test', transform=test_transform)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset_name}")

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, num_workers=0, shuffle=False
    )

    return test_loader


def test(model, test_loader, args, device):
    if args.dataset_name == 'MuReD':
        acc, f1, precision, mean_ap = evaluate_multi_label(model, test_loader, device)
        logging.info(f"Accuracy: {acc:.2f}% | F1: {f1:.2f}% | "
                     f"Precision: {precision:.2f}% | mAP: {mean_ap:.4f}")
    elif args.dataset_name == 'DDR':
        acc, precision, kappa, auc = evaluate_dr(model, test_loader, device)
        logging.info(f"Accuracy: {acc:.2f}% | Precision: {precision:.2f}% | "
                     f"Kappa: {kappa:.2f}% | AUC: {auc:.4f}")


def get_args():
    parser = argparse.ArgumentParser(description="CPAN Testing Script")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pth)')
    parser.add_argument('--root', type=str, required=True,
                        help='Path to test data folder')
    parser.add_argument('--dataset_name', type=str, default='MuReD',
                        help='Dataset name: MuReD / DDR / Eye / ODIR-5K')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--in_channels', type=int, default=3)
    parser.add_argument('--classes', type=int, default=20)
    parser.add_argument('--num_features', type=int, default=3)
    parser.add_argument('--gpu_ids', default='0')
    return parser.parse_args()


def main():
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device(f'cuda:{args.gpu_ids}' if torch.cuda.is_available() else 'cpu')

    model = CPAN(in_channels=args.in_channels,
                 outputs=args.classes,
                 num_features=args.num_features,
                 device=device)

    checkpoint_path = Path(args.checkpoint)
    assert checkpoint_path.exists(), f"Checkpoint not found: {checkpoint_path}"
    logging.info(f"Loading checkpoint from {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)

    test_loader = get_dataloader(args)

    test(model, test_loader, args, device)

    logging.info(f"Test Results on {args.dataset_name}:")


if __name__ == '__main__':
    main()
