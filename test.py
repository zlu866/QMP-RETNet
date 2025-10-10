import argparse
import logging
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

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
        test_dataset = multi_label.MyDataset(args.data, args.label_path, transform=test_transform)
    elif args.dataset_name == 'ODIR-5K':
        test_dataset = multi_label.ODIR_Dateset(args.data, args.label_path, transform=test_transform)
    elif args.dataset_name == 'DDR':
        test_dataset = dataset.BaseDataset(args.data, args.label_path, transform=test_transform)
    elif args.dataset_name == 'Eye':
        test_dataset = dataset.EyePACS(args.data, args.label_path, transform=test_transform)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset_name}")

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, num_workers=0, shuffle=False
    )

    return test_loader


def evaluate(model, data_loader, device):
    model.eval()
    all_targets, all_outputs, all_probs = [], [], []

    with torch.no_grad():
        for batch in tqdm(data_loader, total=len(data_loader), desc='Testing', unit='batch', leave=False):
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
    precision = (all_outputs * all_targets).sum(0) / (
            (all_outputs * all_targets).sum(0) + (all_outputs * (1 - all_targets)).sum(0) + 1e-9)
    recall = (all_outputs * all_targets).sum(0) / (
            (all_outputs * all_targets).sum(0) + ((1 - all_outputs) * all_targets).sum(0) + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    mean_precision = precision.mean() * 100.0
    mean_f1 = f1.mean() * 100.0
    meanAP = average_precision_score(all_targets, all_probs, average='macro')

    return acc, mean_f1, mean_precision, meanAP


def get_args():
    parser = argparse.ArgumentParser(description="CPAN Testing Script")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pth)')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to test data folder')
    parser.add_argument('--label_path', type=str, required=True,
                        help='Path to test label file')
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

    acc, mean_f1, mean_precision, meanAP = evaluate(model, test_loader, device)

    logging.info(f"Test Results on {args.dataset_name}:")
    logging.info(f"Accuracy: {acc:.2f}% | F1: {mean_f1:.2f}% | "
                 f"Precision: {mean_precision:.2f}% | mAP: {meanAP:.4f}")


if __name__ == '__main__':
    main()
