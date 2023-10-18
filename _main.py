import os
import argparse
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import FrameDataset
from model._jepa import JEPA
from model.encoder import ViViT
from torchvision import transforms
import copy
import numpy as np
import torchmetrics
from utils import JEPAParams

def unsupervised_train(model, unlabel_loader, optimizer, criterion, scheduler, args, skip=0):
    model.train()
    for epoch in range(args.epochs):
        for batch_idx, data in enumerate(unlabel_loader):
            data = data.to(args.device)
            optimizer.zero_grad()

            # x: (b, 11, c, h, w), [0-11]
            # y: (b, 11, c, h, w)  [12-22]
            data_x = data[:, :11, :, :, :]
            # mask for frames 12-22 excluding skip frame
            data_y = data[:, 11:, :, :, :]
            mask = torch.zeros_like(data_y)
            mask[:, skip, :, :, :] = 1
            data_y = data_y * mask

            print("data_x: ", data_x.shape)
            print("data_y: ", data_y.shape)
            print("mask: ", mask.shape)

            # # assert that all frames are masked except for the skip frame
            # assert torch.sum(data_y) == torch.sum(data_y[:, skip, :, :, :])
            # # assert index of skip frame is correct
            # assert torch.sum(data_y[:, skip, :, :, :]) == torch.sum(data[:, 11+skip, :, :, :])

            # print("x: ", x.shape)
            # print("y: ", y.shape)

            # rearrange for encoder (b, num_frames=22, c, h, w) to (b, c, num_frames=22, h, w)
            data_x = data_x.permute(0, 2, 1, 3, 4)
            data_y = data_y.permute(0, 2, 1, 3, 4)

            # encode with previous JEPAs

            pred_y, embed_y, losses = model(data_x, data_y)

            loss = criterion(pred_y, embed_y) + losses['loss']
            loss.backward()
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                print('Unsupervised Train Epoch for JEPA #{} : {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    skip + 1, epoch, batch_idx *
                    len(data), len(unlabel_loader.dataset),
                    100. * batch_idx / len(unlabel_loader), loss.item()))
        scheduler.step()

    # save model
    if args.save_model:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        torch.save(model.state_dict(), os.path.join(
            args.save_dir, f'pretrain_JEPA_skips{skip}.pth'))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--save_dir', type=str, default='save')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--log_interval', type=int, default=1)

    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5061, 0.5045, 0.5008],
                             [0.0553, 0.0551, 0.0591])
    ])

    ds = FrameDataset(args.data_dir + "/" + 'unlabeled',
                      labeled=False, transform=transform)
    # ds = torch.utils.data.Subset(ds, list(range(0, 100)))
    unlabel_loader = DataLoader(
        dataset=ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    hparams = JEPAParams()

    encoder_x = ViViT(
        image_size=(160, 240),
        image_patch_size=8,
        frame_patch_size=1,
        num_classes=128,
        frames=11,
        dim=128,
        spatial_depth=6,
        temporal_depth=6,
        heads=8,
        mlp_dim=256,
    )

    encoder_y = copy.deepcopy(encoder_x)

    predictor = nn.Sequential(
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
    )

    model = JEPA(encoder_x, encoder_y, predictor, hparams).to(args.device)

    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)

    unsupervised_train(model, unlabel_loader, optimizer,
                       criterion, scheduler, args, skip=0)
