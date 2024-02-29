# load model and run inference
# model: trained VAE
# input: (b, num_frames=11, c, h, w)
# output: (b, num_frames=1, c, h, w).squeeze(1)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
import argparse
from model.jepa import JEPA, JEPA_XEncoder_Predictor
from model.encoder import HierarchicalAttentionEncoder, ViViT
from model.vae import VAE, PICVAE
from dataset import FrameDataset
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--save_dir", type=str, default="saved_models") # location of jepas /pretrain7
parser.add_argument("--data_dir", type=str, default="data")
parser.add_argument("--list_of_jepas", type=str, default="0_1_2_3_4_5_6_7_8_9_10")
parser.add_argument("--vae_path", type=str, default="saved_models/pretrain_vae.pth")
parser.add_argument("--final_masks_path", type=str, default="final_masks.pt")

# python /scratch/jre6163/DL_PROJ_SRC/src/inference.py --list_of_jepas=1_5_10 --vae_path=/scratch/jre6163/DL_SAVED_MODELS/unsupervised/pretrain7/<vae.pth> --save_dir=/scratch/jre6163/DL_SAVED_MODELS/unsupervised/pretrain7 --data_dir=/scratch/jre6163/DL_TEST_DATA
# python /scratch/jre6163/DL_PROJ_SRC/src/inference.py --final_masks_path=final_masks_1.pt --list_of_jepas=1_5_10 --vae_path=/scratch/jre6163/DL_SAVED_MODELS/unsupervised/pretrain7/<vae.pth> --save_dir=/scratch/jre6163/DL_SAVED_MODELS/unsupervised/pretrain7 --data_dir=/scratch/jre6163/DL_TEST_DATA

args = parser.parse_args()

encoder_x = ViViT(
    image_size=(160, 240),
    image_patch_size=(8, 8),
    frames=11,
    frame_patch_size=1,
    num_classes=512,
    dim=512,
    spatial_depth=6,
    temporal_depth=6,
    heads=8,
    mlp_dim=2048,
)

encoder_y = ViViT(
    image_size=(160, 240),
    image_patch_size=(8, 8),
    frames=1,
    frame_patch_size=1,
    num_classes=512,
    dim=512,
    spatial_depth=6,
    temporal_depth=6,
    heads=8,
    mlp_dim=2048,
)

# encoder predictor module (bs, 512) -> (bs, 512)
# 
predictor = nn.Sequential(
    nn.Linear(512, 2048),
    nn.GELU(),
    nn.Dropout(0.1),
    nn.Linear(2048, 512),
    nn.Dropout(0.1)
)

list_of_jepas = args.list_of_jepas.split("_")

JEPAs = [
    {
        "model": None,
    }
    for i in range(len(list_of_jepas))
]

# Load saved JEPAs
for idx, i in enumerate(list_of_jepas):
    hsa_x = HierarchicalAttentionEncoder(
        num_encoders=idx + 1,
        embed_dim=512,
        hidden_dim=512,
    )
    hsa_y = copy.deepcopy(hsa_x)
    m = JEPA(img_size=(160, 240), patch_size=(8, 8), in_channels=3,
            embed_dim=512, 
            encoder_x=copy.deepcopy(encoder_x), 
            encoder_y=copy.deepcopy(encoder_y), 
            hsa_x=hsa_x,
            hsa_y=hsa_y,
            predictor=copy.deepcopy(predictor), 
            skip=int(i)
            ).to(args.device)
    optimizer = optim.SGD(m.parameters(), lr=args.lr, momentum=0.9)
    JEPAs[idx]["model"] = m
    JEPAs[idx]["model"].load_state_dict(torch.load(args.save_dir + f"/pretrain_JEPA_skips{i}.pth"))
    print(f"Loaded JEPA {i}")

encoders_x = []
encoders_y = []
for i in range(len(JEPAs)):
    encoders_x.append(JEPAs[i]["model"].encoder_x)
    encoders_y.append(JEPAs[i]["model"].encoder_y)

jepa_encoder = JEPA_XEncoder_Predictor(
    embed_dim=512,
    encoders_x=encoders_x,
    predictor=JEPAs[-1]["model"].predictor,
    hsa_x=JEPAs[-1]["model"].hsa_x,
)
#conv_layers, z_dimension, pool_kernel_size,
            # conv_kernel_size, input_channels, height, width, hidden_dim, use_cuda
model = VAE(
    conv_layers=3,
    z_dimension=512,
    pool_kernel_size=0,
    conv_kernel_size=4,
    input_channels=3,
    encoder=jepa_encoder,
    height=160,
    width=240,
    hidden_dim=512,
    use_cuda=True,
).to(args.device)

# model = PICVAE(
#     conv_layers=3,
#     z_dimension=512,
#     pool_kernel_size=0,
#     conv_kernel_size=4,
#     input_channels=3,
#     encoder=jepa_encoder,
#     height=160,
#     width=240,
#     hidden_dim=512,
#     use_cuda=True,
# ).to(args.device)

model.load_state_dict(torch.load(args.vae_path))

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5061, 0.5045, 0.5008], [0.0553, 0.0551, 0.0591])
]) # normalize for vae 9

dataset = FrameDataset(root_dir=args.data_dir + "/hidden", labeled=False, inference=True, transform=transform)
# subset of dataset

dl = DataLoader(dataset, batch_size=16, shuffle=False)

# here are 2000 videos with 11 frames each in the hidden dataset.
# You need to submit a saved pytorch tensor or numpy array with the size (2000, 160, 240) 
# that each (160, 240) matrix corresponding to the mask of 22nd frame of each video in the hidden set.

def inference(model, input):
    with torch.no_grad():
        model.eval()
        pred_frame, mu, logvar, z = model(input)
        pred_frame = pred_frame[:, :, :160, :240].squeeze(1)
        return pred_frame

final_masks = []

# test dataloader
for i, (frames, video_dir) in enumerate(dl):
    # frames: torch.Size([100, 11, 3, 160, 240])
    # video_dir: ('/scratch/jre6163/DL_TEST_DATA/hidden/video_15000',)
    # print(f"frames: {frames.size()}")
    # print(f"video_dir: {video_dir}")
    # print("")
    frames = frames.to(args.device)
    
    frames = frames.permute(0, 2, 1, 3, 4)
    
    pred_frame = inference(model, frames)
    # pred_frame: torch.Size([100, 160, 240])

    # print(f"pred_frame: {pred_frame.size()}")

    pred_frame = pred_frame.cpu()

    final_masks.append(pred_frame)
    
# final_masks: (2000, 160, 240)
final_masks = torch.cat(final_masks)
# final_masks = pred_frame
print(f"final_masks: {final_masks.size()}")

# save final_masks
if args.final_masks_path:
    # /scratch/jre6163/final_masks/ + args.final_masks_path
    if not os.path.exists("/scratch/jre6163/final_masks/"):
        os.makedirs("/scratch/jre6163/final_masks/")
    torch.save(final_masks, "/scratch/jre6163/final_masks/" + args.final_masks_path)




