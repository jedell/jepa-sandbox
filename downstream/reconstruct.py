# TODO: Implement the diffusion model decoding step here
# Use pretrained represenations for jepa to train diffusion model
# latent space => diffusion model => pixel space
# use /supervised
# https://github.com/lucidrains/denoising-diffusion-pytorch 
# also test simple VAE https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py

import torch
from jepa.models.jepa import JEPA
from jepa.dataset import init_udata
from utils.config import ConfigLoader
from torchvision.utils import save_image
import os

config_loader = ConfigLoader('configs/small.yaml')
data_config = config_loader.get_data_configs()
mask_config = config_loader.get_mask_configs()

data_loader, _ = init_udata(data_config=data_config, mask_config=mask_config)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = JEPA.load_pretrained('path/to/pretrained/model').to(device)
model.eval()

reconstructed_dir = 'reconstructed_images'
os.makedirs(reconstructed_dir, exist_ok=True)

with torch.no_grad():
    for batch_idx, (frames, masks_enc, masks_pred) in enumerate(data_loader):
        frames = frames[0].to(device) # TODO 
        masks_enc = [m.to(device) for m in masks_enc]
        masks_pred = [m.to(device) for m in masks_pred]

        encoded_frames, _, _ = model(frames, masks_enc, masks_pred)

        # TODO: diffusion model training step
        
