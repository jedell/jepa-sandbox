import os
import numpy as np
from PIL import Image
import torch 
from torch.utils.data import Dataset
from masks.multiblock3d import MaskCollator as MB3DMaskCollator
from utils.tensors import repeat_interleave_batch
from torchvision import transforms
import matplotlib.pyplot as plt
import argparse
from utils.config import ConfigLoader

DEBUG = False

class FrameDataset(Dataset):
    def __init__(self, root_dir, labeled=True, inference=False, transform=None):
        """
        Args:
            root_dir (string): Dataset folder path
            labeled (bool, optional): If the input data is labeled. Defaults to True.
            transform (torchvision.transforms, optional): Data transforms. Defaults to None.
        """
        self.root_dir = root_dir
        self.labeled = labeled
        self.transform = transform
        self.inference = inference
        self.video_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        self.video_dirs.sort(key=lambda x: int(x.split('/')[-1].split('_')[-1]))
        self.frames_per_clip = 11 # Adjusted to return 11 clips
        self.num_clips = 2 # Adjusted to split 22 frames into 2 sets of 11 clips each
            
    def __len__(self):
        return len(self.video_dirs) * self.num_clips # Adjusted to account for each half of the clips
    
    def __getitem__(self, idx):
        video_idx = idx // self.num_clips # Adjust video index based on num_clips
        clip_half = idx % self.num_clips # Determine which half of the clips to return
        video_dir = os.path.join(self.root_dir, self.video_dirs[video_idx])
        frame_files = [f for f in os.listdir(video_dir) if f.endswith('.png')]
        frame_files.sort(key=lambda x: int(x.split('/')[-1].split('.')[0].split('_')[-1]))

        # Determine start and end indices for the clips based on clip_half
        start_idx = clip_half * self.frames_per_clip
        end_idx = start_idx + self.frames_per_clip
        frame_files = frame_files[start_idx:end_idx] # Select only the relevant half of the frames

        if DEBUG:
            metadata = {
                "frames": frame_files,
                "video": video_dir
            }
            print("{}".format(metadata))
        
        frames = []
        for frame_file in frame_files:
            frame_path = os.path.join(video_dir, frame_file)
            frame = Image.open(frame_path).convert('RGB')
            if self.transform:
                frame = self.transform(frame)
            frames.append(frame)
            
        frames = torch.stack(frames) # [T, C, H, W]
        frames = frames.permute(0, 2, 3, 1)  # [T, H, W, C]

        frames = [frames]
        
        if self.labeled:
            mask_path = os.path.join(video_dir, 'mask.npy')
            mask = np.load(mask_path)
            mask = torch.from_numpy(mask).long()
            return frames, mask
        elif self.inference:
            return frames, video_dir
        else:
            return frames
        
    def test_plot_frame(self, idx):
        video_idx = idx // self.num_clips
        clip_half = idx % self.num_clips
        video_dir = os.path.join(self.root_dir, self.video_dirs[video_idx])
        frame_files = [f for f in os.listdir(video_dir) if f.endswith('.png')]
        frame_files.sort(key=lambda x: int(x.split('/')[-1].split('.')[0].split('_')[-1]))

        start_idx = clip_half * self.frames_per_clip
        end_idx = start_idx + self.frames_per_clip
        frame_files = frame_files[start_idx:end_idx]

        frames = []
        for frame_file in frame_files:
            frame_path = os.path.join(video_dir, frame_file)
            frame = Image.open(frame_path).convert('RGB')
            if self.transform:
                frame = self.transform(frame)
            frames.append(frame)

        frames = torch.stack(frames)  # [T, C, H, W]
        frames = frames.permute(0, 2, 3, 1)  # [T, H, W, C]

        clips = [frames]

        fig, axs = plt.subplots(1, len(clips[0]), figsize=(20, 5))
        for i, frame in enumerate(clips[0]):
            axs[i].imshow(frame.cpu().numpy())
            axs[i].axis('off')
        plt.tight_layout()
        plt.savefig(f"clips_{idx}.png")
        plt.close()

def init_udata(
        data_config,
        mask_config
    ):

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset = FrameDataset(root_dir=data_config['unlabel_dir'], labeled=False, transform=transform)

    sampler = torch.utils.data.SequentialSampler(dataset)

    mask_collator = MB3DMaskCollator(
        crop_size=data_config['collate']['crop_size'],
        num_frames=data_config['collate']['num_frames'],
        patch_size=data_config['patch_size'],
        tubelet_size=data_config['collate']['tubelet_size'],
        cfgs_mask=mask_config)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=mask_collator,
        sampler=sampler,
        batch_size=data_config['batch_size'],
        drop_last=data_config['drop_last'],
        pin_memory=data_config['pin_mem'],
        num_workers=data_config['num_workers'],
        persistent_workers=True)
    

    return data_loader, dataset

import matplotlib.pyplot as plt
import numpy as np

def save_image_with_title(image, title, file_path):
    """
    Saves an image with a title at a normal size.
    
    Args:
    - image (torch.Tensor): The image to save.
    - title (str): The title of the image.
    - file_path (str): The path to save the image file.
    """
    # Convert tensor image to numpy if not already
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    # Create figure and axes
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.set_title(title)
    ax.axis('off')
    
    # Save the figure
    fig.savefig(file_path)
    plt.close(fig)  # Close the figure to free memory

if __name__ == "__main__":
    DEBUG = False
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, default='./data/Dataset_Student/train')
    parser.add_argument('--val_dir', type=str, default='./data/Dataset_Student/val')
    parser.add_argument('--unlabel_dir', type=str, default='./data/Dataset_Student/unlabeled')

    args = parser.parse_args()

    # test dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset = FrameDataset(root_dir=args.train_dir, labeled=True, transform=transform)
    print('dataset size: {}'.format(len(dataset)))

    video_frames, video_mask = dataset[0]
    print('video_frames size: {}'.format(video_frames[0].shape))
    print('video_mask size: {}'.format(video_mask.shape))

    print (video_frames[0].shape)
    print(video_mask[0].shape)

    config_loader = ConfigLoader('configs/small.yaml')
    mask_config = config_loader.get_mask_configs()
    data_config = config_loader.get_data_configs()

    data_loader, dataset = init_udata(data_config=data_config, mask_config=mask_config)

    data, masks_enc, masks_pred = next(iter(data_loader))

    print("Dataloader:")
    print(data[0].shape)
    print(masks_enc[0].shape)
    print(masks_pred[0].shape)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_size = data_config['batch_size']
    num_clips = data_config['num_clips']

    def load_clips():
        # -- unsupervised video clips
        # Put each clip on the GPU and concatenate along batch
        # dimension
        clips = torch.cat([u.to(device, non_blocking=True) for u in data[0]], dim=0)

        # Put each mask-enc/mask-pred pair on the GPU and reuse the
        # same mask pair for each clip
        _masks_enc, _masks_pred = [], []
        for _me, _mp in zip(masks_enc, masks_pred):
            _me = _me.to(device, non_blocking=True)
            _mp = _mp.to(device, non_blocking=True)
            _me = repeat_interleave_batch(_me, batch_size, repeat=num_clips)
            _mp = repeat_interleave_batch(_mp, batch_size, repeat=num_clips)
            _masks_enc.append(_me)
            _masks_pred.append(_mp)

        return (clips, _masks_enc, _masks_pred)
    clips, masks_enc, masks_pred = load_clips()

    print("Load Clips:")
    print(clips.shape)
    print(masks_enc[0].shape)
    print(masks_pred[0].shape)

    dataset.test_plot_frame(0)
    dataset.test_plot_frame(1)


