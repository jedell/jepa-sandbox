import torch
import copy
import os
import logging
import argparse
from typing import Tuple
from utils.config import ConfigLoader
from utils.tensors import repeat_interleave_batch, trunc_normal_
from dataset import init_udata

from models.jepa import JEPA
from models.vit import vit_tiny
from models.predictor import vit_predictor
from models.utils.multimask import MultiMaskWrapper, PredictorMultiMaskWrapper
from utils.config import JEPAParams
from utils.schedulers import WarmupCosineSchedule

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train JEPA model')
parser.add_argument('--run_name', type=str, default='default_run', help='Name of the training run')
parser.add_argument('--load_checkpoint', type=bool, default=False, help='Whether to load the model from a checkpoint')
parser.add_argument('--checkpoint_file', type=str, default='', help='Path to the checkpoint file')
args = parser.parse_args()

config_loader = ConfigLoader('configs/small.yaml')
mask_config = config_loader.get_mask_configs()
data_config = config_loader.get_data_configs()

data_loader, dataset = init_udata(data_config=data_config, mask_config=mask_config)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = data_config['batch_size']
num_clips = data_config['num_clips']

def load_clips(udata, masks_enc, masks_pred, batch_size=16, num_clips=1, device='cuda'):
    # clips = torch.cat([u.to(device, non_blocking=True) for u in udata[0]], dim=0)
    clips = udata[0].to(device, non_blocking=True)
    _masks_enc, _masks_pred = [], []
    for _me, _mp in zip(masks_enc, masks_pred):
        _me = _me.to(device, non_blocking=True)
        _mp = _mp.to(device, non_blocking=True)
        _me = repeat_interleave_batch(_me, batch_size, repeat=num_clips)
        _mp = repeat_interleave_batch(_mp, batch_size, repeat=num_clips)
        _masks_enc.append(_me)
        _masks_pred.append(_mp)

    return (clips, _masks_enc, _masks_pred)

def init_model(
    model_config: dict,
    device: torch.device
) -> JEPA:
    
    encoder = vit_tiny(
        image_size=data_config['collate']['crop_size'],
        num_frames=data_config['collate']['num_frames'],
        tubelet_size=data_config['collate']['tubelet_size'],
        use_sdpa=False,
        uniform_power=False,
    )

    encoder = MultiMaskWrapper(encoder)

    predictor = vit_predictor(
        image_size=data_config['collate']['crop_size'],
        num_frames=data_config['collate']['num_frames'],
        tubelet_size=data_config['collate']['tubelet_size'],
        patch_size=data_config['patch_size'],
        use_mask_tokens=False,
        embed_dim=encoder.backbone.embed_dim,
        predictor_embed_dim=384,
        depth=6,
        num_heads=encoder.backbone.num_heads,
        num_mask_tokens=2,
        zero_init_mask_tokens=True,
        use_sdpa=False,
        uniform_power=False,
    )

    predictor = PredictorMultiMaskWrapper(predictor)

    def _init_weights(m: torch.nn.Module) -> None:
        if isinstance(m, torch.nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)

    for m in encoder.modules():
        _init_weights(m)

    for m in predictor.modules():
        _init_weights(m)

    hparams = JEPAParams()

    target_encoder = copy.deepcopy(encoder)

    model = JEPA(encoder, target_encoder, predictor, hparams, training=True)

    model.to(device)

    return model

def init_optimizer_and_scheduler(
    model: JEPA,
    optimizer_config: dict,
    scheduler_config: dict
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    
    for p in model.target_encoder.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=optimizer_config['lr'],
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=scheduler_config['weight_decay']
    )

    warmup = scheduler_config['warmup']
    iterations_per_epoch = scheduler_config['ipe']
    start_lr = scheduler_config['start_lr']
    ref_lr = scheduler_config['lr']
    final_lr = scheduler_config['final_lr']
    ipe_scale = scheduler_config['ipe_scale']
    num_epochs = scheduler_config['epochs']
    
    scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=int(warmup * iterations_per_epoch),
        start_lr=start_lr,
        ref_lr=ref_lr,
        final_lr=final_lr,
        T_max=int(ipe_scale * num_epochs * iterations_per_epoch),
    )

    return optimizer, scheduler

model = init_model({}, device)

start_epoch = 0
optimizer_config = config_loader.get_optimizer_configs()

ema = optimizer_config['ema']
ipe = optimizer_config['ipe']
ipe_scale = optimizer_config['ipe_scale']
num_epochs = optimizer_config['num_epochs']
clip_grad = optimizer_config['clip_grad']

momentum_scheduler = (ema[0] + i*(ema[1]-ema[0])/(ipe*num_epochs*ipe_scale)
                          for i in range(int(ipe*num_epochs*ipe_scale)+1))

scheduler_config = {
    'ipe': optimizer_config['ipe'],
    'ipe_scale': optimizer_config['ipe_scale'],
    'clip_grad': optimizer_config['clip_grad'],
    'weight_decay': optimizer_config['weight_decay'],
    'final_weight_decay': optimizer_config['final_weight_decay'],
    'epochs': optimizer_config['num_epochs'],
    'warmup': optimizer_config['warmup'],
    'start_lr': optimizer_config['start_lr'],
    'lr': optimizer_config['lr'],
    'final_lr': optimizer_config['final_lr'],
}

optimizer, scheduler = init_optimizer_and_scheduler(model, optimizer_config, scheduler_config)

scaler = torch.cuda.amp.GradScaler()

if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024 ** 3):.2f} GB")
    print(f"Memory Allocated: {torch.cuda.memory_allocated(0) / (1024 ** 3):.2f} GB")
    print(f"Memory Cached: {torch.cuda.memory_reserved(0) / (1024 ** 3):.2f} GB")
else:
    print("CUDA is not available. Training on CPU.")


checkpoint_dir = f"checkpoints/{args.run_name}"
os.makedirs(checkpoint_dir, exist_ok=True)

# Load checkpoint by file name from command line args if exists and if load_checkpoint is True
if args.load_checkpoint and args.checkpoint_file:
    checkpoint_path = os.path.join(checkpoint_dir, args.checkpoint_file)
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        # Manually set the scheduler's step to the correct epoch
        for _ in range(start_epoch * ipe):
            scheduler.step()
            next(momentum_scheduler)
        logger.info(f"Loaded checkpoint from {checkpoint_path} at epoch {start_epoch}")

mixed_precision = False

for epoch in range(start_epoch, num_epochs):
    model.train()
    total_loss = 0
    for batch_idx, (X, masks_enc, masks_pred) in enumerate(data_loader):
        X, masks_enc, masks_pred = load_clips(X, masks_enc, masks_pred, batch_size, num_clips, device)            


        with torch.cuda.amp.autocast(enabled=mixed_precision):
            _, _, losses = model(X, masks_enc, masks_pred)
            loss = losses['loss']

        if mixed_precision:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
        else:
            loss.backward()
            
        if clip_grad:
            _enc_norm = torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), clip_grad)
            _pred_norm = torch.nn.utils.clip_grad_norm_(model.predictor.parameters(), clip_grad)

        if mixed_precision:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        optimizer.zero_grad()

        m = next(momentum_scheduler)
        with torch.no_grad():
            for param_q, param_k in zip(model.encoder.parameters(), model.target_encoder.parameters()):
                param_k.data.mul_(m).add_((1.-m) * param_q.detach().data)

        total_loss += loss.item()

        if batch_idx % 10 == 0:
            logger.info(f"Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item()}")

    scheduler.step()

    avg_loss = total_loss / len(data_loader)
    current_lr = scheduler.get_last_lr()[0]

    logger.info(f"Epoch: {epoch}, Average Loss: {avg_loss}, Current Learning Rate: {current_lr}")

    # Save checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch}.pth")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),  # Save scheduler state
        'loss': avg_loss,
    }, checkpoint_path)
    logger.info(f"Saved checkpoint to {checkpoint_path}")

logger.info("Training completed.")

