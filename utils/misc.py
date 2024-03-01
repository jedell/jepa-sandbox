import torch
import copy
import os
import datetime
import logging
from typing import Tuple
from utils.tensors import trunc_normal_

from models.jepa import JEPA
from utils.config import JEPAParams
from models.vit import vit_small
from models.predictor import vit_predictor
from models.utils.multimask import MultiMaskWrapper, PredictorMultiMaskWrapper
from utils.schedulers import WarmupCosineSchedule

def init_model(
    model_config: dict,
    data_config: dict,
    device: torch.device
) -> JEPA:
    
    encoder = vit_small(
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

def init_logger(dir_tag: str):
    start_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_dir = os.path.join('logs', dir_tag)
    os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    file_handler = logging.FileHandler(os.path.join(log_dir, f'{start_time}.log'))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    return logger

def save_checkpoint(epoch, model, optimizer, loss, checkpoint_dir):
    checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch}.pth")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, checkpoint_path)


