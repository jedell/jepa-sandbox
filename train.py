import torch
import os
import json
import argparse
import numpy as np
from utils.config import ConfigLoader
from utils.tensors import repeat_interleave_batch
from dataset import init_udata
from utils.misc import init_model, init_optimizer_and_scheduler, init_logger, save_checkpoint
from utils.logs import log_loss_metrics, log_grad_metrics, init_metrics

GLOBAL_SEED = 42
torch.manual_seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(GLOBAL_SEED)

parser = argparse.ArgumentParser(description='Train JEPA model')
parser.add_argument('--tag', type=str, default='default_run', help='Name of the training run')
parser.add_argument('--load_checkpoint', type=bool, default=False, help='Whether to load the model from a checkpoint')
parser.add_argument('--checkpoint_file', type=str, default='', help='Path to the checkpoint file')
args = parser.parse_args()

logger = init_logger(args.tag)

config_loader = ConfigLoader('configs/small.yaml')
mask_config = config_loader.get_mask_configs()
data_config = config_loader.get_data_configs()
meta_config = config_loader.get_meta_configs()

_dtype = meta_config['dtype']

if _dtype.lower() == 'bfloat16':
        dtype = torch.bfloat16
        mixed_precision = True
elif _dtype.lower() == 'float16':
    dtype = torch.float16
    mixed_precision = True
else:
    dtype = torch.float32
    mixed_precision = False

# DATA
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

model = init_model({}, data_config, device)

start_epoch = 0
optimizer_config = config_loader.get_optimizer_configs()

ema = optimizer_config['ema']
ipe = optimizer_config['ipe']
ipe_scale = optimizer_config['ipe_scale']
num_epochs = optimizer_config['num_epochs']
clip_grad = optimizer_config['clip_grad']

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

ipe = len(data_loader)

optimizer, scheduler = init_optimizer_and_scheduler(model, optimizer_config, scheduler_config)

scaler = torch.cuda.amp.GradScaler() if mixed_precision else None

momentum_scheduler = (ema[0] + i*(ema[1]-ema[0])/(ipe*num_epochs*ipe_scale)
                          for i in range(int(ipe*num_epochs*ipe_scale)+1))

if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024 ** 3):.2f} GB")
    print(f"Memory Allocated: {torch.cuda.memory_allocated(0) / (1024 ** 3):.2f} GB")
    print(f"Memory Cached: {torch.cuda.memory_reserved(0) / (1024 ** 3):.2f} GB")
else:
    print("CUDA is not available. Training on CPU.")


checkpoint_dir = f"checkpoints/{args.tag}"
os.makedirs(checkpoint_dir, exist_ok=True)

if args.load_checkpoint and args.checkpoint_file:
    checkpoint_path = os.path.join(checkpoint_dir, args.checkpoint_file)
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        for _ in range(start_epoch * ipe):
            scheduler.step()
            next(momentum_scheduler)
        logger.info(f"Loaded checkpoint from {checkpoint_path} at epoch {start_epoch}")
        
loader = iter(data_loader)

class StatsLogger:
    def __init__(self):
        self.stats = {}

    def log(self, key, value):
        if key not in self.stats:
            self.stats[key] = []
        self.stats[key].append(value)

    def get_stats(self, key):
        return self.stats.get(key, [])

    def save_stats(self, filepath):
        with open(filepath, 'w') as f:
            for key, values in self.stats.items():
                f.write(f"{key}: {values}\n")

    def load_stats(self, filepath):
        self.stats = {}
        with open(filepath, 'r') as f:
            for line in f:
                key, values_str = line.strip().split(': ')
                values = eval(values_str)
                self.stats[key] = values

loss_metrics, grad_metrics = init_metrics()
log_freq = 10

logger.info(f"Start {args.tag} training")

current_lr = optimizer_config['start_lr']
warmup = optimizer_config['warmup']

for epoch in range(start_epoch, num_epochs):
    logger.info(f"Epoch {epoch}/{num_epochs}")
    model.train()
    total_loss = 0
    loader = iter(data_loader)
    for batch_idx in range(ipe):
        try:
            batch = next(loader)
            X, masks_enc, masks_pred = batch
            X, masks_enc, masks_pred = load_clips(X, masks_enc, masks_pred, batch_size, num_clips, device)
        except StopIteration:
            logger.info(f"End of data loader reached at iteration {batch_idx}")
            break
        except Exception as e:
            logger.error(f"Error loading batch {batch_idx}: {e}")
            continue            

        loss = 0.
        with torch.cuda.amp.autocast(enabled=mixed_precision):
            _, _, losses = model(X, masks_enc, masks_pred)
            loss = losses['loss']

        _enc_norm, _pred_norm = 0., 0.
        if mixed_precision:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            loss.backward()
            
        if (epoch > warmup) and clip_grad is not None:
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

        if batch_idx % log_freq == 0:
            logger.info(f"Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item()}")
            # grad metrics
            logger.info(f"Grad Norms: Encoder: {_enc_norm}, Predictor: {_pred_norm}")
            log_loss_metrics(loss_metrics, epoch, batch_idx, loss.item(), total_loss / (batch_idx + 1), current_lr)
            log_grad_metrics(grad_metrics, epoch, batch_idx, _enc_norm, _pred_norm)
            

    avg_loss = total_loss / ipe
    current_lr = scheduler.get_last_lr()[0]

    logger.info(f"Epoch: {epoch}, Average Loss: {avg_loss}, Current Learning Rate: {current_lr}")

    save_checkpoint(epoch, model, optimizer, avg_loss, checkpoint_dir)
    logger.info(f"Saved checkpoint to {checkpoint_dir}")

    metrics_path = os.path.join('logs', args.tag, "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump({'loss_metrics': loss_metrics, 'grad_metrics': grad_metrics}, f, indent=4)

logger.info("Training completed.")


