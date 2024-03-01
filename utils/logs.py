def init_metrics():
    loss_metrics = {
        'epoch': [],
        'batch_idx': [],
        'loss': [],
        'avg_loss': [],
        'lr': [],
    }

    grad_metrics = {
        'epoch': [],
        'batch_idx': [],
        'enc_global_norm': [],
        'pred_global_norm': [],
    }

    return loss_metrics, grad_metrics

def log_loss_metrics(metrics, epoch, batch_idx, loss, avg_loss, lr):
    metrics['epoch'].append(epoch)
    metrics['batch_idx'].append(batch_idx)
    metrics['loss'].append(loss)
    metrics['avg_loss'].append(avg_loss)
    metrics['lr'].append(lr)

def log_grad_metrics(metrics, epoch, batch_idx, enc_global_norm, pred_global_norm):
    metrics['epoch'].append(epoch)
    metrics['batch_idx'].append(batch_idx)
    metrics['enc_global_norm'].append(enc_global_norm)
    metrics['pred_global_norm'].append(pred_global_norm)