import matplotlib.pyplot as plt
import os

def plot_metrics(loss_metrics, grad_metrics, plot_dir):
    plt.figure(figsize=(10, 8))

    # Plotting loss metrics
    plt.subplot(2, 1, 1)
    plt.plot(loss_metrics['epoch'], loss_metrics['loss'], label='Loss')
    plt.plot(loss_metrics['epoch'], loss_metrics['avg_loss'], label='Average Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Metrics Over Epochs')
    plt.legend()

    # Plotting gradient norms
    plt.subplot(2, 1, 2)
    plt.plot(grad_metrics['epoch'], grad_metrics['enc_global_norm'], label='Encoder Gradient Norm')
    plt.plot(grad_metrics['epoch'], grad_metrics['pred_global_norm'], label='Predictor Gradient Norm')
    plt.xlabel('Epoch')
    plt.ylabel('Gradient Norm')
    plt.title('Gradient Norms Over Epochs')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'training_metrics.png'))
    plt.close()

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Plot training metrics from a JSON file.")
    parser.add_argument("--metrics_file", type=str, required=True, help="Path to the metrics JSON file.")
    parser.add_argument("--plot_dir", type=str, required=True, help="Directory to save the plots.")

    args = parser.parse_args()

    with open(args.metrics_file, 'r') as f:
        metrics = json.load(f)

    loss_metrics = metrics['loss_metrics']
    grad_metrics = metrics['grad_metrics']

    plot_metrics(loss_metrics, grad_metrics, args.plot_dir)