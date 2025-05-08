import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_training_progress(metrics_file, save_dir=None):
    """Plot training progress from metrics CSV file.
    
    Args:
        metrics_file: Path to the metrics CSV file
        save_dir: Directory to save the plot (if None, show the plot)
    """
    # Read metrics
    df = pd.read_csv(metrics_file)
    
    # Set style
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 6))
    
    # Plot training and evaluation returns
    plt.subplot(1, 2, 1)
    plt.plot(df['total_frames'], df['train_return'], label='Training Return')
    plt.plot(df['total_frames'], df['eval_return'], label='Evaluation Return')
    plt.xlabel('Total Frames')
    plt.ylabel('Average Return')
    plt.title('Training Progress')
    plt.legend()
    
    # Plot normalized scores
    plt.subplot(1, 2, 2)
    plt.plot(df['total_frames'], df['train_norm_score'], label='Training Score')
    plt.plot(df['total_frames'], df['eval_norm_score'], label='Evaluation Score')
    plt.xlabel('Total Frames')
    plt.ylabel('Normalized Score')
    plt.title('Normalized Performance')
    plt.legend()
    
    # Adjust layout and save/show
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'training_progress.png'))
    else:
        plt.show()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--metrics_file', type=str, required=True,
                       help='Path to the metrics CSV file')
    parser.add_argument('--save_dir', type=str, default=None,
                       help='Directory to save the plot (if None, show the plot)')
    args = parser.parse_args()
    
    plot_training_progress(args.metrics_file, args.save_dir) 