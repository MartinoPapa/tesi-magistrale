import os
import torch
import matplotlib.pyplot as plt

# User preference for smoothing training losses
SMOOTH_LOSSES = False

models = [
    ("Binary/saved_models_GAT/model_trained_final.pt", "../tex/IMG/GAT_graph.png", "Training and Validation Loss - GAGNN-GAT (Binary task)"),
    ("Binary/saved_models_GIN/model_trained_final.pt", "../tex/IMG/GIN_graph.png", "Training and Validation Loss - GAGNN-GIN (Binary task)"),
    ("Multiclass/saved_models_GAT/model_trained_final.pt", "../tex/IMG/GAT_multiclass_graph.png", "Training and Validation Loss - GAGNN-GAT (Multiclass task)"),
    ("Multiclass/saved_models_GIN/model_trained_final.pt", "../tex/IMG/GIN_multiclass_graph.png", "Training and Validation Loss - GAGNN-GIN (Multiclass task)"),
    ("Binary/saved_models_GAT/baseline/baseline_model_final.pt", "../tex/IMG/GAT_baseline_graph.png", "Training and Validation Loss - GAT baseline (Binary task)"),
    ("Binary/saved_models_GIN/baseline/baseline_model_final.pt", "../tex/IMG/GIN_baseline_graph.png", "Training and Validation Loss - GIN baseline (Binary task)"),
    ("Multiclass/saved_models_GAT/baseline/baseline_model_final.pt", "../tex/IMG/GAT_baseline_multiclass_graph.png", "Training and Validation Loss - GAT baseline (Multiclass task)"),
    ("Multiclass/saved_models_GIN/baseline/baseline_model_final.pt", "../tex/IMG/GIN_baseline_multiclass_graph.png", "Training and Validation Loss - GIN baseline (Multiclass task)"),
]

def exponential_moving_average(data, alpha=0.1):
    if not data:
        return data
    ema = [data[0]]
    for point in data[1:]:
        ema.append(alpha * point + (1 - alpha) * ema[-1])
    return ema

def main():
    # First pass: find global min and max for training and validation losses
    global_min = float('inf')
    global_max = float('-inf')

    loss_data = []

    for pt_path, img_path, title in models:
        if not os.path.exists(pt_path):
            print(f"Warning: {pt_path} not found.")
            continue
        
        checkpoint = torch.load(pt_path, map_location='cpu')
        t_loss = checkpoint.get('training_losses', [])
        v_loss = checkpoint.get('val_losses', [])
        
        if SMOOTH_LOSSES and t_loss:
            t_loss = exponential_moving_average(t_loss, alpha=0.1)
        
        all_losses = t_loss + v_loss
        if all_losses:
            global_min = min(global_min, min(all_losses))
            global_max = max(global_max, max(all_losses))
            
        loss_data.append((pt_path, img_path, title, t_loss, v_loss))

    if global_min == float('inf'):
        print("No valid loss data found.")
        return

    # Ensure min is greater than 0 for log scale
    global_min = max(1e-6, global_min)

    y_min_lim = global_min * 0.9
    y_max_lim = global_max * 1.1

    print(f"Unified y-axis scale determined: min={y_min_lim:.4f}, max={y_max_lim:.4f}")

    for pt_path, img_path, title, t_loss, v_loss in loss_data:
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        if t_loss:
            ax1.plot(t_loss, label='Training Loss', color='blue', linewidth=2)
        if v_loss:
            ax1.plot(v_loss, label='Validation / Patience Loss', color='orange', linewidth=2)
            
        ax1.set_ylabel('Loss (log scale)')
        ax1.set_yscale('log')
        ax1.set_ylim([y_min_lim, y_max_lim])
        ax1.set_xlabel('Training Iterations / Epochs')
        ax1.legend()
        
        plt.title(title)
        plt.grid(True, linestyle='--', alpha=0.7)
        fig.tight_layout()
        
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        plt.savefig(img_path, dpi=300)
        plt.close(fig)
        print(f"Saved {img_path}")

    print("All plots generated successfully.")

if __name__ == "__main__":
    main()
