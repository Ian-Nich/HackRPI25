"""
Visualize model results with retro-style charts
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.patches as mpatches


def load_results(results_path='retrobrain/results/training_results.json'):
    """Load training/evaluation results"""
    try:
        with open(results_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        # Try evaluation results
        try:
            with open('retrobrain/results/evaluation_results.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print("⚠️  No results found. Please train models first.")
            return None


def plot_accuracy_comparison(results, save_path='retrobrain/results/accuracy_comparison.png'):
    """Plot accuracy comparison across eras"""
    if results is None:
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Retro color scheme
    colors_1980s = ['#00ff00', '#ff00ff', '#00ffff', '#ffff00']
    colors_2000s = ['#0080ff', '#ff0080', '#00ff80', '#ff8000']
    colors_2020s = ['#ff0080']
    
    x_pos = 0
    all_labels = []
    all_accuracies = []
    era_colors = []
    
    # 1980s models
    if '1980s' in results.get('eras', {}):
        for i, (model_name, model_result) in enumerate(results['eras']['1980s'].items()):
            accuracy = model_result.get('accuracy', 0)
            all_labels.append(f"1980s\n{model_name}")
            all_accuracies.append(accuracy)
            era_colors.append(colors_1980s[i % len(colors_1980s)])
    
    # 2000s models
    if '2000s' in results.get('eras', {}):
        for i, (model_name, model_result) in enumerate(results['eras']['2000s'].items()):
            accuracy = model_result.get('accuracy', 0)
            all_labels.append(f"2000s\n{model_name}")
            all_accuracies.append(accuracy)
            era_colors.append(colors_2000s[i % len(colors_2000s)])
    
    # 2020s models
    if '2020s' in results.get('eras', {}):
        for i, (model_name, model_result) in enumerate(results['eras']['2020s'].items()):
            accuracy = model_result.get('accuracy', 0)
            all_labels.append(f"2020s\n{model_name}")
            all_accuracies.append(accuracy)
            era_colors.append(colors_2020s[i % len(colors_2020s)])
    
    # Create bar chart
    bars = ax.bar(range(len(all_labels)), all_accuracies, color=era_colors, alpha=0.8)
    
    # Retro styling
    ax.set_facecolor('#0a0a0a')
    fig.patch.set_facecolor('#0a0a0a')
    ax.tick_params(colors='#00ff00')
    ax.spines['bottom'].set_color('#00ff00')
    ax.spines['top'].set_color('#00ff00')
    ax.spines['left'].set_color('#00ff00')
    ax.spines['right'].set_color('#00ff00')
    
    ax.set_xlabel('Models by Era', color='#00ff00', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', color='#00ff00', fontsize=12, fontweight='bold')
    ax.set_title('RetroBrain: Accuracy Comparison Across Eras', 
                 color='#00ffff', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(all_labels)))
    ax.set_xticklabels(all_labels, rotation=45, ha='right', color='#00ff00', fontsize=9)
    ax.set_ylim(0, 1)
    ax.grid(True, color='#00ff00', alpha=0.2, linestyle='--')
    
    # Add glow effect (simulated with outline)
    for bar in bars:
        bar.set_edgecolor('#00ff00')
        bar.set_linewidth(1)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, facecolor='#0a0a0a', dpi=150, bbox_inches='tight')
    print(f"✅ Saved accuracy comparison to {save_path}")
    plt.close()


def plot_confusion_matrices(results, save_path='retrobrain/results/confusion_matrices.png'):
    """Plot confusion matrices for each era"""
    if results is None:
        return
    
    eras = ['1980s', '2000s', '2020s']
    n_eras = sum(1 for era in eras if era in results.get('eras', {}))
    
    if n_eras == 0:
        return
    
    fig, axes = plt.subplots(1, n_eras, figsize=(6*n_eras, 5))
    if n_eras == 1:
        axes = [axes]
    
    idx = 0
    for era_name in eras:
        if era_name not in results.get('eras', {}):
            continue
        
        era_results = results['eras'][era_name]
        # Use first model in era
        first_model_name = list(era_results.keys())[0]
        first_model_result = era_results[first_model_name]
        
        cm_data = np.array(first_model_result.get('confusion_matrix', [[0, 0], [0, 0]]))
        
        ax = axes[idx]
        im = ax.imshow(cm_data, cmap='Greens', aspect='auto', vmin=0, vmax=cm_data.max())
        
        # Add text annotations
        thresh = cm_data.max() / 2.
        for i in range(cm_data.shape[0]):
            for j in range(cm_data.shape[1]):
                ax.text(j, i, format(cm_data[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm_data[i, j] > thresh else "black",
                       fontweight='bold')
        
        ax.set_title(f'{era_name.upper()} Era\n{first_model_name}', 
                    color='#00ff00', fontsize=12, fontweight='bold')
        ax.set_xlabel('Predicted', color='#00ff00')
        ax.set_ylabel('Actual', color='#00ff00')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.tick_params(colors='#00ff00')
        
        fig.patch.set_facecolor('#0a0a0a')
        ax.set_facecolor('#0a0a0a')
        
        idx += 1
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, facecolor='#0a0a0a', dpi=150, bbox_inches='tight')
    print(f"✅ Saved confusion matrices to {save_path}")
    plt.close()


def generate_all_visualizations():
    """Generate all visualization charts"""
    print("\n📊 Generating visualizations...")
    
    results = load_results()
    if results is None:
        return
    
    plot_accuracy_comparison(results)
    plot_confusion_matrices(results)
    
    print("✅ All visualizations generated!\n")


if __name__ == "__main__":
    generate_all_visualizations()

