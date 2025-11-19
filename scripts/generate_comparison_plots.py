"""
Generate comparison plots for QAT methods
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11

# Load data
data = pd.read_csv('./results/qat_results_table.csv')

# Create output directory
os.makedirs('./results/figures', exist_ok=True)

# ============================================================================
# Plot 1: Quality Loss Comparison (Bar Chart)
# ============================================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

models = ['LSTM', 'ESPCN', 'SASRec']
methods_order = ['LSQ', 'DSQ', 'AdaRound', 'PACT', 'APoT']
colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12']

for idx, model in enumerate(models):
    ax = axes[idx]
    model_data = data[data['Model'] == model]
    model_data = model_data[model_data['Method'] != 'Baseline']

    # Sort by methods order
    model_data['Method'] = pd.Categorical(model_data['Method'], categories=methods_order, ordered=True)
    model_data = model_data.sort_values('Method')

    # Create bars
    bars = ax.bar(range(len(model_data)), model_data['Loss_Percent'],
                   color=[colors[methods_order.index(m)] if m in methods_order else 'gray'
                         for m in model_data['Method']])

    # Customize
    ax.set_xlabel('Quantization Method', fontsize=12, fontweight='bold')
    ax.set_ylabel('Quality Loss (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'{model}', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(model_data)))
    ax.set_xticklabels(model_data['Method'], rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, model_data['Loss_Percent'])):
        if val < 10:  # Normal case
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f'{val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
        else:  # LSTM-PACT case
            ax.text(bar.get_x() + bar.get_width()/2, 5,
                   f'{val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10, color='white')

    # Set y-limit
    max_val = model_data['Loss_Percent'].max()
    if max_val > 10:
        ax.set_ylim([0, min(max_val * 1.1, 50)])
    else:
        ax.set_ylim([0, max_val * 1.3])

plt.tight_layout()
plt.savefig('./results/figures/quality_loss_comparison.png', dpi=300, bbox_inches='tight')
print("[OK] Saved: quality_loss_comparison.png")

# ============================================================================
# Plot 2: Method Rankings Across Models
# ============================================================================

fig, ax = plt.subplots(figsize=(12, 6))

# Prepare data for heatmap
methods = ['LSQ', 'DSQ', 'AdaRound', 'PACT', 'APoT']
models = ['LSTM', 'ESPCN', 'SASRec']

# Create matrix of quality loss
matrix = np.zeros((len(methods), len(models)))

for i, method in enumerate(methods):
    for j, model in enumerate(models):
        row = data[(data['Model'] == model) & (data['Method'] == method)]
        if not row.empty:
            loss = row['Loss_Percent'].values[0]
            matrix[i, j] = loss if loss < 20 else np.nan  # Mark PACT-LSTM as failed

# Create heatmap
sns.heatmap(matrix, annot=True, fmt='.2f', cmap='RdYlGn_r',
            xticklabels=models, yticklabels=methods,
            cbar_kws={'label': 'Quality Loss (%)'},
            vmin=0, vmax=10, ax=ax)

ax.set_title('Quality Loss Heatmap: Methods vs Models', fontsize=14, fontweight='bold', pad=20)
ax.set_xlabel('Model Architecture', fontsize=12, fontweight='bold')
ax.set_ylabel('Quantization Method', fontsize=12, fontweight='bold')

# Mark LSTM-PACT as failed
if np.isnan(matrix[3, 0]):  # PACT on LSTM
    ax.text(0.5, 3.5, 'FAILED\n(46.35%)', ha='center', va='center',
           fontsize=10, fontweight='bold', color='red')

plt.tight_layout()
plt.savefig('./results/figures/method_heatmap.png', dpi=300, bbox_inches='tight')
print("[OK] Saved: method_heatmap.png")

# ============================================================================
# Plot 3: Best Method for Each Model
# ============================================================================

fig, ax = plt.subplots(figsize=(10, 6))

best_methods = []
best_losses = []

for model in models:
    model_data = data[(data['Model'] == model) & (data['Method'] != 'Baseline') & (data['Status'] == 'Success')]
    best = model_data.loc[model_data['Loss_Percent'].idxmin()]
    best_methods.append(f"{model}\n({best['Method']})")
    best_losses.append(best['Loss_Percent'])

bars = ax.bar(range(len(models)), best_losses,
              color=['#2ecc71', '#3498db', '#9b59b6'])

ax.set_xlabel('Model', fontsize=12, fontweight='bold')
ax.set_ylabel('Best Quality Loss (%)', fontsize=12, fontweight='bold')
ax.set_title('Best Quantization Method per Model', fontsize=14, fontweight='bold')
ax.set_xticks(range(len(models)))
ax.set_xticklabels(best_methods)
ax.grid(axis='y', alpha=0.3)

# Add value labels
for bar, val in zip(bars, best_losses):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
           f'{val:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)

ax.set_ylim([0, max(best_losses) * 1.3])

plt.tight_layout()
plt.savefig('./results/figures/best_methods.png', dpi=300, bbox_inches='tight')
print("[OK] Saved: best_methods.png")

# ============================================================================
# Plot 4: Method Performance Distribution
# ============================================================================

fig, ax = plt.subplots(figsize=(12, 7))

# Prepare data
method_losses = {method: [] for method in methods_order}

for method in methods_order:
    method_data = data[(data['Method'] == method) & (data['Status'] == 'Success')]
    if not method_data.empty:
        method_losses[method] = method_data['Loss_Percent'].tolist()

# Create box plot
positions = range(1, len(methods_order) + 1)
bp = ax.boxplot([method_losses[m] for m in methods_order],
                positions=positions,
                labels=methods_order,
                patch_artist=True,
                widths=0.6)

# Color boxes
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax.set_xlabel('Quantization Method', fontsize=12, fontweight='bold')
ax.set_ylabel('Quality Loss (%)', fontsize=12, fontweight='bold')
ax.set_title('Quality Loss Distribution Across All Models', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# Add mean markers
for i, method in enumerate(methods_order, 1):
    if method_losses[method]:
        mean_val = np.mean(method_losses[method])
        ax.plot(i, mean_val, 'D', color='red', markersize=8, zorder=3)
        ax.text(i, mean_val + 0.3, f'{mean_val:.1f}%', ha='center', fontweight='bold', fontsize=9)

plt.tight_layout()
plt.savefig('./results/figures/loss_distribution.png', dpi=300, bbox_inches='tight')
print("[OK] Saved: loss_distribution.png")

# ============================================================================
# Plot 5: Summary - Method Recommendation Matrix
# ============================================================================

fig, ax = plt.subplots(figsize=(10, 6))

# Define recommendations
recommendations = {
    'LSTM': {'Best': 'LSQ\n(0.99%)', 'Good': 'DSQ\n(3.05%)', 'Avoid': 'PACT\n(FAILED)'},
    'CNN': {'Best': 'DSQ\n(3.35%)', 'Good': 'AdaRound\n(3.39%)', 'Avoid': 'APoT\n(8.61%)'},
    'Transformer': {'Best': 'LSQ\n(0.20%)', 'Good': 'AdaRound\n(0.29%)', 'Avoid': '-'}
}

# Create table
table_data = []
for arch in ['LSTM', 'CNN', 'Transformer']:
    table_data.append([
        arch,
        recommendations[arch]['Best'],
        recommendations[arch]['Good'],
        recommendations[arch]['Avoid']
    ])

# Create table
table = ax.table(cellText=table_data,
                colLabels=['Architecture', 'Recommended', 'Alternative', 'Avoid'],
                cellLoc='center',
                loc='center',
                bbox=[0, 0, 1, 1])

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.5)

# Color cells
for i in range(4):
    table[(0, i)].set_facecolor('#34495e')
    table[(0, i)].set_text_props(weight='bold', color='white')

for i in range(1, 4):
    table[(i, 0)].set_facecolor('#ecf0f1')
    table[(i, 0)].set_text_props(weight='bold')
    table[(i, 1)].set_facecolor('#d5f4e6')  # Green
    table[(i, 2)].set_facecolor('#fff3cd')  # Yellow
    table[(i, 3)].set_facecolor('#f8d7da')  # Red

ax.axis('off')
ax.set_title('Quantization Method Selection Guide', fontsize=16, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('./results/figures/method_recommendations.png', dpi=300, bbox_inches='tight')
print("[OK] Saved: method_recommendations.png")

print("\n" + "="*60)
print("ALL PLOTS GENERATED SUCCESSFULLY!")
print("="*60)
print(f"Location: ./results/figures/")
print(f"Files created:")
print(f"  1. quality_loss_comparison.png")
print(f"  2. method_heatmap.png")
print(f"  3. best_methods.png")
print(f"  4. loss_distribution.png")
print(f"  5. method_recommendations.png")
print("="*60)
