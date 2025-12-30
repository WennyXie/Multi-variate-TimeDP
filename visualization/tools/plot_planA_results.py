#!/usr/bin/env python3
"""
Plot Plan A evaluation results.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load results
data_dir = Path('data/nuscenes_planA_allscenes_T32_stride32_zscore_seed0/fewshot_eval')
with open(data_dir / 'summary.json') as f:
    results = json.load(f)

# Organize results by domain and mode
domains = [0, 1, 2, 3]
modes = ['no_prompt', 'k_shot', 'shuffled']
K_values = [1, 4, 8, 16]
metrics = ['mmd', 'kl', 'v_wass', 'corr_frob']

# Create figures
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Plan A Evaluation: K vs Metrics on Unseen Domain (D3)', fontsize=14, fontweight='bold')

colors = {'no_prompt': '#2ca02c', 'k_shot': '#1f77b4', 'shuffled': '#ff7f0e'}
markers = {'no_prompt': 's', 'k_shot': 'o', 'shuffled': '^'}

for idx, metric in enumerate(metrics):
    ax = axes[idx // 2, idx % 2]
    
    # Filter for domain 3 (unseen)
    domain_results = [r for r in results if r['domain'] == 3]
    
    for mode in modes:
        mode_results = [r for r in domain_results if r['mode'] == mode]
        
        if mode == 'no_prompt':
            # Single point for no_prompt
            val = mode_results[0][metric]
            ax.axhline(y=val, color=colors[mode], linestyle='--', alpha=0.7, label='no_prompt')
        else:
            # K vs metric curve
            K_vals = [r['K'] for r in mode_results]
            metric_vals = [r[metric] for r in mode_results]
            ax.plot(K_vals, metric_vals, marker=markers[mode], color=colors[mode], 
                   label=mode, linewidth=2, markersize=8)
    
    ax.set_xlabel('K (number of prompts)')
    ax.set_ylabel(metric)
    ax.set_title(f'{metric.upper()}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(K_values)

plt.tight_layout()
plt.savefig(data_dir / 'unseen_domain_curves.png', dpi=150, bbox_inches='tight')
print(f"Saved: {data_dir / 'unseen_domain_curves.png'}")

# Create comparison bar chart for all domains at K=16
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
fig2.suptitle('Plan A: Seen vs Unseen Domain Comparison (K=16)', fontsize=14, fontweight='bold')

domain_names = ['D0: Boston\n(seen)', 'D1: SG-onenorth\n(seen)', 'D2: SG-queenstown\n(seen)', 'D3: SG-hollandvillage\n(UNSEEN)']

for idx, metric in enumerate(['v_wass', 'corr_frob']):
    ax = axes2[idx]
    
    x = np.arange(len(domains))
    width = 0.25
    
    for i, mode in enumerate(['no_prompt', 'k_shot', 'shuffled']):
        vals = []
        for d in domains:
            domain_results = [r for r in results if r['domain'] == d and r['mode'] == mode]
            if mode == 'no_prompt':
                vals.append(domain_results[0][metric])
            else:
                k16_result = [r for r in domain_results if r['K'] == 16]
                vals.append(k16_result[0][metric] if k16_result else 0)
        
        ax.bar(x + i * width, vals, width, label=mode, color=colors[mode], alpha=0.8)
    
    ax.set_xlabel('Domain')
    ax.set_ylabel(metric)
    ax.set_title(f'{metric.upper()} (K=16)')
    ax.set_xticks(x + width)
    ax.set_xticklabels(domain_names)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(data_dir / 'domain_comparison.png', dpi=150, bbox_inches='tight')
print(f"Saved: {data_dir / 'domain_comparison.png'}")

plt.show()
print("\nDone!")
