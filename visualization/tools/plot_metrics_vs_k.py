#!/usr/bin/env python3
"""
Metrics vs K curves for zscore normalization (by domain).

Main figure: 2x2 grid showing how metrics change with K for each domain (D0, D1, D2, D3).
Ablation: Bar chart comparing zscore vs centered_pit on unseen at K=16.

Usage:
    python tools/plot_metrics_vs_k.py \
        --zscore_summary data/nuscenes_planA_allscenes_T32_stride32_zscore_seed0/fewshot_eval/summary.json \
        --pit_summary data/nuscenes_planA_T32_centered_pit_seed0/fewshot_eval/summary.json \
        --outdir figures
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# Metrics to plot
METRICS = ['mmd', 'kl', 'v_wass', 'corr_frob']
METRIC_LABELS = {
    'mmd': 'MMD',
    'kl': 'KL',
    'v_wass': r'$W_v$',
    'corr_frob': r'$E_{\mathrm{corr}}$'
}

# Colors and markers for each domain
DOMAIN_COLORS = {
    0: '#1f77b4',    # Blue
    1: '#ff7f0e',    # Orange
    2: '#2ca02c',    # Green
    3: '#d62728'     # Red
}
DOMAIN_MARKERS = {
    0: 'o',          # Circle
    1: 's',          # Square
    2: '^',          # Triangle
    3: 'D'           # Diamond
}
DOMAIN_LABELS = {
    0: 'D0',
    1: 'D1',
    2: 'D2',
    3: 'D3'
}


def load_summary(path: Path) -> List[Dict]:
    """Load summary.json file (list of result dicts)."""
    with open(path) as f:
        return json.load(f)


def extract_kshot_metrics(summary: List[Dict], metrics: List[str], domains: List[int]) -> Dict:
    """
    Extract k_shot metrics for different K values, per domain.
    
    Returns:
        Dict mapping domain -> metric -> {K: value}
    """
    # Collect values: domain -> metric -> K -> value
    results = {d: {m: {} for m in metrics} for d in domains}
    
    for entry in summary:
        if entry.get('mode') != 'k_shot':
            continue
        
        domain = entry.get('domain')
        if domain not in domains:
            continue
        
        K = entry.get('K', 0)
        if K <= 0:
            continue
        
        for m in metrics:
            if m in entry:
                results[domain][m][K] = entry[m]
    
    return results


def extract_baseline(summary: List[Dict], metrics: List[str], domains: List[int]) -> Dict:
    """Extract no_prompt baseline, per domain."""
    results = {d: {} for d in domains}
    
    for entry in summary:
        if entry.get('mode') != 'no_prompt':
            continue
        
        domain = entry.get('domain')
        if domain not in domains:
            continue
        
        for m in metrics:
            if m in entry:
                results[domain][m] = entry[m]
    
    return results


def plot_zscore_metrics_vs_k(
    domain_kshot: Dict,
    domain_baseline: Dict,
    output_path: Path
):
    """
    Create 2x2 grid: zscore metrics vs K for each domain.
    """
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    axes = axes.flatten()
    
    domains = sorted(domain_kshot.keys())
    
    for idx, metric in enumerate(METRICS):
        ax = axes[idx]
        
        all_ks = set()
        
        # Plot each domain
        for domain in domains:
            domain_data = domain_kshot[domain].get(metric, {})
            if domain_data:
                ks = sorted(domain_data.keys())
                vals = [domain_data[k] for k in ks]
                all_ks.update(ks)
                ax.plot(ks, vals,
                       color=DOMAIN_COLORS.get(domain, '#000000'),
                       linestyle='-',
                       marker=DOMAIN_MARKERS.get(domain, 'o'),
                       markersize=8,
                       linewidth=2,
                       label=DOMAIN_LABELS.get(domain, f'D{domain}'))
        
        # Baseline for each domain (dotted line)
        for domain in domains:
            if metric in domain_baseline.get(domain, {}):
                # Only add label on first metric to avoid duplicate legend entries
                label = f'{DOMAIN_LABELS.get(domain, f"D{domain}")} (no_prompt)' if idx == 0 else None
                ax.axhline(y=domain_baseline[domain][metric],
                          color=DOMAIN_COLORS.get(domain, '#000000'),
                          linestyle=':',
                          alpha=0.5,
                          linewidth=1.5,
                          label=label)
        
        # Axis labels
        ax.set_xlabel('K')
        ax.set_ylabel(METRIC_LABELS.get(metric, metric))
        
        # X-axis ticks
        if all_ks:
            ax.set_xticks(sorted(all_ks))
        
        ax.grid(True, alpha=0.3)
        
        # Legend only on first subplot
        if idx == 0:
            ax.legend(loc='upper right', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print(f"  Saved: {output_path}")


def export_zscore_csv(
    domain_kshot: Dict,
    domain_baseline: Dict,
    output_path: Path
):
    """Export zscore metrics to CSV."""
    rows = []
    
    domains = sorted(domain_kshot.keys())
    
    # k_shot for each domain
    for domain in domains:
        for metric in METRICS:
            for K, val in domain_kshot[domain].get(metric, {}).items():
                rows.append({
                    'domain': domain,
                    'metric': metric,
                    'K': K,
                    'value': val,
                    'type': 'k_shot'
                })
    
    # Baseline for each domain
    for domain in domains:
        for metric, val in domain_baseline.get(domain, {}).items():
            rows.append({
                'domain': domain,
                'metric': metric,
                'K': 0,
                'value': val,
                'type': 'no_prompt'
            })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"  Saved: {output_path}")


def plot_ablation_norm(
    zscore_summary: List[Dict],
    pit_summary: List[Dict],
    output_path: Path,
    K: int = 16
):
    """
    Bar chart comparing zscore vs centered_pit on unseen (D3) at K=16.
    Only key metrics: MMD, v_wass
    """
    ablation_metrics = ['mmd', 'v_wass']
    metric_labels = {'mmd': 'MMD',     'v_wass': r'$W_v$'}
    
    # Extract values for unseen at K=16
    zscore_vals = {}
    pit_vals = {}
    
    for entry in zscore_summary:
        if entry.get('mode') == 'k_shot' and entry.get('domain') == 3 and entry.get('K') == K:
            for m in ablation_metrics:
                if m in entry:
                    zscore_vals[m] = entry[m]
    
    for entry in pit_summary:
        if entry.get('mode') == 'k_shot' and entry.get('domain') == 3 and entry.get('K') == K:
            for m in ablation_metrics:
                if m in entry:
                    pit_vals[m] = entry[m]
    
    if not zscore_vals or not pit_vals:
        print(f"  WARNING: Missing data for ablation plot (zscore={zscore_vals}, pit={pit_vals})")
        return
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=(5, 4))
    
    x = np.arange(len(ablation_metrics))
    width = 0.35
    
    zscore_values = [zscore_vals.get(m, 0) for m in ablation_metrics]
    pit_values = [pit_vals.get(m, 0) for m in ablation_metrics]
    
    bars1 = ax.bar(x - width/2, zscore_values, width, label='zscore', 
                   color='#1f77b4', edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, pit_values, width, label='centered_pit',
                   color='#ff7f0e', edgecolor='black', linewidth=0.5)
    
    # Add value labels
    for bar in bars1:
        h = bar.get_height()
        ax.annotate(f'{h:.3f}', xy=(bar.get_x() + bar.get_width()/2, h),
                   xytext=(0, 3), textcoords='offset points',
                   ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        h = bar.get_height()
        ax.annotate(f'{h:.3f}', xy=(bar.get_x() + bar.get_width()/2, h),
                   xytext=(0, 3), textcoords='offset points',
                   ha='center', va='bottom', fontsize=8)
    
    ax.set_xticks(x)
    ax.set_xticklabels([metric_labels[m] for m in ablation_metrics])
    ax.legend(loc='upper left', fontsize=9)
    # ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print(f"  Saved: {output_path}")
    
    # Also save CSV
    csv_path = output_path.with_suffix('.csv')
    rows = []
    for m in ablation_metrics:
        rows.append({'normalization': 'zscore', 'metric': m, 'K': K, 'value': zscore_vals.get(m)})
        rows.append({'normalization': 'centered_pit', 'metric': m, 'K': K, 'value': pit_vals.get(m)})
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description='Plot metrics vs K curves')
    parser.add_argument('--zscore_summary', type=str, required=True,
                        help='Path to zscore summary.json')
    parser.add_argument('--pit_summary', type=str, required=True,
                        help='Path to centered_pit summary.json')
    parser.add_argument('--outdir', type=str, default='figures',
                        help='Output directory')
    args = parser.parse_args()
    
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("METRICS VS K PLOTS")
    print("=" * 60)
    
    # Load summaries
    print(f"\n[1] Loading summaries...")
    zscore_summary = load_summary(Path(args.zscore_summary))
    pit_summary = load_summary(Path(args.pit_summary))
    
    # Check available data
    zscore_ks_by_domain = {0: set(), 1: set(), 2: set(), 3: set()}
    for e in zscore_summary:
        if e.get('mode') == 'k_shot':
            domain = e.get('domain')
            if domain in zscore_ks_by_domain:
                zscore_ks_by_domain[domain].add(e.get('K'))
    
    for domain in [0, 1, 2, 3]:
        if zscore_ks_by_domain[domain]:
            print(f"    zscore D{domain} K values: {sorted(zscore_ks_by_domain[domain])}")
    
    # ========== MAIN FIGURE: zscore metrics by domain ==========
    print(f"\n[2] Extracting zscore metrics...")
    all_domains = [0, 1, 2, 3]
    
    zscore_domains = extract_kshot_metrics(zscore_summary, METRICS, domains=all_domains)
    zscore_baseline = extract_baseline(zscore_summary, METRICS, domains=all_domains)
    
    for domain in all_domains:
        for m in METRICS:
            ks = list(zscore_domains[domain].get(m, {}).keys())
            if ks:
                print(f"    D{domain} {m}: K={sorted(ks)}")
    
    print(f"\n[3] Generating main figure (zscore metrics by domain)...")
    plot_zscore_metrics_vs_k(
        zscore_domains, zscore_baseline,
        outdir / 'zscore_metrics_vs_k_by_domain.png'
    )
    
    export_zscore_csv(
        zscore_domains, zscore_baseline,
        outdir / 'zscore_metrics_vs_k_by_domain.csv'
    )
    
    # ========== ABLATION: zscore vs centered_pit on unseen K=16 ==========
    print(f"\n[4] Generating ablation figure (zscore vs pit, unseen K=16)...")
    plot_ablation_norm(zscore_summary, pit_summary, outdir / 'ablation_norm_unseen_k16.png', K=16)
    
    print(f"\n[Done] Outputs saved to {outdir}/")


if __name__ == '__main__':
    main()
