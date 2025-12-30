#!/usr/bin/env python3
"""
Aggregate evaluation results across all domains.

Recursively finds all summary.json files under data_dir and builds a unified DataFrame.

Usage:
    python tools/aggregate_domains.py --data_dir data/nuscenes_planA_... --out_dir results/

Output:
    - all_domains_summary.csv
    - all_domains_summary.json
    - Pretty-printed table to stdout
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import numpy as np


def find_summary_files(data_dir: Path) -> List[Path]:
    """Recursively find all summary.json files."""
    summary_files = []
    for root, dirs, files in os.walk(data_dir):
        for f in files:
            if f == 'summary.json':
                summary_files.append(Path(root) / f)
    return sorted(summary_files)


def load_summary(path: Path) -> List[Dict[str, Any]]:
    """Load a summary.json file (can be a list or single dict)."""
    with open(path, 'r') as f:
        data = json.load(f)
    if isinstance(data, dict):
        return [data]
    return data


def build_dataframe(summary_files: List[Path]) -> pd.DataFrame:
    """Build a unified DataFrame from all summary files."""
    all_records = []
    
    for path in summary_files:
        try:
            records = load_summary(path)
            for rec in records:
                rec['file_path'] = str(path)
                # Extract eval folder name for context
                rec['eval_folder'] = path.parent.name
                all_records.append(rec)
        except Exception as e:
            print(f"  WARNING: Failed to load {path}: {e}")
    
    if not all_records:
        raise ValueError("No valid summary.json files found!")
    
    df = pd.DataFrame(all_records)
    return df


def compute_deltas(df: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
    """
    Compute delta vs no_prompt baseline for each domain and metric.
    
    For each (domain, K), delta = metric(k_shot) - metric(no_prompt, same domain)
    """
    df = df.copy()
    
    # Get no_prompt baselines per domain
    no_prompt = df[df['mode'] == 'no_prompt'].copy()
    if no_prompt.empty:
        print("  WARNING: No 'no_prompt' baselines found, skipping delta computation.")
        return df
    
    # Create baseline lookup: domain -> metric values
    baseline_lookup = {}
    for _, row in no_prompt.iterrows():
        domain = row['domain']
        baseline_lookup[domain] = {m: row.get(m, np.nan) for m in metrics}
    
    # Compute deltas
    for metric in metrics:
        delta_col = f'{metric}_delta'
        df[delta_col] = np.nan
        
        for idx, row in df.iterrows():
            domain = row['domain']
            if domain in baseline_lookup:
                baseline_val = baseline_lookup[domain].get(metric, np.nan)
                current_val = row.get(metric, np.nan)
                if pd.notna(baseline_val) and pd.notna(current_val):
                    df.loc[idx, delta_col] = current_val - baseline_val
    
    return df


def pretty_print_table(df: pd.DataFrame, metrics: List[str]):
    """Pretty-print results grouped by domain."""
    print("\n" + "=" * 100)
    print("AGGREGATED EVALUATION RESULTS")
    print("=" * 100)
    
    domains = sorted(df['domain'].unique())
    
    for domain in domains:
        print(f"\n--- Domain {domain} ---")
        domain_df = df[df['domain'] == domain].copy()
        
        # Sort by mode (no_prompt first, then k_shot by K)
        def sort_key(row):
            if row['mode'] == 'no_prompt':
                return (0, 0)
            elif row['mode'] == 'k_shot':
                return (1, row.get('K', 0))
            elif row['mode'] == 'shuffled':
                return (2, row.get('K', 0))
            elif row['mode'] == 'minus_pam':
                return (3, row.get('K', 0))
            else:
                return (9, 0)
        
        domain_df['_sort'] = domain_df.apply(sort_key, axis=1)
        domain_df = domain_df.sort_values('_sort').drop(columns=['_sort'])
        
        # Select columns to display
        display_cols = ['mode', 'K'] + [m for m in metrics if m in domain_df.columns]
        display_cols = [c for c in display_cols if c in domain_df.columns]
        
        # Format for display
        display_df = domain_df[display_cols].copy()
        for col in display_df.columns:
            if display_df[col].dtype in [np.float64, np.float32]:
                display_df[col] = display_df[col].apply(lambda x: f'{x:.4f}' if pd.notna(x) else '-')
        
        print(display_df.to_string(index=False))
    
    print("\n" + "=" * 100)


def main():
    parser = argparse.ArgumentParser(description='Aggregate evaluation results across domains')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Root data directory containing fewshot_eval folders')
    parser.add_argument('--out_dir', type=str, default=None,
                        help='Output directory (default: data_dir/fewshot_eval_all)')
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir) if args.out_dir else data_dir / 'fewshot_eval_all'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[Aggregate] Searching for summary.json files in: {data_dir}")
    
    # Find all summary files
    summary_files = find_summary_files(data_dir)
    print(f"[Aggregate] Found {len(summary_files)} summary.json files:")
    for f in summary_files:
        print(f"  - {f.relative_to(data_dir)}")
    
    if not summary_files:
        print("ERROR: No summary.json files found!")
        return
    
    # Build DataFrame
    df = build_dataframe(summary_files)
    print(f"[Aggregate] Built DataFrame with {len(df)} records")
    
    # Core metrics to track
    core_metrics = ['mmd', 'kl', 'mdd', 'v_wass', 'a_wass', 'curvature_wass', 'corr_frob']
    available_metrics = [m for m in core_metrics if m in df.columns]
    
    # Compute deltas vs no_prompt
    df = compute_deltas(df, available_metrics)
    
    # Pretty print
    pretty_print_table(df, available_metrics)
    
    # Save CSV
    csv_path = out_dir / 'all_domains_summary.csv'
    df.to_csv(csv_path, index=False)
    print(f"\n[Aggregate] Saved CSV: {csv_path}")
    
    # Save JSON (records format)
    json_path = out_dir / 'all_domains_summary.json'
    with open(json_path, 'w') as f:
        json.dump(df.to_dict('records'), f, indent=2, default=str)
    print(f"[Aggregate] Saved JSON: {json_path}")
    
    # Print summary statistics
    print("\n[Aggregate] Summary by domain and mode:")
    if 'mode' in df.columns and 'domain' in df.columns:
        summary = df.groupby(['domain', 'mode']).agg({
            m: ['mean', 'std'] for m in available_metrics if m in df.columns
        })
        print(summary.to_string())


if __name__ == '__main__':
    main()

