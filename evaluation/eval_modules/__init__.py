"""
Evaluation modules for TimeDP Plan A.

Modules:
- pit_diagnostics: Centered-PIT diagnosis and round-trip consistency checks
- plot_corr_heatmaps: Correlation heatmap generation
"""

from evaluation.eval_modules.pit_diagnostics import (
    run_pit_diagnostics,
    compute_norm_space_diagnostics,
    pit_round_trip_check,
    apply_pit_clip,
    CenteredPITInverter
)

from evaluation.eval_modules.plot_corr_heatmaps import (
    run_corr_heatmaps,
    generate_corr_heatmaps,
    compute_lag0_corr_matrix,
    compute_lag1_cross_corr_matrix
)

__all__ = [
    'run_pit_diagnostics',
    'compute_norm_space_diagnostics',
    'pit_round_trip_check',
    'apply_pit_clip',
    'CenteredPITInverter',
    'run_corr_heatmaps',
    'generate_corr_heatmaps',
    'compute_lag0_corr_matrix',
    'compute_lag1_cross_corr_matrix'
]

