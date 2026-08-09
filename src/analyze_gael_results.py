#!/usr/bin/env python3
"""
Analyze GAEL fit results by comparing with true parameters from metadata.
"""

import json
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
from pathlib import Path
import argparse
from collections import defaultdict
import logging

# Define parameter display names and units
PARAM_INFO = {
    'teff': {'name': 'T_eff', 'unit': 'K', 'label': r'$T_{\rm eff}$ [K]', 'color': 'tab:blue'},
    'logg': {'name': 'log g', 'unit': 'dex', 'label': r'$\log g$ [dex]', 'color': 'tab:orange'},
    'vsini': {'name': 'v sin i', 'unit': 'km/s', 'label': r'$v \sin i$ [km/s]', 'color': 'tab:green'},
    'xi': {'name': 'ξ', 'unit': 'km/s', 'label': r'$\xi$ [km/s]', 'color': 'tab:red'},
    'z': {'name': '[M/H]', 'unit': 'dex', 'label': '[M/H] [dex]', 'color': 'tab:purple'},
    'zeta': {'name': 'ζ', 'unit': 'km/s', 'label': r'$\zeta$ [km/s]', 'color': 'tab:brown'},
    'he': {'name': 'log(He/H)', 'unit': 'dex', 'label': r'$\log$(He/H) [dex]', 'color': 'tab:pink'},
    'vrad': {'name': 'v_rad', 'unit': 'km/s', 'label': r'$v_{\rm rad}$ [km/s]', 'color': 'tab:gray'},
}

logger = logging.getLogger(__name__)

def setup_logging(log_file: str = "gael_analysis.log", level: str = "INFO"):
    # Clear existing handlers to allow reconfiguration (useful in notebooks/tests)
    root = logging.getLogger()
    if root.handlers:
        for h in list(root.handlers):
            root.removeHandler(h)

    root.setLevel(logging.DEBUG)  # capture everything; handlers will filter

    # Formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler (more verbose)
    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    root.addHandler(fh)

    # Console handler (level from arg)
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    ch = logging.StreamHandler()
    ch.setLevel(level_map.get(level.upper(), logging.INFO))
    ch.setFormatter(formatter)
    root.addHandler(ch)

class GAELAnalyzer:
    def __init__(self, data_dir='.', results_dir='fit_results', exclude_params=None):
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir)
        self.data = defaultdict(list)
        self.exclude_params = set(exclude_params or [])
        
    def load_single_dataset(self, folder_idx):
        """Load metadata and fit results for a single folder."""
        folder_name = f"{folder_idx:04d}"
        
        # Load metadata
        metadata_path = self.data_dir / folder_name / "metadata.json"
        if not metadata_path.exists():
            logger.warning(f"{folder_name}: metadata.json not found at {metadata_path}")
            return None
            
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        except Exception:
            logger.exception(f"{folder_name}: Failed to read/parse metadata.json")
            return None
        
        # Load fit results
        fit_path = self.results_dir / folder_name / "fit_parameters.csv"
        if not fit_path.exists():
            logger.warning(f"{folder_name}: fit_parameters.csv not found at {fit_path}")
            return None
            
        fit_params = {}
        fit_errors = {}
        
        try:
            with open(fit_path, 'r') as f:
                reader = csv.reader(f)
                # Skip header row if it exists
                first_row = next(reader, None)
                try:
                    if first_row and (first_row[0].lower() in ['parameter', 'param', 'name'] or 
                                      first_row[1].lower() in ['value', 'val']):
                        pass  # header detected
                    else:
                        # data row
                        if first_row and len(first_row) >= 3:
                            param_name = first_row[0]
                            try:
                                value = float(first_row[1])
                                error = float(first_row[2])
                                fit_params[param_name] = value
                                fit_errors[param_name] = error
                            except ValueError:
                                logger.debug(f"{folder_name}: Skipping first row due to non-numeric value/error: {first_row}")
                except Exception:
                    logger.debug(f"{folder_name}: Could not reliably detect header in CSV; continuing as data.", exc_info=True)
                
                # Process remaining rows
                for row in reader:
                    if len(row) >= 3:
                        param_name = row[0]
                        try:
                            value = float(row[1])
                            error = float(row[2])
                            fit_params[param_name] = value
                            fit_errors[param_name] = error
                        except ValueError:
                            logger.debug(f"{folder_name}: Skipping row due to non-numeric value/error: {row}")
                            continue
        except Exception:
            logger.exception(f"{folder_name}: Failed to read/parse fit_parameters.csv")
            return None
        
        return {
            'metadata': metadata,
            'fit_params': fit_params,
            'fit_errors': fit_errors,
            'folder': folder_name
        }
    
    def extract_comparisons(self, dataset):
        """Extract parameter comparisons from a single dataset."""
        metadata = dataset['metadata']
        fit_params = dataset['fit_params']
        fit_errors = dataset['fit_errors']
        
        try:
            true_params = metadata['true_parameters']
        except KeyError:
            logger.error(f"{dataset.get('folder','????')}: 'true_parameters' missing in metadata")
            return {}
        comparisons = {}
        
        # Compare global parameters
        for param in ['teff', 'logg', 'vsini', 'xi', 'z', 'zeta', 'he']:
            if param in self.exclude_params:
                continue
                
            fit_key = f'c1_{param}'
            if fit_key in fit_params and param in true_params:
                comparisons[param] = {
                    'true': true_params[param],
                    'fit': fit_params[fit_key],
                    'error': fit_errors.get(fit_key, np.nan),
                    'diff': fit_params[fit_key] - true_params[param]
                }
            else:
                logger.debug(f"{dataset.get('folder','????')}: Missing comparison keys for {param} (fit_key={fit_key})")
        
        # Compare individual vrad values
        if 'vrad' not in self.exclude_params:
            vrad_comparisons = []
            spectra = metadata.get('spectra', [])
            if not isinstance(spectra, list):
                logger.warning(f"{dataset.get('folder','????')}: 'spectra' not a list in metadata")
                spectra = []
            for i, spectrum in enumerate(spectra, 1):
                fit_key = f'c1_vrad_d{i}'
                if fit_key in fit_params:
                    if 'vrad_actual' not in spectrum:
                        logger.debug(f"{dataset.get('folder','????')}: spectrum {i} missing 'vrad_actual'")
                        continue
                    vrad_comparisons.append({
                        'true': spectrum['vrad_actual'],
                        'fit': fit_params[fit_key],
                        'error': fit_errors.get(fit_key, np.nan),
                        'diff': fit_params[fit_key] - spectrum['vrad_actual'],
                        'spectrum_idx': i
                    })
            
            if vrad_comparisons:
                comparisons['vrad'] = vrad_comparisons
            
        return comparisons
    
    def collect_all_data(self, start_idx, end_idx):
        """Collect data from all folders."""
        logger.info("Loading data...")
        
        if self.exclude_params:
            logger.info(f"Excluding parameters: {', '.join(self.exclude_params)}")
        
        successful_loads = 0
        failed_loads = 0
        
        for idx in range(start_idx, end_idx + 1):
            try:
                dataset = self.load_single_dataset(idx)
                if dataset:
                    comparisons = self.extract_comparisons(dataset)
                    
                    # Store global parameters
                    for param, comp in comparisons.items():
                        if param != 'vrad':
                            self.data[param].append(comp)
                        else:
                            # Store vrad comparisons
                            self.data['vrad'].extend(comp)
                    successful_loads += 1
                else:
                    failed_loads += 1
            except Exception:
                logger.exception(f"Unexpected error loading dataset {idx:04d}")
                failed_loads += 1
        
        logger.info(f"Successfully loaded {successful_loads} datasets")
        logger.info(f"Failed to load {failed_loads} datasets")
        
        # Print summary of loaded parameters
        for param, data_list in self.data.items():
            logger.info(f"  {param}: {len(data_list)} measurements")
    
    def plot_scatter_histograms(self):
        """Create scatter plots and histograms for all parameters."""
        # Determine layout - only plot parameters that are not excluded
        params = [p for p in PARAM_INFO.keys() 
                 if p in self.data and self.data[p] and p not in self.exclude_params]
        n_params = len(params)
        
        if n_params == 0:
            logger.warning("No data to plot!")
            return
        
        logger.info(f"Plotting {n_params} parameters: {', '.join(params)}")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 4 * n_params))
        gs = GridSpec(n_params, 3, width_ratios=[2, 2, 1], hspace=0.3, wspace=0.3)
        
        for i, param in enumerate(params):
            param_data = self.data[param]
            color = PARAM_INFO[param]['color']
            
            if not param_data:
                logger.debug(f"No data for parameter {param} after filtering")
                continue
                
            # Extract values
            true_vals = np.array([d['true'] for d in param_data])
            fit_vals = np.array([d['fit'] for d in param_data])
            errors = np.array([d.get('error', np.nan) for d in param_data])
            diffs = np.array([d['diff'] for d in param_data])
            
            # 1. True vs Fitted scatter plot
            ax1 = fig.add_subplot(gs[i, 0])
            try:
                ax1.errorbar(true_vals, fit_vals, yerr=errors, fmt='o', alpha=0.6, 
                             markersize=4, capsize=3, capthick=1, color=color)
            except Exception:
                logger.exception(f"Error plotting True vs Fitted for {param}")
            
            # Add diagonal line
            try:
                val_range = [min(true_vals.min(), fit_vals.min()), 
                             max(true_vals.max(), fit_vals.max())]
                ax1.plot(val_range, val_range, 'k--', alpha=0.5, label='1:1 line')
            except ValueError:
                logger.warning(f"Insufficient data to compute value range for {param}")
            
            ax1.set_xlabel(f'True {PARAM_INFO[param]["label"]}')
            ax1.set_ylabel(f'Fitted {PARAM_INFO[param]["label"]}')
            ax1.set_title(f'{PARAM_INFO[param]["name"]} Comparison')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # 2. Residuals scatter plot
            ax2 = fig.add_subplot(gs[i, 1])
            try:
                ax2.errorbar(true_vals, diffs, yerr=errors, fmt='o', alpha=0.6,
                             markersize=4, capsize=3, capthick=1, color=color)
            except Exception:
                logger.exception(f"Error plotting residuals for {param}")
            ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
            
            ax2.set_xlabel(f'True {PARAM_INFO[param]["label"]}')
            ax2.set_ylabel(f'Residual (Fit - True) [{PARAM_INFO[param]["unit"]}]')
            ax2.set_title(f'{PARAM_INFO[param]["name"]} Residuals')
            ax2.grid(True, alpha=0.3)
            
            # Add RMS text
            try:
                rms = np.sqrt(np.mean(diffs**2))
                ax2.text(0.02, 0.98, f'RMS = {rms:.3f} {PARAM_INFO[param]["unit"]}',
                         transform=ax2.transAxes, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            except Exception:
                logger.exception(f"Error computing RMS for {param}")
            
            # 3. Histogram of residuals
            ax3 = fig.add_subplot(gs[i, 2])
            try:
                mean_diff = np.mean(diffs)
                std_diff = np.std(diffs)
                # Avoid degenerate std
                if not np.isfinite(std_diff) or std_diff == 0:
                    hist_range = (mean_diff - 1, mean_diff + 1)
                else:
                    hist_range = (mean_diff - 4*std_diff, mean_diff + 4*std_diff)
                
                n, bins, patches = ax3.hist(diffs, bins=30, range=hist_range, 
                                            orientation='horizontal', alpha=0.7, 
                                            color=color, edgecolor='black')
                
                # Add normal distribution overlay if std_diff > 0
                if std_diff and np.isfinite(std_diff):
                    x_hist = np.linspace(hist_range[0], hist_range[1], 100)
                    bin_width = bins[1] - bins[0] if len(bins) > 1 else 1.0
                    y_hist = len(diffs) * bin_width * \
                             np.exp(-(x_hist - mean_diff)**2 / (2 * std_diff**2)) / \
                             (std_diff * np.sqrt(2 * np.pi))
                    ax3.plot(y_hist, x_hist, 'r-', linewidth=2, label='Normal dist.')
                    ax3.legend()
                else:
                    logger.debug(f"Skipping normal overlay for {param} due to zero/NaN std")
                
                ax3.set_xlabel('Count')
                ax3.set_ylabel(f'Residual [{PARAM_INFO[param]["unit"]}]')
                ax3.set_title('Distribution')
                ax3.grid(True, alpha=0.3, axis='y')
                
                ax3.text(0.98, 0.98, f'μ = {mean_diff:.3f}\nσ = {std_diff:.3f}',
                         transform=ax3.transAxes, verticalalignment='top',
                         horizontalalignment='right',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            except Exception:
                logger.exception(f"Error creating histogram for {param}")
        
        try:
            plt.suptitle('GAEL Fit Results Analysis', fontsize=16, y=0.995)
            plt.tight_layout()
        except Exception:
            logger.exception("Error finalizing figure layout")
        
        # Save figure
        output_file = 'gael_analysis_results.png'
        try:
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            logger.info(f"Saved analysis plots to {output_file}")
        except Exception:
            logger.exception(f"Failed to save analysis plots to {output_file}")
        
        # Also create a summary statistics file
        try:
            self.save_statistics_summary()
        except Exception:
            logger.exception("Failed to save statistics summary")
        
        try:
            plt.show()
        except Exception:
            logger.exception("Error displaying the plot window")
    
    def save_statistics_summary(self):
        """Save summary statistics to a text file."""
        summary_path = 'gael_analysis_summary.txt'
        with open(summary_path, 'w') as f:
            f.write("GAEL Fit Analysis Summary\n")
            f.write("=" * 50 + "\n\n")
            
            if self.exclude_params:
                f.write(f"Excluded parameters: {', '.join(self.exclude_params)}\n\n")
            
            for param in PARAM_INFO.keys():
                if param not in self.data or not self.data[param] or param in self.exclude_params:
                    continue
                    
                param_data = self.data[param]
                
                diffs = np.array([d['diff'] for d in param_data])
                errors = np.array([d.get('error', np.nan) for d in param_data])
                
                f.write(f"{PARAM_INFO[param]['name']} ({param}):\n")
                f.write(f"  Number of measurements: {len(diffs)}\n")
                f.write(f"  Mean residual: {np.mean(diffs):.4f} {PARAM_INFO[param]['unit']}\n")
                f.write(f"  Std deviation: {np.std(diffs):.4f} {PARAM_INFO[param]['unit']}\n")
                f.write(f"  RMS: {np.sqrt(np.mean(diffs**2)):.4f} {PARAM_INFO[param]['unit']}\n")
                f.write(f"  Median residual: {np.median(diffs):.4f} {PARAM_INFO[param]['unit']}\n")
                f.write(f"  Mean fit error: {np.nanmean(errors):.4f} {PARAM_INFO[param]['unit']}\n")
                f.write(f"  Min residual: {np.min(diffs):.4f} {PARAM_INFO[param]['unit']}\n")
                f.write(f"  Max residual: {np.max(diffs):.4f} {PARAM_INFO[param]['unit']}\n")
                f.write("\n")
            
        logger.info(f"Saved summary statistics to {summary_path}")

def main():
    parser = argparse.ArgumentParser(description='Analyze GAEL fit results')
    parser.add_argument('--start', type=int, default=1, help='Starting folder index (default: 1)')
    parser.add_argument('--end', type=int, required=True, help='Ending folder index')
    parser.add_argument('--data-dir', type=str, default='.', help='Directory containing mock data folders (default: .)')
    parser.add_argument('--results-dir', type=str, default='fit_results', help='Directory containing fit results (default: fit_results)')
    parser.add_argument('--exclude', nargs='*', default=[], 
                        help='Parameters to exclude from plots (e.g., --exclude teff logg)')
    parser.add_argument('--log-file', type=str, default='gael_analysis.log', help='Log file path (default: gael_analysis.log)')
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG','INFO','WARNING','ERROR','CRITICAL'],
                        help='Console log level (default: INFO)')
    
    args = parser.parse_args()

    setup_logging(log_file=args.log_file, level=args.log_level)
    logger.info("Starting GAEL analysis")
    
    # Validate exclude parameters
    invalid_params = [p for p in args.exclude if p not in PARAM_INFO]
    if invalid_params:
        logger.warning(f"Invalid parameter names to exclude: {invalid_params}")
        logger.info(f"Valid parameters are: {list(PARAM_INFO.keys())}")
        args.exclude = [p for p in args.exclude if p in PARAM_INFO]
    
    analyzer = GAELAnalyzer(args.data_dir, args.results_dir, args.exclude)
    try:
        analyzer.collect_all_data(args.start, args.end)
        analyzer.plot_scatter_histograms()
    except Exception:
        logger.exception("Fatal error during analysis")
        raise

if __name__ == "__main__":
    main()