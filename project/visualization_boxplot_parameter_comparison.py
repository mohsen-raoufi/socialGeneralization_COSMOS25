# -*- coding: utf-8 -*-
"""
Boxplot Comparison of ASG Parameters: Group vs Solo
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

def load_and_prepare_data():
    """Load both group and solo data and prepare for plotting"""
    
    print("Loading group and solo data...")
    
    try:
        # Load group data (social condition)
        group_df = pd.read_csv("./Data/ASG_dynamics_analysis/ASG_dynamics_group.csv")
        
        # Load solo data (individual condition)
        solo_df = pd.read_csv("./Data/ASG_dynamics_analysis/ASG_dynamics_solo.csv")
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
    
    # Get unique parameter combinations (one per agent)
    # Group by agent and group to get unique parameter sets
    group_params = group_df.groupby(['agent', 'group']).first().reset_index()
    solo_params = solo_df.groupby(['agent', 'group']).first().reset_index()
    
    # Add condition labels
    group_params['condition'] = 'Group'
    solo_params['condition'] = 'Solo'
    
    # Combine datasets
    combined_data = pd.concat([group_params, solo_params], ignore_index=True)
    
    print(f"Parameters range:")
    print(f"initial_eps_soc: {combined_data['initial_eps_soc'].min():.3f} - {combined_data['initial_eps_soc'].max():.3f}")
    print(f"eta_eps_soc: {combined_data['eta_eps_soc'].min():.6f} - {combined_data['eta_eps_soc'].max():.6f}")
    
    return combined_data

def create_parameter_boxplots(data):
    """Create side-by-side boxplots with scatter overlays"""
    
    plt.style.use('default')
    sns.set_palette("husl")
    
    colors = {
        'Group': '#E74C3C',  # Red for group
        'Solo': '#3498DB'    # Blue for solo
    }
    
    # Create figure with 2 subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('ASG Model Parameters: Group vs Solo Conditions', fontsize=16, fontweight='bold', y=0.95)
    
    # Add subtle grid
    for ax in [ax1, ax2]:
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_axisbelow(True)
    
    # Plot 1: initial_eps_soc
    print("Plotting initial_eps_soc...")
    
    # Create boxplot
    box_data_initial = [data[data['condition'] == 'Group']['initial_eps_soc'].values,
                       data[data['condition'] == 'Solo']['initial_eps_soc'].values]
    
    bp1 = ax1.boxplot(box_data_initial, 
                      positions=[1, 2],
                      patch_artist=True,
                      labels=['Group', 'Solo'],
                      widths=0.6,
                      showfliers=True,  # Show outliers
                      flierprops=dict(marker='o', markersize=4, alpha=0.6))
    
    # Color the boxplots
    bp1['boxes'][0].set_facecolor(colors['Group'])
    bp1['boxes'][0].set_alpha(0.7)
    bp1['boxes'][1].set_facecolor(colors['Solo']) 
    bp1['boxes'][1].set_alpha(0.7)
    
    # Add scatter points with jitter
    for condition, position in [('Group', 1), ('Solo', 2)]:
        condition_data = data[data['condition'] == condition]['initial_eps_soc']
        
        # Add jitter to x positions for better visibility
        jitter = np.random.normal(0, 0.05, len(condition_data))
        x_positions = np.full(len(condition_data), position) + jitter
        
        ax1.scatter(x_positions, condition_data, 
                   color=colors[condition], 
                   alpha=0.4, 
                   s=20, 
                   edgecolors='white', 
                   linewidth=0.5,
                   zorder=3)
    
    # Formatting for initial_eps_soc plot
    ax1.set_title('Initial Social Skepticism\n(initial_eps_soc)', fontsize=14, fontweight='bold', pad=20)
    ax1.set_ylabel('initial_eps_soc\n(Lower = More Trusting)', fontsize=12)
    ax1.set_xlabel('Condition', fontsize=12)
    ax1.tick_params(axis='both', labelsize=11)
    
    # Plot 2: eta_eps_soc
    print("   Plotting eta_eps_soc...")
    
    # Create boxplot
    box_data_eta = [data[data['condition'] == 'Group']['eta_eps_soc'].values,
                   data[data['condition'] == 'Solo']['eta_eps_soc'].values]
    
    bp2 = ax2.boxplot(box_data_eta,
                      positions=[1, 2], 
                      patch_artist=True,
                      labels=['Group', 'Solo'],
                      widths=0.6,
                      showfliers=True,
                      flierprops=dict(marker='o', markersize=4, alpha=0.6))
    
    # Color the boxplots
    bp2['boxes'][0].set_facecolor(colors['Group'])
    bp2['boxes'][0].set_alpha(0.7)
    bp2['boxes'][1].set_facecolor(colors['Solo'])
    bp2['boxes'][1].set_alpha(0.7)
    
    # Add scatter points with jitter
    for condition, position in [('Group', 1), ('Solo', 2)]:
        condition_data = data[data['condition'] == condition]['eta_eps_soc']
        
        # Add jitter to x positions
        jitter = np.random.normal(0, 0.05, len(condition_data))
        x_positions = np.full(len(condition_data), position) + jitter
        
        ax2.scatter(x_positions, condition_data,
                   color=colors[condition],
                   alpha=0.4,
                   s=20,
                   edgecolors='white',
                   linewidth=0.5,
                   zorder=3)
    
    # Formatting for eta_eps_soc plot
    ax2.set_title('Adaptation Rate\n(eta_eps_soc)', fontsize=14, fontweight='bold', pad=20)
    ax2.set_ylabel('eta_eps_soc\n(Higher = Faster Adaptation)', fontsize=12)
    ax2.set_xlabel('Condition', fontsize=12)
    ax2.tick_params(axis='both', labelsize=11)
    
    # Add statistical annotations
    print("   Adding statistical information...")
    
    # Calculate and display statistics
    for ax, param in [(ax1, 'initial_eps_soc'), (ax2, 'eta_eps_soc')]:
        group_vals = data[data['condition'] == 'Group'][param]
        solo_vals = data[data['condition'] == 'Solo'][param]
        
        # Calculate statistics
        group_mean = group_vals.mean()
        solo_mean = solo_vals.mean()
        group_std = group_vals.std()
        solo_std = solo_vals.std()
        
        # Perform t-test
        from scipy import stats
        t_stat, p_value = stats.ttest_ind(group_vals, solo_vals)
        
        # Add statistical text box
        stats_text = f"Group: μ={group_mean:.3f}, σ={group_std:.3f}\\n"
        stats_text += f"Solo: μ={solo_mean:.3f}, σ={solo_std:.3f}\\n"
        stats_text += f"Difference: {abs(group_mean - solo_mean):.3f}\\n"
        stats_text += f"t-test p: {p_value:.4f}"
        
        # Position text box
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.9))
        
        # Add significance indicator
        if p_value < 0.001:
            sig_text = "***"
        elif p_value < 0.01:
            sig_text = "**" 
        elif p_value < 0.05:
            sig_text = "*"
        else:
            sig_text = "ns"
        
        # Add significance bracket
        y_max = max(group_vals.max(), solo_vals.max())
        y_bracket = y_max * 1.05
        ax.plot([1, 2], [y_bracket, y_bracket], 'k-', linewidth=1)
        ax.plot([1, 1], [y_bracket, y_bracket * 0.98], 'k-', linewidth=1)
        ax.plot([2, 2], [y_bracket, y_bracket * 0.98], 'k-', linewidth=1)
        ax.text(1.5, y_bracket * 1.02, sig_text, ha='center', fontsize=12, fontweight='bold')
    
    # Create custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors['Group'], alpha=0.7, label='Group/Social Condition'),
        Patch(facecolor=colors['Solo'], alpha=0.7, label='Solo/Individual Condition')
    ]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.02), 
               ncol=2, fontsize=12, frameon=False)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.08, 1, 0.92])
    
    # Save plot
    output_dir = "./Data/visualization_boxplot_parameter_comparison"
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/ASG_parameters_boxplot_comparison.png"
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Parameter comparison plot saved: {output_file}")
    
    group_initial = data[data['condition'] == 'Group']['initial_eps_soc']
    solo_initial = data[data['condition'] == 'Solo']['initial_eps_soc']
    group_eta = data[data['condition'] == 'Group']['eta_eps_soc'] 
    solo_eta = data[data['condition'] == 'Solo']['eta_eps_soc']
    
    print(f"GROUP starts more trusting: {group_initial.mean():.3f} vs {solo_initial.mean():.3f}")
    print(f"SOLO starts more skeptical: {solo_initial.mean():.3f} vs {group_initial.mean():.3f}")
    print(f"GROUP adapts faster: {group_eta.mean():.3f} vs {solo_eta.mean():.3f}")
    print(f"SOLO adapts slower: {solo_eta.mean():.3f} vs {group_eta.mean():.3f}")
    
    trust_diff = solo_initial.mean() - group_initial.mean()
    adapt_diff = group_eta.mean() - solo_eta.mean()
    print(f"\nTrust difference: {trust_diff:.3f} (Solo more skeptical)")
    print(f"Adaptation difference: {adapt_diff:.3f} (Group faster)")
    
    return output_file

def save_parameter_summary(data):
    """Save detailed parameter statistics to CSV"""
    
    summary_stats = []
    
    for condition in ['Group', 'Solo']:
        condition_data = data[data['condition'] == condition]
        
        for param in ['initial_eps_soc', 'eta_eps_soc']:
            values = condition_data[param]
            
            summary_stats.append({
                'condition': condition,
                'parameter': param,
                'count': len(values),
                'mean': values.mean(),
                'std': values.std(),
                'median': values.median(),
                'min': values.min(),
                'max': values.max(),
                'q25': values.quantile(0.25),
                'q75': values.quantile(0.75),
                'iqr': values.quantile(0.75) - values.quantile(0.25)
            })
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(summary_stats)
    
    # Add statistical tests
    from scipy import stats
    
    # T-tests for each parameter
    test_results = []
    for param in ['initial_eps_soc', 'eta_eps_soc']:
        group_vals = data[data['condition'] == 'Group'][param]
        solo_vals = data[data['condition'] == 'Solo'][param]
        
        t_stat, p_value = stats.ttest_ind(group_vals, solo_vals)
        cohens_d = (group_vals.mean() - solo_vals.mean()) / np.sqrt(((group_vals.std()**2 + solo_vals.std()**2) / 2))
        
        test_results.append({
            'parameter': param,
            'test': 'independent_t_test',
            't_statistic': t_stat,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'effect_size': 'small' if abs(cohens_d) < 0.5 else 'medium' if abs(cohens_d) < 0.8 else 'large'
        })
    
    tests_df = pd.DataFrame(test_results)
    
    # Save both summary and tests
    output_dir = "./Data/parameter_comparison"
    os.makedirs(output_dir, exist_ok=True)
    
    summary_file = f"{output_dir}/ASG_parameter_statistics.csv"
    tests_file = f"{output_dir}/ASG_parameter_tests.csv"
    
    summary_df.to_csv(summary_file, index=False)
    tests_df.to_csv(tests_file, index=False)
    
    print(f"Parameter statistics saved: {summary_file}")
    print(f"Statistical tests saved: {tests_file}")
    
    return summary_file, tests_file

def main():
    """Main function to create parameter comparison boxplots"""
    
    # Load and prepare data
    data = load_and_prepare_data()
    if data is None:
        return
    
    # Create boxplot comparison
    plot_file = create_parameter_boxplots(data)
    
    # Save detailed statistics
    summary_file, tests_file = save_parameter_summary(data)
    
    print(f"\nDone")
    print(f"Plot saved: {plot_file}")
    print(f"Statistics: {summary_file}")
    print(f"Tests: {tests_file}")

if __name__ == "__main__":
    main()