# -*- coding: utf-8 -*-
"""
Trust-to-Reward Analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import os

def load_data():
    """Load both summary and trial-by-trial data"""
    # Load summary data (corrected path)
    summary_path = "./Data/ASG_dynamics_analysis/trust_changes_summary.csv"
    summary_df = pd.read_csv(summary_path)
    print(f"Loaded summary data: {len(summary_df)} agents")
    
    # Load trial-by-trial data
    dynamics_path = "./Data/ASG_dynamics_analysis/ASG_dynamics_group.csv"
    dynamics_df = pd.read_csv(dynamics_path)
    print(f"Loaded dynamics data: {len(dynamics_df)} trials")
    
    return summary_df, dynamics_df

def plot_summary_trust_vs_reward(summary_df, output_dir="./Data/visualization_trust_reward"):
    """Plot trust change vs total reward (agent-level)"""
    print("Creating agent-level trust change vs total reward plot...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    
    # Create scatter plot
    plt.scatter(summary_df['total_reward'], summary_df['trust_change'], 
                alpha=0.7, s=60, color='#3498DB', edgecolors='black', linewidth=0.5)
    
    # Add regression line
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        summary_df['total_reward'], summary_df['trust_change']
    )
    
    x_line = np.linspace(summary_df['total_reward'].min(), summary_df['total_reward'].max(), 100)
    y_line = slope * x_line + intercept
    plt.plot(x_line, y_line, color='red', linestyle='--', linewidth=2, 
             label=f'Linear fit (r={r_value:.3f}, p={p_value:.3f})')
    
    # Formatting
    plt.xlabel('Total Reward', fontsize=14, fontweight='bold')
    plt.ylabel('Trust Change (Final - Initial eps_soc)', fontsize=14, fontweight='bold')
    plt.title('Agent-Level Trust Change vs Total Reward', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Add correlation statistics
    pearson_r, pearson_p = pearsonr(summary_df['total_reward'], summary_df['trust_change'])
    spearman_r, spearman_p = spearmanr(summary_df['total_reward'], summary_df['trust_change'])
    
    stats_text = f'Pearson r = {pearson_r:.3f} (p = {pearson_p:.3f})\n'
    stats_text += f'Spearman ρ = {spearman_r:.3f} (p = {spearman_p:.3f})\n'
    stats_text += f'N = {len(summary_df)} agents'
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
             verticalalignment='top', fontsize=11)
    
    # Save plot
    output_file = os.path.join(output_dir, 'agent_trust_change_vs_total_reward.png')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print(f"Saved agent-level plot to: {output_file}")
    return pearson_r, pearson_p

def calculate_trial_eps_changes(dynamics_df):
    """Calculate eps_soc changes for each trial"""
    print("Calculating trial-by-trial eps_soc changes...")
    
    trial_changes = []
    
    # Group by agent and round
    for (agent, group), agent_data in dynamics_df.groupby(['agent', 'group']):
        for round_num in sorted(agent_data['round'].unique()):
            round_data = agent_data[agent_data['round'] == round_num].copy()
            round_data = round_data.sort_values('trial')
            
            # Calculate eps_soc change for each trial (current - previous)
            for i in range(1, len(round_data)):
                current_row = round_data.iloc[i]
                previous_row = round_data.iloc[i-1]
                
                eps_change = current_row['current_eps_soc'] - previous_row['current_eps_soc']
                
                trial_changes.append({
                    'agent': agent,
                    'group': group,
                    'round': round_num,
                    'trial': current_row['trial'],
                    'reward': current_row['reward'],
                    'eps_change': eps_change,
                    'is_random': current_row['is_random'],
                    'current_eps_soc': current_row['current_eps_soc'],
                    'cumulative_reward': current_row['cumulative_reward']
                })
    
    trial_changes_df = pd.DataFrame(trial_changes)
    print(f"Calculated {len(trial_changes_df)} trial-by-trial changes")
    
    return trial_changes_df

def plot_trial_eps_change_vs_reward(trial_changes_df, output_dir="./Data/visualization_trust_reward"):
    """Plot eps_soc change vs reward (trial-level)"""
    print("Creating trial-level eps_soc change vs reward plot...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter out random trials for cleaner analysis
    choice_trials = trial_changes_df[trial_changes_df['is_random'] == 0].copy()
    
    plt.figure(figsize=(12, 8))
    
    # Create scatter plot with some transparency due to many points
    plt.scatter(choice_trials['reward'], choice_trials['eps_change'], 
                alpha=0.4, s=20, color='#E74C3C', edgecolors='none')
    
    # Add regression line
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        choice_trials['reward'], choice_trials['eps_change']
    )
    
    x_line = np.linspace(choice_trials['reward'].min(), choice_trials['reward'].max(), 100)
    y_line = slope * x_line + intercept
    plt.plot(x_line, y_line, color='darkred', linestyle='-', linewidth=2, 
             label=f'Linear fit (r={r_value:.3f}, p={p_value:.3f})')
    
    # Add horizontal line at zero
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5, label='No change')
    
    # Formatting
    plt.xlabel('Trial Reward', fontsize=14, fontweight='bold')
    plt.ylabel('Δ eps_soc (Current - Previous)', fontsize=14, fontweight='bold')
    plt.title('Trial-Level: eps_soc Change vs Reward\n(Choice trials only)', 
              fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Add correlation statistics
    pearson_r, pearson_p = pearsonr(choice_trials['reward'], choice_trials['eps_change'])
    spearman_r, spearman_p = spearmanr(choice_trials['reward'], choice_trials['eps_change'])
    
    stats_text = f'Pearson r = {pearson_r:.3f} (p = {pearson_p:.3f})\n'
    stats_text += f'Spearman ρ = {spearman_r:.3f} (p = {spearman_p:.3f})\n'
    stats_text += f'N = {len(choice_trials)} choice trials'
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
             verticalalignment='top', fontsize=11)
    
    # Save plot
    output_file = os.path.join(output_dir, 'trial_eps_change_vs_reward.png')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print(f"Saved trial-level plot to: {output_file}")
    return pearson_r, pearson_p

def plot_binned_analysis(trial_changes_df, output_dir="./Data/visualization_trust_reward"):
    """Create binned analysis showing mean eps change by reward quartiles"""
    print("Creating binned analysis plot...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    choice_trials = trial_changes_df[trial_changes_df['is_random'] == 0].copy()
    
    # Create reward quartiles
    choice_trials['reward_quartile'] = pd.qcut(choice_trials['reward'], 
                                              q=4, labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)'])
    
    # Calculate mean and SEM by quartile
    quartile_stats = choice_trials.groupby('reward_quartile')['eps_change'].agg([
        'mean', 'sem', 'count', 'std'
    ]).reset_index()
    
    plt.figure(figsize=(10, 8))
    
    # Bar plot with error bars
    x_pos = range(len(quartile_stats))
    plt.bar(x_pos, quartile_stats['mean'], 
            yerr=quartile_stats['sem'], 
            capsize=5, alpha=0.7, color=['#E74C3C', '#3498DB', '#2ECC71', '#9B59B6'])
    
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Formatting
    plt.xlabel('Reward Quartile', fontsize=14, fontweight='bold')
    plt.ylabel('Mean Δ eps_soc ± SEM', fontsize=14, fontweight='bold')
    plt.title('Mean eps_soc Change by Reward Quartile', fontsize=16, fontweight='bold')
    plt.xticks(x_pos, quartile_stats['reward_quartile'])
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add sample sizes
    for i, (x, mean, count) in enumerate(zip(x_pos, quartile_stats['mean'], quartile_stats['count'])):
        plt.text(x, mean + 0.001, f'N={count}', ha='center', va='bottom', fontsize=10)
    
    # Save plot
    output_file = os.path.join(output_dir, 'eps_change_by_reward_quartile.png')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print(f"Saved quartile analysis to: {output_file}")
    
    return quartile_stats

def main():
    """Main function to generate trust-reward analysis"""
    
    # Create main output directory
    output_dir = "./Data/visualization_trust_reward"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Load data
    summary_df, dynamics_df = load_data()
    
    # Create plots
    print("\n" + "="*50)
    print("AGENT-LEVEL ANALYSIS")
    agent_r, agent_p = plot_summary_trust_vs_reward(summary_df, output_dir)
    
    print("\n" + "="*50)
    print("TRIAL-LEVEL ANALYSIS") 
    trial_changes_df = calculate_trial_eps_changes(dynamics_df)
    trial_r, trial_p = plot_trial_eps_change_vs_reward(trial_changes_df, output_dir)
    
    print("\n" + "="*50)
    print("QUARTILE ANALYSIS")
    quartile_stats = plot_binned_analysis(trial_changes_df, output_dir)
    
    # Save data files to output directory
    trial_changes_file = os.path.join(output_dir, 'trial_eps_changes.csv')
    quartile_stats_file = os.path.join(output_dir, 'reward_quartile_stats.csv')
    
    trial_changes_df.to_csv(trial_changes_file, index=False)
    quartile_stats.to_csv(quartile_stats_file, index=False)
    
    print(f"Saved trial changes data: {trial_changes_file}")
    print(f"Saved quartile stats: {quartile_stats_file}")
    
    # Summary report
    print(f"Agent-level correlation (trust change vs total reward):")
    print(f"r = {agent_r:.3f}, p = {agent_p:.3f}")
    print(f"Trial-level correlation (eps change vs reward):")
    print(f"r = {trial_r:.3f}, p = {trial_p:.3f}")
    
    # Interpretation
    if abs(agent_r) > 0.3:
        agent_strength = "strong" if abs(agent_r) > 0.5 else "moderate"
        agent_direction = "positive" if agent_r > 0 else "negative"
        print(f"Agent-level: {agent_strength} {agent_direction} correlation")
    else:
        print(f"Agent-level: weak correlation")
        
    if abs(trial_r) > 0.1:
        trial_strength = "strong" if abs(trial_r) > 0.3 else "moderate" if abs(trial_r) > 0.2 else "weak"
        trial_direction = "positive" if trial_r > 0 else "negative"
        print(f"Trial-level: {trial_strength} {trial_direction} correlation")
    else:
        print(f"Trial-level: very weak correlation")
    
    print(f"Generated files:")
    print(f"   • agent_trust_change_vs_total_reward.png")
    print(f"   • trial_eps_change_vs_reward.png") 
    print(f"   • eps_change_by_reward_quartile.png")
    print(f"   • trial_eps_changes.csv")
    print(f"   • reward_quartile_stats.csv")

if __name__ == "__main__":
    main()