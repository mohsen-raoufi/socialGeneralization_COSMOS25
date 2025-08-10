# -*- coding: utf-8 -*-
"""
Raw eps_soc Trust Trajectories Comparison
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Global color mapping for rounds
ROUND_COLORS = [
    '#E74C3C',  # Round 0 - Red
    '#3498DB',  # Round 1 - Blue  
    '#2ECC71',  # Round 2 - Green
    '#9B59B6',  # Round 3 - Purple
    '#F39C12',  # Round 4 - Orange
    '#1ABC9C',  # Round 5 - Turquoise
    '#E91E63',  # Round 6 - Pink
    '#34495E',  # Round 7 - Dark Gray
]

def load_and_prepare_data():
    """Load both group and solo data"""
    
    # Load group data (group condition)
    try:
        group_df = pd.read_csv("./Data/ASG_dynamics_analysis/ASG_dynamics_group.csv")
    except Exception as e:
        print(f"Error loading group data: {e}")
        return None, None
    
    # Load solo data (solo condition)
    try:
        solo_df = pd.read_csv("./Data/ASG_dynamics_analysis/ASG_dynamics_solo.csv")
    except Exception as e:
        print(f"Error loading solo data: {e}")
        return None, None
    
    return group_df, solo_df

def find_comparable_groups(group_df, solo_df):
    """Find groups that exist in both datasets for comparison"""
    
    group_groups = set(group_df['group'].unique())
    solo_groups = set(solo_df['group'].unique())
    
    common_groups = group_groups.intersection(solo_groups)
    print(f"Common groups available for comparison: {sorted(list(common_groups))}")
    
    return sorted(list(common_groups))

def calculate_trajectory_stats(agent_data, condition_name):
    """Calculate statistics for an agent's raw eps_soc trajectory"""
    
    all_eps_soc = agent_data['current_eps_soc'].values
    mean_eps_soc = np.mean(all_eps_soc)
    std_eps_soc = np.std(all_eps_soc)
    min_eps_soc = np.min(all_eps_soc)
    max_eps_soc = np.max(all_eps_soc)
    range_eps_soc = max_eps_soc - min_eps_soc
    
    print(f"   {condition_name}: Mean={mean_eps_soc:.3f}, Std={std_eps_soc:.3f}, Range={range_eps_soc:.3f}")
    
    return {
        'mean': mean_eps_soc,
        'std': std_eps_soc, 
        'min': min_eps_soc,
        'max': max_eps_soc,
        'range': range_eps_soc
    }

def create_raw_comparison_plot(group_df, solo_df, target_group=0):
    """Create 2x2 grid comparison plot showing RAW eps_soc values (no normalization)"""
    
    # Filter data for target group
    group_data = group_df[group_df['group'] == target_group]
    solo_data = solo_df[solo_df['group'] == target_group]
    
    if len(group_data) == 0:
        print(f"No group data found for Group {target_group}")
        return
    if len(solo_data) == 0:
        print(f"No solo data found for Group {target_group}")
        return
    
    print(f"Group data: {len(group_data)} rows, {group_data['agent'].nunique()} agents")
    print(f"Solo data: {len(solo_data)} rows, {solo_data['agent'].nunique()} agents")
    
    # Get common agents between both datasets
    group_agents = set(group_data['agent'].unique())
    solo_agents = set(solo_data['agent'].unique())
    common_agents = sorted(list(group_agents.intersection(solo_agents)))
    
    if len(common_agents) == 0:
        print(f"No common agents found between group and solo data for Group {target_group}")
        return
    
    print(f"Common agents for comparison: {common_agents}")
    
    # Create 2x2 grid subplots
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle(f'Raw Trust Levels (eps_soc) - Group vs Solo - Group {target_group}', fontsize=18, fontweight='bold')
    
    # Agent positions in 2x2 grid
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    
    # Collect statistics for summary
    all_group_stats = []
    all_solo_stats = []
    
    # Plot each agent in its own subplot
    for i, agent in enumerate(common_agents[:4]):  # Ensure max 4 agents
        if i >= 4:
            break
            
        row, col = positions[i]
        ax = axes[row, col]
        
        # Get agent data for both conditions
        group_agent_data = group_data[group_data['agent'] == agent].sort_values(['round', 'trial'])
        solo_agent_data = solo_data[solo_data['agent'] == agent].sort_values(['round', 'trial'])
        
        # Calculate raw statistics
        group_stats = calculate_trajectory_stats(group_agent_data, f"Group Agent {agent}")
        solo_stats = calculate_trajectory_stats(solo_agent_data, f"Solo Agent {agent}")
        
        all_group_stats.append(group_stats)
        all_solo_stats.append(solo_stats)
        
        # Plot GROUP trajectories (solid lines, red color) - RAW VALUES
        group_rounds = sorted(group_agent_data['round'].unique())
        for round_idx, round_num in enumerate(group_rounds):
            round_data = group_agent_data[group_agent_data['round'] == round_num].sort_values('trial')
            
            if len(round_data) > 0:
                # Use RAW eps_soc values (no normalization)
                raw_values = round_data['current_eps_soc']
                
                ax.plot(round_data['trial'], raw_values, 
                       color='red', alpha=0.8, linewidth=2.5, linestyle='-',
                       marker='o', markersize=4, 
                       label=f'Group R{int(round_num)}' if round_idx < 3 else "")
        
        # Plot SOLO trajectories (dashed lines, blue color) - RAW VALUES  
        solo_rounds = sorted(solo_agent_data['round'].unique())
        for round_idx, round_num in enumerate(solo_rounds):
            round_data = solo_agent_data[solo_agent_data['round'] == round_num].sort_values('trial')
            
            if len(round_data) > 0:
                # Use RAW eps_soc values (no normalization)
                raw_values = round_data['current_eps_soc']
                
                ax.plot(round_data['trial'], raw_values, 
                       color='blue', alpha=0.7, linewidth=2.0, linestyle='--',
                       marker='s', markersize=3, 
                       label=f'Solo R{int(round_num)}' if round_idx < 3 else "")
        
        # Formatting for each subplot
        ax.set_title(f'Agent {int(agent)}: Raw Trust Levels\n'
                    f'Group: μ={group_stats["mean"]:.2f}, σ={group_stats["std"]:.2f}\n'
                    f'Solo: μ={solo_stats["mean"]:.2f}, σ={solo_stats["std"]:.2f}', 
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('Trial', fontsize=11)
        ax.set_ylabel('eps_soc (Raw Trust Level)', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Set reasonable y-axis limits based on the data
        all_values = list(group_agent_data['current_eps_soc']) + list(solo_agent_data['current_eps_soc'])
        if all_values:
            y_min = max(0, min(all_values) - 1)  # Don't go below 0
            y_max = max(all_values) + 1
            ax.set_ylim(y_min, y_max)
        
        # Legend for first subplot only (to avoid clutter)
        if i == 0:
            # Create custom legend entries
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], color='red', linewidth=2.5, linestyle='-', marker='o', 
                       markersize=4, label='Group/Social Condition'),
                Line2D([0], [0], color='blue', linewidth=2.0, linestyle='--', marker='s', 
                       markersize=3, label='Solo/Individual Condition')
            ]
            ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        
        # Add statistics comparison text box
        comparison_text = f"Group Range: {group_stats['range']:.2f}\nSolo Range: {solo_stats['range']:.2f}\n"
        comparison_text += f"Difference: {abs(group_stats['mean'] - solo_stats['mean']):.2f}"
        
        ax.text(0.02, 0.98, comparison_text, transform=ax.transAxes, 
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.9))
    
    # Hide unused subplots if fewer than 4 agents
    for i in range(len(common_agents), 4):
        row, col = positions[i]
        axes[row, col].set_visible(False)
    
    # Add overall comparison summary
    overall_group_mean = np.mean([s['mean'] for s in all_group_stats]) if all_group_stats else 0
    overall_solo_mean = np.mean([s['mean'] for s in all_solo_stats]) if all_solo_stats else 0
    overall_group_std = np.mean([s['std'] for s in all_group_stats]) if all_group_stats else 0
    overall_solo_std = np.mean([s['std'] for s in all_solo_stats]) if all_solo_stats else 0
    
    summary_text = f"Group {target_group} Raw eps_soc Comparison Summary\n"
    summary_text += f"GROUP (red solid): μ={overall_group_mean:.2f}, σ={overall_group_std:.3f} | "
    summary_text += f"SOLO (blue dashed): μ={overall_solo_mean:.2f}, σ={overall_solo_std:.3f}\n"
    summary_text += f"Mean Difference: {abs(overall_group_mean - overall_solo_mean):.2f} | "
    summary_text += f"Higher Trust: {'Group' if overall_group_mean < overall_solo_mean else 'Solo'} (Lower eps_soc = Higher Trust)"
    
    fig.text(0.5, 0.02, summary_text, ha='center', fontsize=12, 
             bbox=dict(boxstyle="round,pad=0.6", facecolor="lightblue", alpha=0.9))
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.08, 1, 0.94])
    
    # Save plot - create directory if it doesn't exist
    output_dir = "./Data/visualization_raw_eps_soc_trajectories_comparison"
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = f"{output_dir}/raw_eps_soc_group_{target_group}_vs_solo.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()  # Close to save memory
    
    print(f"Raw eps_soc comparison plot saved: {output_file}")
    
    return {
        'group': target_group,
        'n_agents': len(common_agents),
        'group_mean_eps_soc': overall_group_mean,
        'solo_mean_eps_soc': overall_solo_mean, 
        'group_std_eps_soc': overall_group_std,
        'solo_std_eps_soc': overall_solo_std,
        'mean_difference': abs(overall_group_mean - overall_solo_mean),
        'higher_trust_condition': 'Group' if overall_group_mean < overall_solo_mean else 'Solo'
    }

def main():
    """Main function to create raw eps_soc comparison for all groups"""
    
    # Load data
    group_df, solo_df = load_and_prepare_data()
    if group_df is None or solo_df is None:
        return
    
    # Find common groups
    common_groups = find_comparable_groups(group_df, solo_df)
    if len(common_groups) == 0:
        print("No common groups found for comparison")
        return
    
    print(f"\nProcessing {len(common_groups)} groups for raw eps_soc comparison")
    
    # Store results for summary
    all_results = []
    successful_plots = 0
    
    # Create comparison plot for each group
    for i, group_id in enumerate(common_groups, 1):
        print(f"\n{'='*50}")
        print(f"Processing Group {group_id} ({i}/{len(common_groups)})")

        try:
            # Filter data for this group
            group_data = group_df[group_df['group'] == group_id]
            solo_data = solo_df[solo_df['group'] == group_id]
            
            # Quick check for data availability
            if len(group_data) == 0 or len(solo_data) == 0:
                print(f"Skipping Group {group_id}: No data available")
                continue
            
            # Check for common agents
            group_agents = set(group_data['agent'].unique())
            solo_agents = set(solo_data['agent'].unique())
            common_agents = group_agents.intersection(solo_agents)
            
            if len(common_agents) == 0:
                print(f"Skipping Group {group_id}: No common agents between conditions")
                continue
            
            print(f"Group {group_id}: {len(common_agents)} common agents available")
            
            # Create raw comparison plot
            result = create_raw_comparison_plot(group_df, solo_df, group_id)
            if result:
                all_results.append(result)
                successful_plots += 1
                print(f"Group {group_id} complete")
            
        except Exception as e:
            print(f"Error processing Group {group_id}: {e}")
            continue
    
    # Generate overall summary and save CSV
    if all_results:  
        # Create summary DataFrame and save
        summary_df = pd.DataFrame(all_results)
        
        # Create output directory and save CSV
        output_dir = "./Data/raw_eps_soc_comparison"
        os.makedirs(output_dir, exist_ok=True)
        
        summary_file = f"{output_dir}/raw_eps_soc_summary.csv"
        
        # Add overall summary statistics
        summary_stats = {
            'group': 'OVERALL_SUMMARY',
            'n_agents': summary_df['n_agents'].sum(),
            'group_mean_eps_soc': summary_df['group_mean_eps_soc'].mean(),
            'solo_mean_eps_soc': summary_df['solo_mean_eps_soc'].mean(), 
            'group_std_eps_soc': summary_df['group_std_eps_soc'].mean(),
            'solo_std_eps_soc': summary_df['solo_std_eps_soc'].mean(),
            'mean_difference': summary_df['mean_difference'].mean(),
            'higher_trust_condition': f"{(summary_df['higher_trust_condition'] == 'Group').sum()} Group, {(summary_df['higher_trust_condition'] == 'Solo').sum()} Solo"
        }
        
        # Create final dataframe with summary
        final_summary = pd.concat([
            summary_df,
            pd.DataFrame([{'group': '---', **{k: '---' for k in summary_df.columns if k != 'group'}}]),
            pd.DataFrame([summary_stats])
        ], ignore_index=True)
        
        final_summary.to_csv(summary_file, index=False)
        
        # Print statistics
        print(f"Total groups analyzed: {len(all_results)}")
        print(f"Total agents across all groups: {summary_df['n_agents'].sum()}")
        print(f"Average GROUP eps_soc: {summary_df['group_mean_eps_soc'].mean():.3f}")
        print(f"Average SOLO eps_soc: {summary_df['solo_mean_eps_soc'].mean():.3f}")
        print(f"Overall higher trust: {'Group' if summary_df['group_mean_eps_soc'].mean() < summary_df['solo_mean_eps_soc'].mean() else 'Solo'}")
        print(f"Groups where Group has higher trust: {(summary_df['higher_trust_condition'] == 'Group').sum()}")
        print(f"Groups where Solo has higher trust: {(summary_df['higher_trust_condition'] == 'Solo').sum()}")
        
        print(f"\nRaw eps_soc summary saved: {summary_file}")
        print(f"All plots saved in: {output_dir}/")
        
        # Print top differences
        print(f"\nTOP 5 GROUPS (Largest Trust Differences):")
        top_diff = summary_df.nlargest(5, 'mean_difference')
        for _, row in top_diff.iterrows():
            print(f"Group {row['group']}: {row['mean_difference']:.3f} difference ({row['higher_trust_condition']} has higher trust)")

if __name__ == "__main__":
    main()