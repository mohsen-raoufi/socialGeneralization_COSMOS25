# -*- coding: utf-8 -*-
"""
Group vs Solo Trust Trajectories Comparison
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
    
    print("Loading group and solo data...")
    
    # Load group data (high volatility)
    try:
        group_df = pd.read_csv("./Data/ASG_dynamics_analysis/ASG_dynamics_group.csv")
    except Exception as e:
        print(f"Error loading group data: {e}")
        return None, None
    
    # Load solo data (low volatility - our filtered flat data)
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
    
    return sorted(list(common_groups))

def calculate_trajectory_volatility(agent_data, condition_name):
    """Calculate volatility metrics for an agent's trajectory"""
    
    volatilities = []
    for round_num in agent_data['round'].unique():
        round_data = agent_data[agent_data['round'] == round_num].sort_values('trial')
        if len(round_data) > 1:
            volatility = round_data['current_eps_soc'].std()
            volatilities.append(volatility)
    
    avg_volatility = np.mean(volatilities) if volatilities else 0
    print(f"   {condition_name} volatility: {avg_volatility:.6f}")
    
    return avg_volatility

def create_comparison_plot(group_df, solo_df, target_group=0):
    """Create 2x2 grid comparison plot where each subplot shows both group and solo trajectories for the same agent"""
    
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
    fig.suptitle(f'Group vs Solo Trust Trajectory Comparison - Group {target_group}', fontsize=18, fontweight='bold')
    
    # Agent positions in 2x2 grid
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    
    # Calculate volatilities for summary
    all_group_volatilities = []
    all_solo_volatilities = []
    
    # Plot each agent in its own subplot
    for i, agent in enumerate(common_agents[:4]):  # Ensure max 4 agents
        if i >= 4:
            break
            
        row, col = positions[i]
        ax = axes[row, col]
        
        # Get agent data for both conditions
        group_agent_data = group_data[group_data['agent'] == agent].sort_values(['round', 'trial'])
        solo_agent_data = solo_data[solo_data['agent'] == agent].sort_values(['round', 'trial'])
        
        # Calculate volatilities
        group_volatility = calculate_trajectory_volatility(group_agent_data, f"Group Agent {agent}")
        solo_volatility = calculate_trajectory_volatility(solo_agent_data, f"Solo Agent {agent}")
        
        all_group_volatilities.append(group_volatility)
        all_solo_volatilities.append(solo_volatility)
        
        # Plot GROUP trajectories (solid lines, red color)
        group_rounds = sorted(group_agent_data['round'].unique())
        for round_idx, round_num in enumerate(group_rounds):
            round_data = group_agent_data[group_agent_data['round'] == round_num].sort_values('trial')
            
            if len(round_data) > 0:
                # Normalize to start from 0 (subtract initial value)
                initial_value = round_data['current_eps_soc'].iloc[0]
                normalized_values = round_data['current_eps_soc'] - initial_value
                
                ax.plot(round_data['trial'], normalized_values, 
                       color='red', alpha=0.8, linewidth=2.5, linestyle='-',
                       marker='o', markersize=4, 
                       label=f'Group R{int(round_num)}' if round_idx < 3 else "")
        
        # Plot SOLO trajectories (dashed lines, blue color)
        solo_rounds = sorted(solo_agent_data['round'].unique())
        for round_idx, round_num in enumerate(solo_rounds):
            round_data = solo_agent_data[solo_agent_data['round'] == round_num].sort_values('trial')
            
            if len(round_data) > 0:
                # Normalize to start from 0 (subtract initial value)
                initial_value = round_data['current_eps_soc'].iloc[0]
                normalized_values = round_data['current_eps_soc'] - initial_value
                
                ax.plot(round_data['trial'], normalized_values, 
                       color='blue', alpha=0.7, linewidth=2.0, linestyle='--',
                       marker='s', markersize=3, 
                       label=f'Solo R{int(round_num)}' if round_idx < 3 else "")
        
        # Formatting for each subplot
        volatility_change = ((group_volatility - solo_volatility) / group_volatility * 100) if group_volatility > 0 else 0
        
        ax.set_title(f'Agent {int(agent)}: Group vs Solo\n'
                    f'Group Vol: {group_volatility:.4f}, Solo Vol: {solo_volatility:.4f}\n'
                    f'Reduction: {volatility_change:.1f}%', 
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('Trial', fontsize=11)
        ax.set_ylabel('Change in eps_soc (from initial)', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)  # Reference line at 0
        
        # Legend for first subplot only (to avoid clutter)
        if i == 0:
            # Create custom legend entries
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], color='red', linewidth=2.5, linestyle='-', marker='o', 
                       markersize=4, label='Group (High Volatility)'),
                Line2D([0], [0], color='blue', linewidth=2.0, linestyle='--', marker='s', 
                       markersize=3, label='Solo (Low Volatility)')
            ]
            ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        
        # Add volatility comparison text box
        comparison_text = f"Group: {group_volatility:.4f}\nSolo: {solo_volatility:.4f}\n"
        comparison_text += f"Flatter: {group_volatility/solo_volatility:.1f}x" if solo_volatility > 0 else "Solo: Perfectly flat"
        
        ax.text(0.02, 0.98, comparison_text, transform=ax.transAxes, 
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.9))
    
    # Hide unused subplots if fewer than 4 agents
    for i in range(len(common_agents), 4):
        row, col = positions[i]
        axes[row, col].set_visible(False)
    
    # Add overall comparison summary
    overall_group_volatility = np.mean(all_group_volatilities) if all_group_volatilities else 0
    overall_solo_volatility = np.mean(all_solo_volatilities) if all_solo_volatilities else 0
    overall_reduction = ((overall_group_volatility - overall_solo_volatility) / overall_group_volatility * 100) if overall_group_volatility > 0 else 0
    
    summary_text = f"Group {target_group} Summary: GROUP (solid lines) vs SOLO (dashed lines)\n"
    summary_text += f"Average Group Volatility: {overall_group_volatility:.4f}, Average Solo Volatility: {overall_solo_volatility:.4f}\n"
    summary_text += f"Overall Volatility Reduction: {overall_reduction:.1f}% | Solo trajectories are {overall_group_volatility/overall_solo_volatility:.1f}x flatter" if overall_solo_volatility > 0 else "Solo trajectories are perfectly flat"
    
    fig.text(0.5, 0.02, summary_text, ha='center', fontsize=12, 
             bbox=dict(boxstyle="round,pad=0.6", facecolor="lightblue", alpha=0.9))
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.08, 1, 0.94])
    
    # Save plot - create directory if it doesn't exist
    import os
    output_dir = "./Data/visualization_compare_group_vs_solo_trajectories"
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = f"{output_dir}/four_panel_comparison_group_{target_group}_vs_solo.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    # plt.show()
    
    # Print summary
    print(f"Common agents analyzed: {len(common_agents)}")
    print(f"GROUP condition average volatility: {overall_group_volatility:.6f}")
    print(f"SOLO condition average volatility: {overall_solo_volatility:.6f}")
    print(f"Overall volatility reduction: {overall_reduction:.1f}%")
    print(f"Solo trajectories are {overall_group_volatility/overall_solo_volatility:.1f}x flatter" if overall_solo_volatility > 0 else "Solo trajectories are perfectly flat")
    
    # Agent-by-agent
    print(f"\n AGENT-BY-AGENT:")
    for i, agent in enumerate(common_agents[:4]):
        group_vol = all_group_volatilities[i]
        solo_vol = all_solo_volatilities[i]
        reduction = ((group_vol - solo_vol) / group_vol * 100) if group_vol > 0 else 0
        print(f"   Agent {agent}: Group={group_vol:.4f}, Solo={solo_vol:.4f}, Reduction={reduction:.1f}%")

def main():
    """Main function to create group vs solo comparison for all groups"""

    # Load data
    group_df, solo_df = load_and_prepare_data()
    if group_df is None or solo_df is None:
        return
    
    # Find common groups
    common_groups = find_comparable_groups(group_df, solo_df)
    if len(common_groups) == 0:
        print("No common groups found for comparison")
        return
    
    # Store results for summary
    all_results = []
    successful_plots = 0
    
    # Create comparison plot for each group
    for i, group_id in enumerate(common_groups, 1):
        
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
            
            # Create comparison plot
            create_comparison_plot(group_df, solo_df, group_id)
            successful_plots += 1
            
            # Calculate summary statistics for this group
            group_volatilities = []
            solo_volatilities = []
            
            for agent in common_agents:
                group_agent_data = group_data[group_data['agent'] == agent]
                solo_agent_data = solo_data[solo_data['agent'] == agent]
                
                group_vol = calculate_trajectory_volatility(group_agent_data, f"Group {group_id} Agent {agent}")
                solo_vol = calculate_trajectory_volatility(solo_agent_data, f"Solo {group_id} Agent {agent}")
                
                group_volatilities.append(group_vol)
                solo_volatilities.append(solo_vol)
            
            avg_group_vol = np.mean(group_volatilities) if group_volatilities else 0
            avg_solo_vol = np.mean(solo_volatilities) if solo_volatilities else 0
            reduction = ((avg_group_vol - avg_solo_vol) / avg_group_vol * 100) if avg_group_vol > 0 else 0
            
            all_results.append({
                'group': group_id,
                'n_agents': len(common_agents),
                'avg_group_volatility': avg_group_vol,
                'avg_solo_volatility': avg_solo_vol,
                'volatility_reduction': reduction,
                'flatness_factor': avg_group_vol / avg_solo_vol if avg_solo_vol > 0 else float('inf')
            })
            
            print(f"Group {group_id} complete: {reduction:.1f}% volatility reduction")
            
        except Exception as e:
            print(f"Error processing Group {group_id}: {e}")
            continue
    
    print(f"All plots saved to ./Data/ directory 'four_panel_comparison_group_*_vs_solo.png'")
    
    if all_results:
        # Create summary DataFrame and save to the same output directory
        summary_df = pd.DataFrame(all_results)
        
        # Create the same output directory as plots
        output_dir = "./Data/visualization_compare_group_vs_solo_trajectories"
        os.makedirs(output_dir, exist_ok=True)
        
        summary_file = f"{output_dir}/group_vs_solo_volatility_summary.csv"
        
        # Add overall summary statistics as additional rows
        summary_stats = {
            'group': 'OVERALL_SUMMARY',
            'n_agents': summary_df['n_agents'].sum(),
            'avg_group_volatility': summary_df['avg_group_volatility'].mean(),
            'avg_solo_volatility': summary_df['avg_solo_volatility'].mean(), 
            'volatility_reduction': summary_df['volatility_reduction'].mean(),
            'flatness_factor': summary_df['flatness_factor'].mean()
        }
        
        summary_range_stats = {
            'group': 'RANGE_STATISTICS',
            'n_agents': f"Total: {summary_df['n_agents'].sum()}",
            'avg_group_volatility': f"{summary_df['avg_group_volatility'].min():.4f} - {summary_df['avg_group_volatility'].max():.4f}",
            'avg_solo_volatility': f"{summary_df['avg_solo_volatility'].min():.4f} - {summary_df['avg_solo_volatility'].max():.4f}",
            'volatility_reduction': f"{summary_df['volatility_reduction'].min():.1f}% - {summary_df['volatility_reduction'].max():.1f}%",
            'flatness_factor': f"{summary_df['flatness_factor'].min():.1f}x - {summary_df['flatness_factor'].max():.1f}x"
        }
        
        # Add separator and summary rows
        separator_row = {col: '---' for col in summary_df.columns}
        separator_row['group'] = 'SEPARATOR'
        
        # Create final dataframe with individual results + summary
        final_summary = pd.concat([
            summary_df,
            pd.DataFrame([separator_row]),
            pd.DataFrame([summary_stats]),
            pd.DataFrame([summary_range_stats])
        ], ignore_index=True)
        
        # Round numerical columns for readability
        for col in ['avg_group_volatility', 'avg_solo_volatility', 'volatility_reduction', 'flatness_factor']:
            if col in final_summary.columns:
                final_summary[col] = pd.to_numeric(final_summary[col], errors='coerce').round(4)
        
        final_summary.to_csv(summary_file, index=False)
        
        # Print aggregate statistics
        print(f"Total groups analyzed: {len(all_results)}")
        print(f"Total agents across all groups: {summary_df['n_agents'].sum()}")
        print(f"Average volatility reduction: {summary_df['volatility_reduction'].mean():.1f}%")
        print(f"Range of volatility reduction: {summary_df['volatility_reduction'].min():.1f}% - {summary_df['volatility_reduction'].max():.1f}%")
        print(f"Average flatness factor: {summary_df['flatness_factor'].mean():.1f}x")
        print(f"Most dramatic difference: Group {summary_df.loc[summary_df['volatility_reduction'].idxmax(), 'group']} ({summary_df['volatility_reduction'].max():.1f}% reduction)")
        print(f"Least dramatic difference: Group {summary_df.loc[summary_df['volatility_reduction'].idxmin(), 'group']} ({summary_df['volatility_reduction'].min():.1f}% reduction)")
        
        print(f"\nDetailed volatility summary saved: {summary_file}")
        print(f"Contains individual group statistics + overall summary statistics")
        print(f"Columns: group, n_agents, avg_group_volatility, avg_solo_volatility, volatility_reduction, flatness_factor")
        
        # Print top and bottom groups
        print(f"\nTOP 5 GROUPS (Highest Volatility Reduction):")
        top_groups = summary_df.nlargest(5, 'volatility_reduction')
        for _, row in top_groups.iterrows():
            print(f"Group {row['group']}: {row['volatility_reduction']:.1f}% reduction, {row['flatness_factor']:.1f}x flatter")
        
        print(f"\nBOTTOM 5 GROUPS (Lowest Volatility Reduction):")
        bottom_groups = summary_df.nsmallest(5, 'volatility_reduction')
        for _, row in bottom_groups.iterrows():
            print(f"Group {row['group']}: {row['volatility_reduction']:.1f}% reduction, {row['flatness_factor']:.1f}x flatter")

if __name__ == "__main__":
    main()