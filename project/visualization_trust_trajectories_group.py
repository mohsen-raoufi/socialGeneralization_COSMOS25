# -*- coding: utf-8 -*-
"""
Trust Trajectories Visualization for Group
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Global fixed color mapping for rounds (manually defined hex colors)
# Maps round sequence position (0-7) to colors, not actual round numbers
ROUND_COLORS = [
    '#E74C3C',  # Round sequence 0 - Red
    '#3498DB',  # Round sequence 1 - Blue  
    '#2ECC71',  # Round sequence 2 - Green
    '#9B59B6',  # Round sequence 3 - Purple
    '#F39C12',  # Round sequence 4 - Orange
    '#1ABC9C',  # Round sequence 5 - Turquoise
    '#E91E63',  # Round sequence 6 - Pink
    '#34495E',  # Round sequence 7 - Dark Gray
]

def load_dynamics_data(data_path="./Data/ASG_dynamics_analysis/ASG_dynamics_group.csv"):
    """Load the reconstructed ASG dynamics data"""
    
    print("Loading ASG dynamics data...")
    
    try:
        dynamics_df = pd.read_csv(data_path)
        print(f"Loaded {len(dynamics_df)} trials from {dynamics_df[['agent', 'group']].drop_duplicates().shape[0]} unique agents")
        return dynamics_df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def calculate_agent_trust_metrics(agent_data):
    """Calculate trust-related metrics for a single agent"""
    
    trust_metrics = {}
    
    # Overall trust change
    initial_trust = agent_data['current_eps_soc'].iloc[0]
    final_trust = agent_data['current_eps_soc'].iloc[-1]
    trust_metrics['initial_trust'] = initial_trust
    trust_metrics['final_trust'] = final_trust
    trust_metrics['trust_change'] = final_trust - initial_trust
    trust_metrics['mean_trust'] = agent_data['current_eps_soc'].mean()
    trust_metrics['trust_volatility'] = agent_data['current_eps_soc'].std()
    trust_metrics['became_more_trusting'] = trust_metrics['trust_change'] < 0
    trust_metrics['total_reward'] = agent_data['cumulative_reward'].iloc[-1]
    
    return trust_metrics

def create_group_four_panel_plot(group_data, group_id, output_dir="./Data/visualization_trust_trajectories_group"):
    """Create a 2x2 grid plot showing individual agent trajectories within a group"""
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Group {int(group_id)} Individual Agent Trust Trajectories', 
                 fontsize=16, fontweight='bold')
    
    # Get all agents in this group (should be 4)
    agents = sorted(group_data['agent'].unique())
    
    # Agent positions in 2x2 grid
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    
    # Group-level statistics for summary
    group_trust_changes = []
    group_trust_volatilities = []
    
    # Plot each agent in its own subplot
    for i, agent in enumerate(agents[:4]):  # Ensure max 4 agents
        if i >= 4:
            break
            
        row, col = positions[i]
        ax = axes[row, col]
        
        # Get this agent's data
        agent_data = group_data[group_data['agent'] == agent].copy()
        agent_data = agent_data.sort_values(['round', 'trial'])
        
        # Calculate agent metrics
        agent_metrics = calculate_agent_trust_metrics(agent_data)
        group_trust_changes.append(agent_metrics['trust_change'])
        group_trust_volatilities.append(agent_metrics['trust_volatility'])
        
        # Plot trust trajectory for each round
        sorted_rounds = sorted(agent_data['round'].unique())
        for round_idx, round_num in enumerate(sorted_rounds):
            round_data = agent_data[agent_data['round'] == round_num]
            
            if len(round_data) > 0:
                # Use round sequence position (0-7) instead of actual round number for color
                color = ROUND_COLORS[round_idx % len(ROUND_COLORS)]
                ax.plot(round_data['trial'], round_data['current_eps_soc'], 
                       color=color, marker='o', markersize=4, alpha=0.8, 
                       linewidth=2.5, label=f'Round {int(round_num)} (seq {round_idx})')
        
        # Formatting for each subplot
        trust_direction = "↓ More Trusting" if agent_metrics['became_more_trusting'] else "↑ More Skeptical"
        
        ax.set_title(f'Agent {int(agent)} {trust_direction}\n'
                    f'Change: {agent_metrics["trust_change"]:.3f}, '
                    f'Volatility: {agent_metrics["trust_volatility"]:.3f}', 
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('Trial', fontsize=10)
        ax.set_ylabel('eps_soc (Social Distrust)', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Legend for first subplot only (to avoid clutter)
        if i == 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        
        # Set consistent y-axis limits across all subplots for comparison
        all_values = agent_data['current_eps_soc']
        if len(all_values) > 0:
            y_margin = (all_values.max() - all_values.min()) * 0.1 if all_values.max() != all_values.min() else 0.1
            ax.set_ylim(all_values.min() - y_margin, all_values.max() + y_margin)
        
        # Add text box with key statistics
        stats_text = f"Initial: {agent_metrics['initial_trust']:.2f}\n"
        stats_text += f"Final: {agent_metrics['final_trust']:.2f}\n"
        stats_text += f"Reward: {agent_metrics['total_reward']:.2f}"
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Hide unused subplots if fewer than 4 agents
    for i in range(len(agents), 4):
        row, col = positions[i]
        axes[row, col].set_visible(False)
    
    # Add group summary text
    agents_more_trusting = sum(1 for change in group_trust_changes if change < 0)
    mean_group_change = np.mean(group_trust_changes) if group_trust_changes else 0
    mean_group_volatility = np.mean(group_trust_volatilities) if group_trust_volatilities else 0
    
    summary_text = f"Group Summary: {agents_more_trusting}/{len(agents)} agents became more trusting\n"
    summary_text += f"Mean Trust Change: {mean_group_change:.3f}, Mean Volatility: {mean_group_volatility:.3f}"
    
    fig.text(0.5, 0.02, summary_text, ha='center', fontsize=12, 
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.08, 1, 0.94])
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'group_{int(group_id)}_four_panel.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Saved four-panel plot for Group {int(group_id)} to {output_file}")
    
    # Return group metrics
    return {
        'group': group_id,
        'n_agents': len(agents),
        'agents_more_trusting': agents_more_trusting,
        'mean_trust_change': mean_group_change,
        'mean_trust_volatility': mean_group_volatility,
        'trust_changes': group_trust_changes,
        'trust_volatilities': group_trust_volatilities
    }

def generate_all_group_four_panel_plots(dynamics_df, output_dir="./Data/trust_trajectories_group"):
    """Generate four-panel trust trajectory plots for all groups"""
    
    # Get unique groups
    groups = sorted(dynamics_df['group'].unique())
    print(f"Processing {len(groups)} unique groups...")
    
    # Store all group metrics
    all_group_metrics = {}
    
    # Process each group
    for i, group in enumerate(groups, 1):
        print(f"Processing Group {int(group)} ({i}/{len(groups)})...")
        
        # Get group data (all agents in this group)
        group_data = dynamics_df[dynamics_df['group'] == group].copy()
        
        if len(group_data) == 0:
            print(f"  No data found for Group {int(group)}")
            continue
        
        # Check number of agents
        n_agents = group_data['agent'].nunique()
        if n_agents != 4:
            print(f"  Warning: Group {int(group)} has {n_agents} agents instead of 4")
        
        # Create four-panel plot
        try:
            group_metrics = create_group_four_panel_plot(group_data, group, output_dir)
            all_group_metrics[group] = group_metrics
        except Exception as e:
            print(f"  Error processing Group {int(group)}: {e}")
            continue
    
    print(f"\nSuccessfully generated four-panel plots for {len(all_group_metrics)} groups")
    
    # Generate summary statistics
    generate_four_panel_summary_statistics(all_group_metrics, output_dir)
    
    return all_group_metrics

def generate_four_panel_summary_statistics(all_group_metrics, output_dir):
    """Generate and save summary statistics for four-panel analysis"""
    
    # Compile summary data
    summary_data = []
    all_individual_changes = []
    
    for group, metrics in all_group_metrics.items():
        summary_data.append({
            'group': group,
            'n_agents': metrics['n_agents'],
            'agents_more_trusting': metrics['agents_more_trusting'],
            'mean_trust_change': metrics['mean_trust_change'],
            'mean_trust_volatility': metrics['mean_trust_volatility']
        })
        
        # Collect individual agent changes
        all_individual_changes.extend(metrics['trust_changes'])
    
    summary_df = pd.DataFrame(summary_data)
    
    # Save summary statistics
    summary_file = os.path.join(output_dir, 'four_panel_group_summary.csv')
    summary_df.to_csv(summary_file, index=False)
    
    # Print key statistics
    print(f"\n=== FOUR-PANEL GROUP ANALYSIS SUMMARY ===")
    print(f"Total groups analyzed: {len(summary_df)}")
    print(f"Mean agents per group becoming more trusting: {summary_df['agents_more_trusting'].mean():.1f}/4")
    print(f"Groups where all 4 agents became more trusting: {(summary_df['agents_more_trusting'] == 4).sum()}")
    print(f"Groups where 0 agents became more trusting: {(summary_df['agents_more_trusting'] == 0).sum()}")
    print(f"Mean group trust change: {summary_df['mean_trust_change'].mean():.3f}")
    print(f"Mean group trust volatility: {summary_df['mean_trust_volatility'].mean():.3f}")
    
    print(f"\n=== INDIVIDUAL AGENT SUMMARY ===")
    print(f"Total individual agent observations: {len(all_individual_changes)}")
    agents_more_trusting = sum(1 for change in all_individual_changes if change < 0)
    print(f"Individual agents who became more trusting: {agents_more_trusting}/{len(all_individual_changes)} ({agents_more_trusting/len(all_individual_changes)*100:.1f}%)")
    print(f"Mean individual trust change: {np.mean(all_individual_changes):.3f}")
    print(f"Std individual trust change: {np.std(all_individual_changes):.3f}")
    
    print(f"\nSummary statistics saved to: {summary_file}")

def create_four_panel_overview(all_group_metrics, output_dir):
    """Create overview visualization for four-panel analysis"""
    
    print("Creating four-panel overview visualization...")
    
    # Compile data for overview
    agents_more_trusting = [metrics['agents_more_trusting'] for metrics in all_group_metrics.values()]
    mean_trust_changes = [metrics['mean_trust_change'] for metrics in all_group_metrics.values()]
    mean_volatilities = [metrics['mean_trust_volatility'] for metrics in all_group_metrics.values()]
    
    # All individual changes
    all_changes = []
    for metrics in all_group_metrics.values():
        all_changes.extend(metrics['trust_changes'])
    
    # Create overview figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Distribution of trusting agents per group
    axes[0, 0].hist(agents_more_trusting, bins=range(6), alpha=0.7, edgecolor='black', color='skyblue')
    axes[0, 0].set_xlabel('Number of Agents Becoming More Trusting per Group')
    axes[0, 0].set_ylabel('Number of Groups')
    axes[0, 0].set_title('Distribution of Trusting Agents per Group')
    axes[0, 0].set_xticks(range(5))
    axes[0, 0].grid(True, alpha=0.3)
    
    # Group mean trust changes
    axes[0, 1].hist(mean_trust_changes, bins=20, alpha=0.7, edgecolor='black', color='lightgreen')
    axes[0, 1].axvline(0, color='red', linestyle='--', linewidth=2, label='No Change')
    axes[0, 1].axvline(np.mean(mean_trust_changes), color='orange', linestyle='-', linewidth=2, 
                       label=f'Mean: {np.mean(mean_trust_changes):.3f}')
    axes[0, 1].set_xlabel('Group Mean Trust Change')
    axes[0, 1].set_ylabel('Number of Groups')
    axes[0, 1].set_title('Distribution of Group Mean Trust Changes')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Individual agent trust changes
    axes[1, 0].hist(all_changes, bins=30, alpha=0.7, edgecolor='black', color='lightcoral')
    axes[1, 0].axvline(0, color='red', linestyle='--', linewidth=2, label='No Change')
    axes[1, 0].axvline(np.mean(all_changes), color='orange', linestyle='-', linewidth=2, 
                       label=f'Mean: {np.mean(all_changes):.3f}')
    axes[1, 0].set_xlabel('Individual Agent Trust Change')
    axes[1, 0].set_ylabel('Number of Agents')
    axes[1, 0].set_title('Distribution of Individual Agent Trust Changes')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Trust volatility vs number of trusting agents
    axes[1, 1].scatter(mean_volatilities, agents_more_trusting, alpha=0.7, s=50, color='purple')
    axes[1, 1].set_xlabel('Group Mean Trust Volatility')
    axes[1, 1].set_ylabel('Number of Agents Becoming More Trusting')
    axes[1, 1].set_title('Trust Volatility vs Trusting Agents per Group')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Four-Panel Trust Analysis Overview', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    
    # Save overview plot
    overview_file = os.path.join(output_dir, 'four_panel_overview.png')
    plt.savefig(overview_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Four-panel overview plot saved to: {overview_file}")

def main():
    """Main function to generate all four-panel group visualizations"""
    
    # Load dynamics data
    dynamics_df = load_dynamics_data()
    
    if dynamics_df is None:
        print("Failed to load dynamics data. Exiting.")
        return
    
    # Generate all four-panel group plots
    output_dir = "./Data/trust_trajectories_group"
    all_group_metrics = generate_all_group_four_panel_plots(dynamics_df, output_dir)
    
    if not all_group_metrics:
        print("No group plots were generated successfully.")
        return
    
    # Create overview plot
    create_four_panel_overview(all_group_metrics, output_dir)
    
    print(f"All files saved to: {output_dir}")
    print(f"Generated {len(all_group_metrics)} four-panel group plots")
    
    # Final statistics
    total_agents = sum(metrics['n_agents'] for metrics in all_group_metrics.values())
    total_more_trusting = sum(metrics['agents_more_trusting'] for metrics in all_group_metrics.values())
    

    print(f"{len(all_group_metrics)} groups analyzed with {total_agents} total agents")
    print(f"{total_more_trusting}/{total_agents} agents ({total_more_trusting/total_agents*100:.1f}%) became more trusting")
    
    # Group-level insights
    groups_all_trusting = sum(1 for metrics in all_group_metrics.values() 
                             if metrics['agents_more_trusting'] == 4)
    groups_none_trusting = sum(1 for metrics in all_group_metrics.values() 
                              if metrics['agents_more_trusting'] == 0)
    
    print(f"   • {groups_all_trusting} groups where all 4 agents became more trusting")
    print(f"   • {groups_none_trusting} groups where no agents became more trusting")

if __name__ == "__main__":
    main()