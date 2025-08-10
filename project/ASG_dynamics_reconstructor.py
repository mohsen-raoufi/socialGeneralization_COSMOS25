# -*- coding: utf-8 -*-
"""
ASG Dynamics Reconstructor
Creates detailed trajectories of eps_social adaptation for ASG model
Shows how social trust evolves trial-by-trial in group vs solo conditions
"""

import numpy as np
import pandas as pd
import os
from glob import glob

def reconstruct_asg_dynamics(data, fitted_params, condition_type="group"):
    """
    Reconstruct ASG eps_soc trajectories from experimental data and fitted parameters
    
    Parameters:
    -----------
    data : DataFrame
        Original experimental data (e2_data.csv)
    fitted_params : list
        ASG fitted parameters [agent, group, lambda, beta, tau, initial_eps_soc, eta_eps_soc]
    condition_type : str
        "group" or "solo" condition
        
    Returns:
    --------
    DataFrame with columns: round, trial, current_eps_soc, reward, cumulative_reward, 
                           is_random, agent, group, initial_eps_soc, eta_eps_soc
    """
    
    trajectories = []
    
    print(f"Reconstructing {condition_type} condition dynamics...")
    
    # Group parameters by agent to handle cross-validation duplicates
    agent_params = {}
    for param_row in fitted_params:
        if len(param_row[0]) < 7:  # Skip if not enough parameters
            continue
            
        agent_id = int(param_row[0][0])
        group_id = int(param_row[0][1])
        agent_key = (agent_id, group_id)
        
        if agent_key not in agent_params:
            agent_params[agent_key] = []
        agent_params[agent_key].append(param_row)
    
    print(f"Found parameters for {len(agent_params)} unique agents (after removing CV duplicates)")
    
    # Process each unique agent (taking mean of cross-validation parameters)
    for (agent_id, group_id), param_list in agent_params.items():
        
        # Compute mean parameters across cross-validation folds
        all_initial_eps_soc = []
        all_eta_eps_soc = []
        
        for param_row in param_list:
            # Extract ASG parameters (already log-transformed, need to exponentiate)
            initial_eps_soc = np.exp(param_row[0][5])
            eta_eps_soc = np.exp(param_row[0][6])
            all_initial_eps_soc.append(initial_eps_soc)
            all_eta_eps_soc.append(eta_eps_soc)
        
        # Use mean parameters to reduce cross-validation noise
        mean_initial_eps_soc = np.mean(all_initial_eps_soc)
        mean_eta_eps_soc = np.mean(all_eta_eps_soc)
        
        print(f"Processing Agent {agent_id}, Group {group_id}: initial_eps_soc={mean_initial_eps_soc:.3f}, eta_eps_soc={mean_eta_eps_soc:.3f} (averaged across {len(param_list)} CV folds)")
        
        # Get agent's data for this condition
        if condition_type == "group":
            tasktype_filter = "social"
        else:
            tasktype_filter = "individual"
            
        agent_data = data[(data['agent'] == agent_id) & 
                         (data['group'] == group_id) &
                         (data['taskType'] == tasktype_filter)].copy()
        
        if len(agent_data) == 0:
            print(f"  No {tasktype_filter} data found for Agent {agent_id}, Group {group_id}")
            continue
            
        # Sort by round and trial
        agent_data = agent_data.sort_values(['round', 'trial'])
        
        # Process each round separately
        for round_num in agent_data['round'].unique():
            round_data = agent_data[agent_data['round'] == round_num].copy()
            
            # Initialize eps_soc for this round
            current_eps_soc = mean_initial_eps_soc
            cumulative_reward = 0.0
            
            for idx, row in round_data.iterrows():
                trial = row['trial']
                reward = row['reward'] - 0.5  # Normalize as in fitting (mean=0)
                is_random = row['isRandom']
                
                # Update cumulative reward
                cumulative_reward += reward
                
                # Record current state
                trajectory_row = {
                    'round': round_num,
                    'trial': trial,
                    'current_eps_soc': current_eps_soc,
                    'reward': reward,
                    'cumulative_reward': cumulative_reward,
                    'is_random': is_random,
                    'agent': agent_id,
                    'group': group_id,
                    'initial_eps_soc': mean_initial_eps_soc,
                    'eta_eps_soc': mean_eta_eps_soc
                }
                trajectories.append(trajectory_row)
                
                # Update eps_soc for next trial (ASG adaptation rule)
                # Only adapt after trial 0 and for non-random trials
                if trial > 0 and is_random == 0:
                    # ASG rule: current_eps_soc = max(0.001, current_eps_soc - eta_eps_soc * reward)
                    reward_adjustment = mean_eta_eps_soc * reward
                    current_eps_soc = max(0.001, current_eps_soc - reward_adjustment)
    
    return pd.DataFrame(trajectories)

def load_asg_fitting_results(condition="group"):
    """Load ASG fitting results from the appropriate directory"""
    
    if condition == "group":
        pattern = "./Data/fitting_data/ASG_group/ASG_pars_ASG_group_*.npy"
    else:
        pattern = "./Data/fitting_data/ASG_solo/ASG_pars_solo_*.npy"
    
    print(f"Looking for {condition} fitting files: {pattern}")
    
    all_params = []
    files = glob(pattern)
    
    if len(files) == 0:
        print(f"No {condition} fitting files found!")
        return []
    
    for file in files:
        print(f"Loading {file}")
        try:
            params = np.load(file, allow_pickle=True)
            all_params.extend(params)
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    print(f"Loaded {len(all_params)} parameter sets for {condition} condition")
    return all_params

def main():
    """Main function to generate ASG dynamics CSV files"""
    
    # Load experimental data
    print("Loading experimental data...")
    try:
        data = pd.read_csv("../Data/e2_data.csv")
        print(f"Loaded {len(data)} rows of experimental data")
    except Exception as e:
        print(f"Error loading experimental data: {e}")
        return
    
    # Create output directory
    output_dir = "./Data/ASG_dynamics_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    # Process both conditions
    conditions = ["group", "solo"]
    
    for condition in conditions:
        print(f"\n{'='*50}")
        print(f"Processing {condition.upper()} condition")
        print(f"{'='*50}")
        
        # Load ASG fitting results for this condition
        asg_params = load_asg_fitting_results(condition)
        
        if len(asg_params) == 0:
            print(f"No ASG parameters found for {condition} condition, skipping...")
            continue
        
        # Reconstruct dynamics
        dynamics_df = reconstruct_asg_dynamics(data, asg_params, condition)
        
        if len(dynamics_df) == 0:
            print(f"No dynamics generated for {condition} condition")
            continue
        
        # Save results
        output_file = f"{output_dir}/ASG_dynamics_{condition}.csv"
        dynamics_df.to_csv(output_file, index=False)
        print(f"\n Saved {len(dynamics_df)} trajectory points to {output_file}")
        
        # Show summary statistics
        print(f"\nSummary for {condition} condition:")
        print(f"  Agents: {len(dynamics_df['agent'].unique())}")
        print(f"  Groups: {sorted(dynamics_df['group'].unique())}")
        print(f"  Rounds: {sorted(dynamics_df['round'].unique())}")
        print(f"  Average initial_eps_soc: {dynamics_df['initial_eps_soc'].mean():.3f}")
        print(f"  Average eta_eps_soc: {dynamics_df['eta_eps_soc'].mean():.3f}")
        print(f"  eps_soc range: {dynamics_df['current_eps_soc'].min():.3f} - {dynamics_df['current_eps_soc'].max():.3f}")
    
    print(f"Files saved in: {output_dir}/")

if __name__ == "__main__":
    main()