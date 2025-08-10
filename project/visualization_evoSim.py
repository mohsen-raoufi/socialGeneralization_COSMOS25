# -*- coding: utf-8 -*-
"""
Analyze ASG evolutionary simulation results
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def analyze_evolution_file(filepath, description):
    """Analyze a single evolutionary simulation result."""

    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return None
        
    try:
        # Load evolution data
        evolution = np.load(filepath, allow_pickle=True)
        
        # Analyze initial generation (generation 0)
        initial_gen = evolution[0]
        print(f"Initial population size: {len(initial_gen)}")
        
        # Count model types in initial generation
        initial_counts = count_models(initial_gen)
        print("Initial model distribution:")
        for model, count in initial_counts.items():
            print(f"  {model}: {count} agents")
        
        # Analyze final generation 
        final_gen = evolution[-1]
        final_counts = count_models(final_gen)
        print("Final model distribution:")
        for model, count in final_counts.items():
            print(f"  {model}: {count} agents")
            
        # Track evolution over time
        evolution_data = []
        for gen_idx, generation in enumerate(evolution[::50]):  # Sample every 50 generations
            gen_counts = count_models(generation)
            gen_counts['generation'] = gen_idx * 50
            evolution_data.append(gen_counts)
        
        evolution_df = pd.DataFrame(evolution_data).fillna(0)
        
        # Analyze ASG parameters if ASG agents exist
        asg_agents = [agent for agent in final_gen if agent[0].get('initial_eps_soc', 0) > 0]
        if asg_agents:
            print(f"\nASG Parameter Analysis ({len(asg_agents)} ASG agents in final generation):")
            initial_eps_values = [agent[0]['initial_eps_soc'] for agent in asg_agents]
            eta_eps_values = [agent[0]['eta_eps_soc'] for agent in asg_agents]
            
            print(f"initial_eps_soc: mean={np.mean(initial_eps_values):.3f}, std={np.std(initial_eps_values):.3f}")
            print(f"eta_eps_soc: mean={np.mean(eta_eps_values):.5f}, std={np.std(eta_eps_values):.5f}")
            print(f"Range initial_eps_soc: [{min(initial_eps_values):.3f}, {max(initial_eps_values):.3f}]")
            print(f"Range eta_eps_soc: [{min(eta_eps_values):.5f}, {max(eta_eps_values):.5f}]")
        
        return {
            'initial_counts': initial_counts,
            'final_counts': final_counts, 
            'evolution_df': evolution_df,
            'asg_params': asg_agents[:10] if asg_agents else []  # First 10 for analysis
        }
        
    except Exception as e:
        print(f"Error analyzing {filepath}: {e}")
        return None

def count_models(generation):
    """Count different model types in a generation."""
    counts = {}
    
    for agent in generation:
        params = agent[0]
        
        # Identify model type based on non-zero parameters
        if params.get('initial_eps_soc', 0) > 0:
            model_type = 'ASG'
        elif params.get('eps_soc', 0) > 0:
            model_type = 'SG'
        elif params.get('alpha', 0) > 0:
            model_type = 'VS'
        elif params.get('gamma', 0) > 0:
            model_type = 'DB'
        elif params.get('dummy', 0) > 0:
            model_type = 'Dummy'
        else:
            model_type = 'AS'
            
        counts[model_type] = counts.get(model_type, 0) + 1
    
    return counts

def plot_evolution_trajectories(results_dict):
    """Plot model evolution over time."""
    plt.figure(figsize=(15, 10))
    
    subplot_idx = 1
    for name, results in results_dict.items():
        if results is None:
            continue
            
        plt.subplot(2, 2, subplot_idx)
        evolution_df = results['evolution_df']
        
        # Plot each model type
        model_colors = {'ASG': 'red', 'SG': 'blue', 'AS': 'green', 'DB': 'orange', 'VS': 'purple', 'Dummy': 'gray'}
        
        for model in model_colors.keys():
            if model in evolution_df.columns:
                plt.plot(evolution_df['generation'], evolution_df[model], 
                        color=model_colors[model], label=model, linewidth=2)
        
        plt.title(f'Evolution Trajectory: {name}')
        plt.xlabel('Generation')
        plt.ylabel('Number of Agents')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        subplot_idx += 1
    
    plt.tight_layout()
    plt.savefig('./Data/evoSims/visualization_evoSim.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    print("ASG Evolutionary Analysis")
    print("=" * 50)
    
    # Define files to analyze
    evolution_files = {
        'ASG Only (150)': './Data/evoSims/corr09/ASG_c09_150.npy',
        'SG vs ASG (190)': './Data/evoSims/corr09/SG.ASG_c09_190.npy', 
        'All Models (300)': './Data/evoSims/corr09/AS.DB.VS.SG.ASG_c09_300.npy'
    }
    
    # Analyze each simulation
    results = {}
    for name, filepath in evolution_files.items():
        results[name] = analyze_evolution_file(filepath, name)
    
    # Generate summary
    print("\n" + "=" * 50)
    print("EVOLUTIONARY ANALYSIS SUMMARY")
    
    # Check ASG performance
    for name, result in results.items():
        if result is None:
            continue
            
        print(f"\n{name}:")
        initial = result['initial_counts']
        final = result['final_counts']
        
        # Calculate changes
        for model in set(list(initial.keys()) + list(final.keys())):
            initial_count = initial.get(model, 0)
            final_count = final.get(model, 0)
            change = final_count - initial_count
            
            if change > 0:
                print(f"{model}: {initial_count} → {final_count} (+{change})")
            elif change < 0:
                print(f"{model}: {initial_count} → {final_count} ({change})")
            else:
                print(f"{model}: {initial_count} → {final_count} (no change)")
    
    # Key findings
    print(f"\nKEY FINDINGS:")
    
    # Check if ASG dominated any simulations
    asg_dominated = False
    for name, result in results.items():
        if result and result['final_counts'].get('ASG', 0) > 50:
            print(f" ASG dominated in: {name}")
            asg_dominated = True
    
    if not asg_dominated:
        print(f" ASG did not achieve dominance in any simulation")
        
    # Check SG vs ASG direct competition
    if results.get('SG vs ASG (190)'):
        sg_asg_result = results['SG vs ASG (190)']
        final_sg = sg_asg_result['final_counts'].get('SG', 0)
        final_asg = sg_asg_result['final_counts'].get('ASG', 0)
        
        if final_asg > final_sg:
            print(f" ASG beat SG in direct competition ({final_asg} vs {final_sg})")
        else:
            print(f" SG beat ASG in direct competition ({final_sg} vs {final_asg})")
    
    # Plot evolution trajectories
    try:
        plot_evolution_trajectories(results)
        print(f" Evolution plots saved to: ./Data/evolution_analysis_trajectories.png")
    except Exception as e:
        print(f" Could not generate plots: {e}")
    
    print(f"\n Evolutionary analysis complete!")
    
    return results

if __name__ == "__main__":
    results = main()