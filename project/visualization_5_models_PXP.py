# -*- coding: utf-8 -*-
"""
5-Model Results Visualization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality plots
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 10

def load_results():
    """Load PXP and model fits results"""
    
    # Load PXP results
    pxp_individual = pd.read_csv("./Data/pxp_5models_individual.csv")
    pxp_social = pd.read_csv("./Data/pxp_5models_social.csv")
    
    # Load model fits
    fits_individual = pd.read_csv("./Data/model_fits_5models_individual.csv")
    fits_social = pd.read_csv("./Data/model_fits_5models_social.csv")
    
    return pxp_individual, pxp_social, fits_individual, fits_social

def create_pxp_comparison_plot(pxp_individual, pxp_social):
    """Create main PXP comparison plot"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    models = ['AS', 'DB', 'VS', 'SG', 'ASG']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  # ASG in purple
    
    # Individual condition
    pxp_ind_values = pxp_individual['exceedance'].values
    bars1 = ax1.bar(models, pxp_ind_values, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
    
    # Highlight the winner
    winner_idx = np.argmax(pxp_ind_values)
    bars1[winner_idx].set_alpha(1.0)
    bars1[winner_idx].set_edgecolor('gold')
    bars1[winner_idx].set_linewidth(3)
    
    ax1.set_title('Individual Condition\n(AS Dominates as Expected)', fontweight='bold', fontsize=14)
    ax1.set_ylabel('Protected Exceedance Probability', fontweight='bold')
    ax1.set_ylim(0, 1.1)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars1, pxp_ind_values)):
        if val > 0.01:
            ax1.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.3f}', 
                    ha='center', va='bottom', fontweight='bold')
        else:
            ax1.text(bar.get_x() + bar.get_width()/2, 0.05, f'{val:.1e}', 
                    ha='center', va='bottom', fontsize=9, rotation=90)
    
    # Social condition  
    pxp_soc_values = pxp_social['exceedance'].values
    bars2 = ax2.bar(models, pxp_soc_values, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
    
    # Highlight the winner (ASG)
    winner_idx = np.argmax(pxp_soc_values)
    bars2[winner_idx].set_alpha(1.0)
    bars2[winner_idx].set_edgecolor('gold')
    bars2[winner_idx].set_linewidth(3)
    
    ax2.set_title('Social Condition\n(ASG Wins!)', fontweight='bold', fontsize=14)
    ax2.set_ylabel('Protected Exceedance Probability', fontweight='bold')
    ax2.set_ylim(0, 0.4)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars2, pxp_soc_values):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 0.01, f'{val:.3f}', 
                ha='center', va='bottom', fontweight='bold')
    
    # Remove significance threshold line and legend as requested
    
    plt.suptitle('5-Model Comparison: Protected Exceedance Probability\nAS vs DB vs VS vs SG vs ASG', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    return fig

def create_summary_table():
    """Create summary statistics table and save as CSV"""
    
    # Load the PXP results
    pxp_ind = pd.read_csv("./Data/pxp_5models_individual.csv")
    pxp_soc = pd.read_csv("./Data/pxp_5models_social.csv")
    
    # Create summary
    summary_data = {
        'Model': ['AS', 'DB', 'VS', 'SG', 'ASG'],
        'Individual_PXP': pxp_ind['exceedance'].values,
        'Social_PXP': pxp_soc['exceedance'].values,
        'Individual_Rank': pxp_ind['exceedance'].rank(ascending=False).astype(int).values,
        'Social_Rank': pxp_soc['exceedance'].rank(ascending=False).astype(int).values
    }
    
    summary_df = pd.DataFrame(summary_data)
    
    # Add formatted columns for better readability
    summary_df['Individual_PXP_Formatted'] = summary_df['Individual_PXP'].apply(
        lambda x: f"{x:.6f}" if x > 0.001 else f"{x:.2e}")
    summary_df['Social_PXP_Formatted'] = summary_df['Social_PXP'].apply(
        lambda x: f"{x:.6f}")
    
    # Add winner indicators
    summary_df['Individual_Winner'] = summary_df['Individual_Rank'] == 1
    summary_df['Social_Winner'] = summary_df['Social_Rank'] == 1
    
    return summary_df

def main():
    """Generate only the requested visualizations"""
    
    print("Loading 5-model comparison results...")
    pxp_individual, pxp_social, fits_individual, fits_social = load_results()
    
    print(f"Data loaded:")
    print(f"  Individual condition: {len(fits_individual)} participants")
    print(f"  Social condition: {len(fits_social)} participants")
    
    # Create output directory
    import os
    os.makedirs('./Data/visualization_5_models_PXP', exist_ok=True)
    
    # Generate only the requested outputs
    print("\nGenerating visualizations...")
    
    # 1. PXP comparison plot (without dash line and legend)
    fig1 = create_pxp_comparison_plot(pxp_individual, pxp_social)
    fig1.savefig('./Data/visualization_5_models_PXP/pxp_comparison.png', bbox_inches='tight', dpi=300)
    plt.close(fig1)  # Close to free memory
    print("PXP comparison plot saved")
    
    # 2. Summary table as CSV (instead of PNG)
    summary_df = create_summary_table()
    summary_df.to_csv('./Data/visualization_5_models_PXP/summary_table.csv', index=False)
    print("Summary table saved as CSV")
    
    # Print key results
    print(f"\n{'='*60}")
    print("KEY RESULTS SUMMARY")
    print(f"{'='*60}")
    
    print(f"\n INDIVIDUAL CONDITION:")
    ind_winner = pxp_individual.loc[pxp_individual['exceedance'].idxmax()]
    print(f"   Winner: {ind_winner['model']} (PXP = {ind_winner['exceedance']:.6f})")
    
    print(f"\n SOCIAL CONDITION:")
    soc_winner = pxp_social.loc[pxp_social['exceedance'].idxmax()]
    print(f"   Winner: {soc_winner['model']} (PXP = {soc_winner['exceedance']:.6f})")
    
    asg_soc_pxp = pxp_social[pxp_social['model'] == 'ASG']['exceedance'].values[0]
    sg_soc_pxp = pxp_social[pxp_social['model'] == 'SG']['exceedance'].values[0]
    print(f"\n ASG vs SG in Social Condition:")
    print(f"   ASG: {asg_soc_pxp:.6f} vs SG: {sg_soc_pxp:.6f}")
    print(f"   ASG advantage: {(asg_soc_pxp/sg_soc_pxp - 1)*100:+.1f}%")
    
    print(f"\nGenerated files:")
    print(f"   ./Data/visualization_5_models_PXP/pxp_comparison.png")
    print(f"   ./Data/visualization_5_models_PXP/summary_table.csv")

if __name__ == "__main__":
    main()