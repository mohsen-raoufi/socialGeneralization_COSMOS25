# -*- coding: utf-8 -*-
"""
5-Model Complete Synthesis Script (AS, DB, VS, SG, ASG)
Based on original fitting_synthesis_e2.py structure
Combines original 4 models with ASG

modified following original Alex structure
"""

import os
import numpy as np
import pandas as pd
import re
from fnmatch import fnmatch

def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(data, key=alphanum_key)

def synthesize_5_models(condition_name, original_path, asg_path, output_suffix):
    """
    Synthesize results from original 4 models + ASG for proper 5-model comparison
    
    Parameters:
    -----------
    condition_name : str
        "Individual" or "Social"
    original_path : str
        Path to original 4-model results
    asg_path : str
        Path to ASG results  
    output_suffix : str
        Suffix for output files
    """
    
    print(f"\n{'='*60}")
    print(f"5-MODEL SYNTHESIS: {condition_name.upper()} CONDITION")
    print(f"{'='*60}")
    print(f"Original data: {original_path}")
    print(f"ASG data: {asg_path}")
    
    # Check paths exist
    if not os.path.exists(original_path):
        print(f"Original path does not exist: {original_path}")
        return
    if not os.path.exists(asg_path):
        print(f"ASG path does not exist: {asg_path}")
        return
    
    # Load participant structure from original data
    data = pd.read_csv("../Data/e2_data.csv")
    
    if condition_name == "Individual":
        condition_data = data[data['taskType'] == 'individual']
    else:
        condition_data = data[data['taskType'] == 'social']
    
    fits = pd.concat([condition_data['agent'], condition_data['group']], axis=1)
    fits = fits.drop_duplicates()
    fits = fits.reset_index(drop=True)
    groupSize = len(fits)
    
    print(f"Participants: {groupSize}")
    
    # ========================================
    # LOAD ORIGINAL 4 MODELS (AS, DB, VS, SG)
    # ========================================
    
    print(f"\n--- Loading Original 4 Models ---")
    
    # Get overall fits (CSV files)
    files = [original_path + "/" + file for file in os.listdir(original_path) if fnmatch(file, 'model*.csv')]
    files = sorted_alphanumeric(files)
    
    if len(files) == 0:
        print(f"No original model CSV files found in {original_path}")
        return
    
    fit = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    fit = fit.dropna()
    fit = fit.reset_index(drop=True)
    
    print(f"Original models loaded: {len(fit)} participants")
    
    # Load original model nLLs
    models = ['AS', 'DB', 'VS', 'SG']
    model_nLLs = {}
    
    for model in models:
        nLL_files = [file for file in os.listdir(original_path) if fnmatch(file, f'nLL_{model}*.npy')]
        nLL_files = sorted_alphanumeric(nLL_files)
        
        if len(nLL_files) == 0:
            print(f"No {model} nLL files found")
            return
        
        nLL = np.squeeze(np.array([np.load(original_path + "/" + file, allow_pickle=True) for file in nLL_files]))
        nLL = nLL.flatten()
        nLL = np.split(nLL, groupSize)
        nLL = np.squeeze(nLL)
        model_nLLs[model] = nLL
        
        print(f"{model} nLL loaded: shape {nLL.shape}")
    
    # ========================================
    # LOAD ASG MODEL
    # ========================================
    
    print(f"\n--- Loading ASG Model ---")
    
    # Load ASG nLLs
    ASG_nLL_files = [file for file in os.listdir(asg_path) if fnmatch(file, 'nLL_ASG*.npy')]
    ASG_nLL_files = sorted_alphanumeric(ASG_nLL_files)
    
    if len(ASG_nLL_files) == 0:
        print(f"No ASG nLL files found in {asg_path}")
        return
    
    ASG_nLL = np.squeeze(np.array([np.load(asg_path + "/" + file, allow_pickle=True) for file in ASG_nLL_files]))
    ASG_nLL = ASG_nLL.flatten()
    ASG_nLL = np.split(ASG_nLL, groupSize)
    ASG_nLL = np.squeeze(ASG_nLL)
    model_nLLs['ASG'] = ASG_nLL
    
    print(f"ASG nLL loaded: shape {ASG_nLL.shape}")
    
    # ========================================
    # CALCULATE R² FOR ALL MODELS
    # ========================================
    
    print(f"\n--- Calculating R² ---")
    
    # Calculate random baseline
    if condition_name == "Individual":
        randomCounts = data.loc[(data["taskType"] == "individual")].groupby(['agent', "group", "round"]).sum("isRandom")
    else:
        randomCounts = data.loc[(data["taskType"] == "social")].groupby(['agent', "group", "round"]).sum("isRandom")
    
    randomCounts = randomCounts["isRandom"].to_numpy().reshape(8, groupSize)
    randnLL = -np.log(1/121) * (15 - randomCounts)  # 15 trials per round
    
    # Calculate R² for all models
    for model in models + ['ASG']:
        r2 = 1 - (model_nLLs[model] / randnLL.T)
        r2 = np.mean(r2, axis=1)
        fit[f"r2_{model}"] = r2
        print(f"{model} R² calculated. Mean: {np.mean(r2):.4f}")
    
    # ========================================
    # LOAD PARAMETERS FOR ALL MODELS
    # ========================================
    
    print(f"\n--- Loading Parameters ---")
    
    # Original models parameters
    model_params = {}
    param_columns = {
        'AS': ["agent", "group", "lambda", "beta", "tau"],
        'DB': ["agent", "group", "lambda", "beta", "tau", "par"],
        'VS': ["agent", "group", "lambda", "beta", "tau", "par"], 
        'SG': ["agent", "group", "lambda", "beta", "tau", "par"]
    }
    
    for model in models:
        param_files = [file for file in os.listdir(original_path) if fnmatch(file, f'{model}_pars*.npy')]
        param_files = sorted_alphanumeric(param_files)
        
        if len(param_files) == 0:
            print(f"No {model} parameter files found")
            return
        
        params = np.concatenate(np.squeeze(np.array([np.load(original_path + "/" + file, allow_pickle=True) for file in param_files])))
        
        # Exponentiate parameters (saved as logged)
        params[:, 2:] = np.exp(params[:, 2:])
        
        # Average across rounds
        mean_params = np.mean(np.split(params, len(fit)), axis=1)
        mean_params_df = pd.DataFrame(mean_params, columns=param_columns[model])
        mean_params_df["model"] = model
        model_params[model] = mean_params_df
        
        print(f"{model} parameters loaded: shape {mean_params_df.shape}")
    
    # ASG parameters
    ASG_param_files = [file for file in os.listdir(asg_path) if fnmatch(file, 'ASG_pars*.npy')]
    ASG_param_files = sorted_alphanumeric(ASG_param_files)
    
    if len(ASG_param_files) == 0:
        print(f"No ASG parameter files found")
        return
    
    ASG_params = np.concatenate(np.squeeze(np.array([np.load(asg_path + "/" + file, allow_pickle=True) for file in ASG_param_files])))
    ASG_params[:, 2:] = np.exp(ASG_params[:, 2:])
    
    mean_ASG_params = np.mean(np.split(ASG_params, len(fit)), axis=1)
    mean_ASG_params_df = pd.DataFrame(mean_ASG_params, columns=["agent", "group", "lambda", "beta", "tau", "initial_eps_soc", "eta_eps_soc"])
    mean_ASG_params_df["model"] = "ASG"
    model_params['ASG'] = mean_ASG_params_df
    
    print(f"ASG parameters loaded: shape {mean_ASG_params_df.shape}")
    
    # ========================================
    # CREATE COMBINED RESULTS
    # ========================================
    
    print(f"\n--- Creating Combined Results ---")
    
    # Add nLL columns to fit dataframe
    for model in models + ['ASG']:
        fit[f"fit_{model}"] = np.sum(model_nLLs[model], axis=1)
    
    # Create wide-to-long format
    fit['id'] = range(len(fit))
    parfits = pd.wide_to_long(fit, "r2", ["agent", "group"], "model", "_", '\\w+')
    parfits = parfits.reset_index()
    
    # Merge with parameters
    all_params = pd.concat(list(model_params.values()), ignore_index=True)
    parfits = pd.merge(parfits, all_params, on=['agent', 'group', 'model'], how='left')
    parfits.to_csv(f"./Data/fit+pars_5models_{output_suffix}.csv", index=False)
    
    # ========================================
    # DETERMINE BEST MODEL PER PARTICIPANT
    # ========================================
    
    print(f"\n--- Determining Best Models ---")
    
    # Initialize parameter columns
    fit['model'] = fit['lambda'] = fit['beta'] = fit['tau'] = np.nan
    fit['par'] = fit['initial_eps_soc'] = fit['eta_eps_soc'] = np.nan
    
    for i in range(len(fit)):
        model_fits = fit.loc[i, ['fit_AS', 'fit_DB', 'fit_VS', 'fit_SG', 'fit_ASG']]
        best_model_idx = np.argmin(model_fits)
        best_model = ['AS', 'DB', 'VS', 'SG', 'ASG'][best_model_idx]
        
        fit.loc[i, "model"] = best_model
        
        # Get parameters for best model
        match_pars = model_params[best_model].loc[(model_params[best_model]["agent"] == fit.loc[i, "agent"]) & 
                                                 (model_params[best_model]["group"] == fit.loc[i, "group"])]
        
        if len(match_pars) > 0:
            # Set common parameters
            fit.loc[i, ["lambda", "beta", "tau"]] = match_pars[["lambda", "beta", "tau"]].values[0]
            
            # Set model-specific parameters
            if best_model in ['DB', 'VS', 'SG']:
                fit.loc[i, "par"] = match_pars["par"].values[0]
            elif best_model == 'ASG':
                fit.loc[i, ["initial_eps_soc", "eta_eps_soc"]] = match_pars[["initial_eps_soc", "eta_eps_soc"]].values[0]
    
    fit.to_csv(f"./Data/model_fits_5models_{output_suffix}.csv", index=False)
    
    # Show model selection summary
    print(f"\nModel Selection Summary:")
    print(fit['model'].value_counts())
    
    print(f"\nMean Model Fits (nLL, lower = better):")
    for model in ['AS', 'DB', 'VS', 'SG', 'ASG']:
        mean_fit = fit[f'fit_{model}'].mean()
        print(f"{model}: {mean_fit:.2f}")
    
    # ========================================
    # COMPUTE PROTECTED EXCEEDANCE PROBABILITY
    # ========================================
    
    print(f"\n--- Computing Protected Exceedance Probability ---")
    
    try:
        from groupBMC.groupBMC import GroupBMC
        
        # Create LL matrix (5 models × N participants)
        LL = -fit[['fit_AS', 'fit_DB', 'fit_VS', 'fit_SG', 'fit_ASG']].to_numpy().T
        
        # Run GroupBMC
        result = GroupBMC(LL).get_result().protected_exceedance_probability
        
        # Save results
        pxp_data = {'model': ['AS', 'DB', 'VS', 'SG', 'ASG'], 'exceedance': result}
        result_df = pd.DataFrame(pxp_data)
        result_df.to_csv(f"./Data/pxp_5models_{output_suffix}.csv", index=False)
        
        print(f"\n=== PROTECTED EXCEEDANCE PROBABILITIES ===")
        for model, pxp in zip(['AS', 'DB', 'VS', 'SG', 'ASG'], result):
            print(f"{model}: {pxp:.6f}")
        
        best_model = ['AS', 'DB', 'VS', 'SG', 'ASG'][np.argmax(result)]
        best_pxp = np.max(result)
        print(f"\nBest Model: {best_model} (PXP = {best_pxp:.6f})")
        
    except ImportError:
        print("GroupBMC not installed")
        return
    except Exception as e:
        print(f"Error computing PXP: {e}")
        return
    
    print(f"\n{condition_name} condition 5-model synthesis completed!")
    return result_df

def main():
    """Main function to run 5-model synthesis for both conditions"""
    
    print("5-MODEL COMPLETE SYNTHESIS")
    print("Combining Original 4 Models (AS, DB, VS, SG) + ASG")
    print("Using proper GroupBMC for hierarchical Bayesian model comparison")
    
    # Define paths
    conditions = [
        {
            'name': 'Individual',
            'original_path': '../Data/fitting_data/e2_ind',
            'asg_path': './Data/fitting_data/ASG_solo',
            'suffix': 'individual'
        },
        {
            'name': 'Social', 
            'original_path': '../Data/fitting_data/e2_soc',
            'asg_path': './Data/fitting_data/ASG_group',
            'suffix': 'social'
        }
    ]
    
    results = {}
    
    for condition in conditions:
        try:
            result = synthesize_5_models(
                condition['name'], 
                condition['original_path'], 
                condition['asg_path'], 
                condition['suffix']
            )
            results[condition['name']] = result
        except Exception as e:
            print(f"Error processing {condition['name']}: {e}")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"5-MODEL SYNTHESIS COMPLETE!")
    print(f"{'='*60}")
    
    print(f"\nGenerated Files:")
    print(f"- model_fits_5models_individual.csv")
    print(f"- model_fits_5models_social.csv")
    print(f"- fit+pars_5models_individual.csv") 
    print(f"- fit+pars_5models_social.csv")
    print(f"- pxp_5models_individual.csv")
    print(f"- pxp_5models_social.csv")
    
    if results:
        print(f"\nSUMMARY RESULTS:")
        for condition_name, result_df in results.items():
            if result_df is not None:
                print(f"\n{condition_name} Condition:")
                best_idx = result_df['exceedance'].argmax()
                best_model = result_df.loc[best_idx, 'model']
                best_pxp = result_df.loc[best_idx, 'exceedance']
                print(f"Winner: {best_model} (PXP = {best_pxp:.6f})")
                
                asg_pxp = result_df[result_df['model'] == 'ASG']['exceedance'].values[0]
                print(f"ASG Performance: PXP = {asg_pxp:.6f}")

if __name__ == "__main__":
    main()