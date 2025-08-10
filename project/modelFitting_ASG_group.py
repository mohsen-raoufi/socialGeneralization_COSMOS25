# -*- coding: utf-8 -*-
"""
ASG Model Fitting Script for GROUP rounds
modified following original Alex structure
"""
import os
import sys

import numpy as np
import pandas as pd
import scipy.optimize as opt

import modelFit as mf

# Get data
data = pd.read_csv("../Data/e2_data.csv") 
fits = pd.concat([data['agent'], data['group']], axis=1)
fits = fits.drop_duplicates()
fits["ASG_fit"] = np.nan

# Storage for ASG only
ASG_pars = []
nLL_ASG = []

# For cluster: what group is being recovered
g = int(sys.argv[1])  # Use group number directly, not as array index
np.random.seed(12345+g)

# Subset to said group + task type
subdata = data.loc[(data['group']==g)]
subdata = subdata.loc[(subdata["taskType"]=="social")]
# Mean needs to be 0 for GP, is .5 in data
subdata.loc[:,"reward"] = subdata["reward"]-0.5 

shor = len(np.unique(subdata["trial"]))
rounds = len(np.unique(subdata["round"]))

# Create output directory
path = "./Data/fitting_data/ASG_group/"
if not os.path.exists(path):
    os.makedirs(path, exist_ok=True)

print(f"Starting ASG-only fitting for group {g}")
print(f"Group {g}: {len(np.unique(subdata['agent']))} agents, {rounds} rounds, {shor} trials per round")

# Goes through agents of the given group
for ag in np.unique(subdata['agent']):
    fit_asg = np.zeros(rounds)
    r = 0
    
    # Separates one round out as test for crossvalidation
    for test in np.unique(subdata["round"]):
        tardata = subdata.loc[(subdata['agent']==ag) &
                              (subdata['round']!=test),
                              ['round','trial','choice','reward','isRandom']]
        tardata = tardata.to_numpy()
        socdata = subdata.loc[(subdata['agent']!=ag) &
                              (subdata['round']!=test),
                              ['round','trial','choice','reward','isRandom','agent']]
        socdata = socdata.to_numpy()
        testtar = subdata.loc[(subdata['agent']==ag) &
                              (subdata['round']==test),
                              ['round','trial','choice','reward','isRandom']]
        testtar = testtar.to_numpy()
        testsoc = subdata.loc[(subdata['agent']!=ag) &
                              (subdata['round']==test),
                              ['round','trial','choice','reward','isRandom','agent']]
        testsoc = testsoc.to_numpy()
        
        # Fit ASG model only (model index 4)
        print(f"Group {g}, Agent {ag}, ASG model, Round {r}")
        # Parameters: lambda, beta, tau, initial_eps_soc, eta_eps_soc
        # Bounds: initial_eps_soc similar to SG eps_soc, eta_eps_soc small positive values
        pars = opt.differential_evolution(mf.model_fit,
                                        [(-5,3),(-5,3),(-7.5,3),(-5,np.log(19)),(-6,-1)],
                                        (4, tardata, socdata, shor),
                                        maxiter=100)['x']
        fit_asg[r] = mf.model_fit(pars, 4, testtar, testsoc, shor)
        nLL_ASG.append(fit_asg[r])
        pars = np.append([ag, g], pars)
        ASG_pars.append([pars])
        
        r += 1
        print(f"Round {r} completed")
    
    # Store fit for this agent
    fits.loc[(fits['group']==g) & (fits["agent"]==ag), "ASG_fit"] = np.sum(fit_asg)
    fits.to_csv(path + f"model_recov_ASG_group_{g}.csv")
    print(f"Agent {ag} completed. ASG fit: {np.sum(fit_asg):.2f}")

# Save results
np.save(path + f"ASG_pars_ASG_group_{g}.npy", ASG_pars)
np.save(path + f"nLL_ASG_ASG_group_{g}.npy", nLL_ASG)

print(f"Group {g} ASG-only fitting completed!")
print(f"Results saved to: {path}")
print(f"Total ASG parameters fitted: {len(ASG_pars)}")
print(f"Total ASG nLL values: {len(nLL_ASG)}")