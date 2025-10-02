# -*- coding: utf-8 -*-
"""
following original Alex structure
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import invwishart

def ucb(pred,beta=0.5): 
    """    
    Parameters
    ----------
    pred : tuple of arrays in shape (2,n,1) 
        mean and variance values for n options (GPR prediction output).
    beta : numerical, optional
        governs exploration tendency. The default is 0.5.

    Returns
    -------
    Upper confidence bound of the data.
    """
    out = pred[0] + beta * np.sqrt(pred[1]) 
    return(out)

def rbf(x,y,l):
    D = cdist(x,y)**2
    return(np.exp(-D/(2*l**2)))

def GPR(Xstar,obs,rewards,socIndex, pars, baseEpsilon, k = rbf,vs_boost={}):
    """
    Parameters
    ----------
    Xstar : list
        The space we project to. Maps observation indices to coordinates as well.
    obs : list
        Indices of agent's observation history.
    rewards : list
        Agent's reward history.
    socIndex : list
        Relevant for SG model; 1 if observation is social, 0 if individual.
    pars : dict
        Model parameters.
    k : function, optional
        Kernel function. The default is rbf.
    vs_boost : dictionary, optional
        Priors for VS with memory. The default is {}.

    Returns
    -------
    List of posterior mean and variance.

    """
    choices = np.array([Xstar[x] for x in obs])
    K = k(choices,choices,pars['lambda']) #covariance of observations
    #add noise (social noise gets added to social observations)
    # For ASG agents, use current_eps_soc (adaptive), otherwise use eps_soc (fixed)
    social_noise = pars.get('current_eps_soc', pars['eps_soc'])
    noise = np.ones_like(socIndex)*baseEpsilon + socIndex*social_noise
    KK = K+np.identity(len(noise))*noise
    KK_inv = np.linalg.inv(KK)
    Ky = np.dot(KK_inv,rewards)
    Kstarstar = k(Xstar,Xstar,pars['lambda']) #covariance of Xstar with itself
    Kstar = k(Xstar,choices,pars['lambda'])
    mu = np.dot(Kstar,Ky) #get mean
    cov = Kstarstar - np.dot(np.dot(Kstar,KK_inv),Kstar.T)#covariance
    var = np.diagonal(cov).copy() #and variance; if I don't copy, var isn't writeable, which breaks for SG
    return([mu,var])

def model_sim(allParams, envList, rounds, shor, baseEpsilon=.0001, debug = False, memory=False,payoff=True,prior_mean=0.5,prior_scale=1):
    """
    Simulates experimental data based on input specifications
    
    Parameters
    ----------
    allParams : list of list of dictionaries
        List of agent's parameter values. Number of groups is specified based on this.
    envList : list
        Reward environments.
    rounds : int
        Number of rounds (and thus environments used).
    shor : int
        Search horizon (number of datapoints per round).
    baseEpsilon : numeric, optional
        Assumed observation noise for individual observations. The default is .0001.
    debug : Boolean, optional
        Should GP posterior, value function and policy be plotted when running. The default is False.
    memory : Boolean, optional
        Should VS use info from all previous trials rather than t-1. The default is False.
    payoff : Boolean, optional
        Should VS use outcome information. The default is True.

    Returns
    -------
    A pandas dataframe with the simulated data (specific agents, environments, scores and parameters).
    """
    #set up lists to collect output info
    agent = []
    group = []
    r0und = [] 
    env = []
    trial = []
    choice = []
    reward = []
    lambda_ind = []
    dummy = []
    beta = []
    tau = []
    gamma = []
    alpha = []
    eps_soc = []
    initial_eps_soc = []
    eta_eps_soc = []
    #simulation parameters
    nAgents = len(allParams[0])
    gridSize = len(envList[0][0])
    Xstar = np.array([(x, y) for x in range(np.sqrt(gridSize).astype(int)) for y in range(np.sqrt(gridSize).astype(int))])
    for g in range(len(allParams)): #iterate over groups
        #get parameters and set up collectors for value and policy
        pars = allParams[g]
        vals = np.zeros((gridSize,nAgents))
        policy = vals = np.zeros((gridSize,nAgents))
        #sample random set of environments
        envs = np.random.randint(0,len(envList[0]),(rounds)) #change here to use set envs
        for r in range(rounds): 
            #in every round, reset observations and rewards
            X = np.zeros((shor,nAgents)).astype(int)
            Y = np.zeros((shor,nAgents))
            # Initialize current_eps_soc for ASG agents at the beginning of each round
            for ag in range(nAgents):
                if 'initial_eps_soc' in pars[ag] and pars[ag]['initial_eps_soc'] > 0:
                    pars[ag]['current_eps_soc'] = pars[ag]['initial_eps_soc']
            for t in range(shor):
                #on each trial, reset prediction
                prediction = []
                if t==0:
                    #First choice is random
                    X[0,:] = np.random.randint(0,gridSize,(nAgents))
                    Y[0,:] = [(envList[ag][envs[r]][X[0,ag]]['payoff']-prior_mean)/prior_scale + np.random.normal(0,.01) for ag in range(nAgents)] 
                else:
                    for ag in range(nAgents):
                        obs = X[:t,ag] #self observations
                        rewards = Y[:t,ag] #self rewards
                        socIndex = np.zeros_like(obs) #individual info
                        #social generalization uses social obs in GP
                        if (pars[ag]['eps_soc']>0) or (pars[ag]["dummy"]!=0):
                            obs = np.append(obs, X[:t,np.arange(nAgents)!=ag]) #social observations
                            rewards = np.append(rewards, Y[:t,np.arange(nAgents)!=ag]) #social rewards
                            if pars[ag]["dummy"]!=0:
                                socIndex = np.zeros_like(obs) #dummy model treats everything the same
                            else:
                                socIndex = np.append(socIndex, np.ones_like(X[:t,np.arange(nAgents)!=ag])) #otherwise flag as social observations
                       
                        # For ASG model, use current_eps_soc if available, otherwise use eps_soc
                        current_pars = pars[ag].copy()
                        if 'current_eps_soc' in pars[ag]:
                            current_pars['eps_soc'] = pars[ag]['current_eps_soc']
                        prediction.append(GPR(Xstar,obs,rewards,socIndex,current_pars,baseEpsilon)) 
                        #values from UCB
                        vals[:,ag] = np.squeeze(ucb(prediction[ag],pars[ag]['beta']))
                        #count occurrences of social choices in the previous round
                        bias = [(i,np.sum(X[t-1,np.arange(nAgents)!=ag]==i),
                                 np.mean(Y[t-1,np.arange(nAgents)!=ag][np.nonzero(X[t-1,np.arange(nAgents)!=ag]==i)])) #mean of Y at t-1 for each unique X at t-1
                                for i in np.unique(X[t-1,np.arange(nAgents)!=ag])]  
                        
                        #Value shaping
                        if pars[ag]['alpha']>0: #prediction error learning from individual value vs. soc info
                            vs_boost = [(i,np.sum(X[:t,np.arange(nAgents)!=ag]==i),np.mean(Y[X==i])) for i in np.unique(X[:t,np.arange(nAgents)!=ag])] 
                            #vs_boost = [(X[np.where(Y==i)],i) for i in np.max(Y[:t,np.arange(nAgents)!=ag],axis=0)]
                            for b in vs_boost:
                                #vals[b[0],ag] += pars[ag]['alpha']*b[1]*(b[2]-rew_exp)
                                vals[b[0],ag] = vals[b[0],ag] + pars[ag]['alpha']*(b[1]*b[2]-vals[b[0],ag]) # np.mean(rewards)

                        #avoid overflow
                        vals[:,ag] = vals[:,ag]-np.max(vals[:,ag]) 
                        #Policy
                        policy[:,ag] = np.exp(vals[:,ag]/pars[ag]['tau'])
                        #Decision biasing
                        if pars[ag]['gamma']>0:
                                #payoff bias (generally unused)
                                rew_exp=np.mean(rewards) #individual reward experience
                                socChoices = np.ones(gridSize)*.000000000001  #just zeros breaks policy sometimes
                                if payoff:
                                    for b in bias:
                                        #subtracting from policy overcomplicates things; only boost helpful info
                                        if b[2]-rew_exp<0:
                                            continue
                                        #social policy proportional to how much better than experience social option is
                                        socChoices[b[0]] += b[1]*(b[2]-rew_exp)
                                else:
                                    for b in bias:
                                        socChoices[b[0]] += b[1]
                                socpolicy = socChoices/sum(socChoices)
                                #mixture policy
                                policy[:,ag] = ((1 - pars[ag]['gamma'])*policy[:,ag]) + (pars[ag]['gamma'] * socpolicy)
                    #Choice
                    policy = policy/np.sum(policy,axis=0) 
                    X[t,:] = [np.random.choice(gridSize, p = policy[:,ag]) for ag in range(nAgents)]
                    # Store original rewards for ASG adaptation before normalization
                    original_rewards = [envList[ag][envs[r]][X[t,ag]]['payoff'] + np.random.normal(0,.01) for ag in range(nAgents)]
                    Y[t,:] = [(original_rewards[ag] - prior_mean)/prior_scale for ag in range(nAgents)]
                    
                    # ASG adaptation: Update current_eps_soc based on ORIGINAL reward for ASG agents
                    for ag in range(nAgents):
                        if 'current_eps_soc' in pars[ag] and 'eta_eps_soc' in pars[ag] and t > 0:
                            # Adapt eps_soc based on original reward (0-1 scale)
                            # Higher reward -> lower eps_soc (more trust in social info)  
                            # Formula: current_eps_soc = max(0.001, current_eps_soc - eta_eps_soc * (reward - 0.5))
                            # This way, rewards > 0.5 decrease eps_soc (more trust), rewards < 0.5 increase eps_soc (less trust)
                            reward_adjustment = pars[ag]['eta_eps_soc'] * (original_rewards[ag] - 0.5)
                            pars[ag]['current_eps_soc'] = max(0.001, pars[ag]['current_eps_soc'] - reward_adjustment)
            for ag in range(nAgents):
                #collect information
                agent.append(np.ones((shor,1))*ag)
                group.append(np.ones((shor,1))*g)
                r0und.append(np.ones((shor,1))*r)
                env.append(np.ones((shor,1))*envs[r])
                trial.append(np.arange(shor))
                choice.append(X[:,ag])
                reward.append(Y[:,ag])
                lambda_ind.append(np.ones((shor,1))*pars[ag]['lambda'])
                beta.append(np.ones((shor,1))*pars[ag]['beta'])
                tau.append(np.ones((shor,1))*pars[ag]['tau'])
                gamma.append(np.ones((shor,1))*pars[ag]['gamma'])
                alpha.append(np.ones((shor,1))*pars[ag]['alpha'])
                eps_soc.append(np.ones((shor,1))*pars[ag]['eps_soc'])
                dummy.append(np.ones((shor,1))*pars[ag]["dummy"])
                initial_eps_soc.append(np.ones((shor,1))*pars[ag]['initial_eps_soc'])
                eta_eps_soc.append(np.ones((shor,1))*pars[ag]['eta_eps_soc'])
    #format dataset
    data = np.column_stack((np.concatenate(agent),np.concatenate(group),np.concatenate(r0und),np.concatenate(env),np.concatenate(trial),
            np.concatenate(choice),np.concatenate(reward),np.concatenate(lambda_ind),
            np.concatenate(beta),np.concatenate(tau),np.concatenate(gamma),np.concatenate(alpha),np.concatenate(eps_soc),np.concatenate(dummy),
            np.concatenate(initial_eps_soc),np.concatenate(eta_eps_soc)))
    agentData = pd.DataFrame(data,columns=('agent','group','round','env','trial','choice','reward','lambda','beta',
                                           'tau','gamma','alpha','eps_soc',"dummy",'initial_eps_soc','eta_eps_soc'))      
    return(agentData)
    
def param_gen(nAgents,nGroups,models=None):
    """
    Generates parameter sets for simulations

    Parameters
    ----------
    nAgents : int
        Number of agents per group. Since we have 4 environments per set, the default is 4.
    nGroups : int
        Number of groups.
    models : int, list, or None, optional
        Model specification:
        - int (0-5): All agents in all groups use this model (homogeneous)
        - list of length nAgents: Each agent in each group uses the model at that index (heterogeneous)
        - None: Random model assignment for each agent
        0=AS,1=DB,2=VS,3=SG,4=ASG,5=dummy model. The default is None.
        
    Returns
    -------
    List of list of dictionaries as used in model_sim.

    """
    par_names = ["lambda","beta","tau","gamma","alpha","eps_soc","dummy","initial_eps_soc","eta_eps_soc"] #needed for dicts later #
    all_pars = []
    
    for g in range(nGroups):
        all_pars.append([])
        par = np.zeros((nAgents,len(par_names)))
        par[:,0] = np.random.lognormal(-0.75,0.5,(nAgents)) #lambda
        par[:,1] = np.random.lognormal(-0.75,0.5,(nAgents))
        par[:,2] = np.random.lognormal(-4.5,0.9,(nAgents))  #tau
        
        # Determine model assignment for this group
        if models is None:
            # Random assignment for each agent
            model_assignment = np.random.randint(0,6,(nAgents))
        elif isinstance(models, int):
            # Homogeneous: all agents use the same model
            assert models in range(0,6), "Model has to be between 0 and 5"
            model_assignment = np.ones(nAgents) * models
        elif isinstance(models, (list, np.ndarray)):
            # Heterogeneous: each agent gets its own model
            assert len(models) == nAgents, f"models list must have length {nAgents}"
            assert all(m in range(0,6) for m in models), "All models must be between 0 and 5"
            model_assignment = np.array(models)
        else:
            raise ValueError("models must be int, list/array, or None")
        
        # Set model-specific parameters
        for ag in range(nAgents):
            model = model_assignment[ag]
            if model==1:
                par[ag,3] = np.random.uniform(0,1,(1)) #gamma
            elif model==2:
                par[ag,4] = np.random.uniform(0,1,(1))
            elif model==3:
                par[ag,5] = np.random.exponential(2,(1)) #eps_soc
            elif model == 4: # ASG model
                par[ag,7] = np.random.exponential(2,(1)) #initial_eps_soc
                par[ag,8] = np.random.lognormal(-3,0.5,(1)) #eta_eps_soc
            elif model == 5:
                par[ag,6] = 1 #dummy variable for a model test
            
            pars = dict(zip(par_names,par[ag,:]))
            all_pars[g].append(pars)
    
    return(all_pars)

def param_gen_pilot(nAgents,nGroups,hom=True,models=None):
    """
    Generates parameter sets for simulations based on parameter distribution from pilot

    Parameters
    ----------
    nAgents : int
        Number of agents per group. Since we have 4 environments per set, the default is 4.
    nGroups : int
        Number of groups.
    hom : boolean, optional
        Should the groups be homogenous. The default is True.
    models : int in range(6), optional
        If specified, homogenous groups will only have the input model. 
        0=AS,1=DB,2=VS,3=SG,4=ASG,5=dummy model. The default is None.
        
    Returns
    -------
    List of list of dictionaries as used in model_sim.

    """
    par_names = ["lambda","beta","tau","gamma","alpha","eps_soc","dummy","initial_eps_soc","eta_eps_soc"] #needed for dicts later #
    all_pars = []
    if hom:
        #randomly select model unless given as an argument
        if models is None:
            models = np.random.randint(0,6,nGroups)
        else:
            assert models in range(0,6), "Model has to be between 0 and 5"
            models = np.ones(nGroups)*models
        for g in range(nGroups):
            model = models[g]
            all_pars.append([]) #set up list for parameter dictionary
            par = np.zeros((nAgents,len(par_names)))
            par[:,0] = np.random.lognormal(0.1,0.1,(nAgents)) #lambda
            par[:,1] = np.random.lognormal(-1.25,0.2,(nAgents))
            #par[:,1] = np.zeros(nAgents)*0.1#beta
            par[:,2] = np.random.lognormal(-4.5,0.1,(nAgents))  #tau
            if model==1:
                par[:,3] = np.random.lognormal(-3.5,1,(nAgents)) #gamma previously 1/14
            elif model==2:
                par[:,4] = np.random.lognormal(-3.5,1,(nAgents))
            elif model==3:
                par[:,5] = np.random.lognormal(3.6,0.3,(nAgents)) #eps_soc #formerly 3,0.5 for dist_grids
            elif model == 4:
                par[:,6] = np.ones(nAgents) #dummy variable for a model test
            elif model == 5: # ASG model
                par[:,7] = np.random.lognormal(3.6,0.3,(nAgents)) #initial_eps_soc (similar to SG)
                par[:,8] = np.random.lognormal(-3,0.5,(nAgents)) #eta_eps_soc (small positive values)
            for ag in range(nAgents):
                pars = dict(zip(par_names,par[ag,:]))
                all_pars[g].append(pars)
    else:
        for g in range(nGroups):
            all_pars.append([])
            model = np.random.randint(0,6,(nAgents))
            par = np.zeros((nAgents,len(par_names)))
            par[:,0] = np.random.lognormal(0.1,0.1,(nAgents)) #lambda
            #par[:,0] = np.ones((nAgents))*2
            par[:,1] = np.random.lognormal(-1.25,0.2,(nAgents)) #beta
            par[:,2] = np.random.lognormal(-4.5,0.1,(nAgents))  #tau
            for ag in range(nAgents):
                if model[ag]==1:
                    while par[ag,3]<0.143 or par[ag,3]>1:
                        #par[ag,3] = np.random.lognormal(-3.5,1,(1))     #gamma
                        par[ag,3] = np.random.uniform(0,1,(1))
                elif model[ag]==2:
                    while par[ag,4]<0.116:
                    #par[ag,4] = np.random.exponential(6.5,(1))     #alpha minmax 0.2; VS+ has 6.5 mean
                        #par[ag,4] = np.random.lognormal(-3.5,1,(1))
                        par[ag,4] = np.random.uniform(0,1,(1))
                elif model[ag]==3:
                    while par[ag,5]==0 or par[ag,5]>19:
                      par[ag,5] = np.random.lognormal(3.6,0.3,(1)) #eps_soc
                elif model[ag]==4:
                    par[ag,6] = 1 #dummy variable for a model test
                elif model[ag]==5: # ASG model
                    par[ag,7] = np.random.lognormal(3.6,0.3,(1)) #initial_eps_soc
                    par[ag,8] = np.random.lognormal(-3,0.5,(1)) #eta_eps_soc
                pars = dict(zip(par_names,par[ag,:]))
                all_pars[g].append(pars)
    return(all_pars)


#for evolutionary simulations
def pop_gen(popSize, models):
    """
    Generates populations of size popSize and including
    strategies in models for evolutionary simulations.

    Parameters
    ----------
    popSize : int
        Population size.
    models : list of int range(6)
        Specifies models desired in the population.
        0=AS,1=DB,2=VS,3=SG,4=ASG,5=dummy model

    Returns
    -------
    List of list of dict.

    """
    pop = []
    subpop = popSize//len(models) #get even proportion of desired models
    for i in range(6):  # Now handle all 6 models
        if i in models:
            pop.extend(param_gen(1,subpop,models=i)) 
    while len(pop)<popSize: #if even proportion can't result in correct population size, populate with random model from list of desired
        pop.extend(param_gen_pilot(1,1,models = np.random.choice(models)))
    return(pop)


#for evolutionary simulations
def pop_gen_pilot(popSize, models):
    """
    Generates populations of size popSize and including
    strategies in models for evolutionary simulations.

    Parameters
    ----------
    popSize : int
        Population size.
    models : list of int range(4)
        Specifies models desired in the population.
        0=AS,1=DB,2=VS,3=SG

    Returns
    -------
    List of list of dict.

    """
    pop = []
    subpop = popSize//len(models) #get even proportion of desired models
    for i in range(4):
        if i in models:
            pop.extend(param_gen_pilot(1,subpop,models=i)) 
    while len(pop)<popSize: #if even proportion can't result in correct population size, populate with random model from list of desired
        pop.extend(param_gen_pilot(1,1,models = np.random.choice(models)))
    return(pop)
    
#simulate roger's paradox
def roger(nAgents,nGroups,socModel,nSoc):
    par_names = ["lambda","beta","tau","gamma","alpha","eps_soc","dummy"] #needed for dicts later
    all_pars = []
    model = np.zeros(nAgents)
    model[:nSoc] = socModel
    for g in range(nGroups):
        all_pars.append([])
        par = np.zeros((nAgents,len(par_names)))
        for ag in range(nAgents):
            if model[ag]==0:
                par[ag,0] = np.random.lognormal(0.55,0.2) #lambda
                par[ag,1] = np.random.lognormal(-1.45,0.2) #beta
                par[ag,2] = np.random.lognormal(-5.5,0.9)  #tau
            elif model[ag]==1:
                par[ag,0] = np.random.lognormal(0.55,0.2) #lambda
                par[ag,1] = np.random.lognormal(-1.6,0.2) #beta
                par[ag,2] = np.random.lognormal(-5.5,0.9) #tau
                par[ag,3] = np.random.lognormal(-2.25,0.4)    #gamma
            elif model[ag]==2:
                par[ag,0] = np.random.lognormal(0.4,0.15) #lambda
                par[ag,1] = np.random.lognormal(-1.9,0.4) #beta
                par[ag,2] = np.random.lognormal(-5.3,0.9) #tau
                par[ag,4] = np.random.lognormal(-2.2,0.4) #alpha
            else:
                par[ag,0] = np.random.lognormal(0.7,0.2) #lambda
                par[ag,1] = np.random.lognormal(-1.7,0.2) #beta
                par[ag,2] = np.random.lognormal(-5.5,0.9)  #tau
                par[ag,5] = np.random.lognormal(1.09,0.35) #eps_soc
            pars = dict(zip(par_names,par[ag,:]))
            all_pars[g].append(pars)
    return(all_pars)

def estimate_correlation_bayesian(focal_gpr_output, partner_rewards, partner_locations, 
                                 kappa0=1e-3, m0=None, nu0=4, S0=None, w_max=100):
    """
    Estimates correlation between focal agent's GPR predictions and partner agent's rewards
    using Bayesian Normal-Inverse-Wishart prior.
    
    Parameters
    ----------
    focal_gpr_output : tuple
        (mean, variance) from GPR for all locations. mean and variance are arrays of length n_locations.
    partner_rewards : array-like
        Observed rewards for partner agent at each trial.
    partner_locations : array-like  
        Location indices visited by partner agent at each trial.
    kappa0 : float, optional
        Prior strength on mean (pseudo-sample size). Default is 1e-3 (weak prior).
    m0 : array-like, optional
        Prior mean vector. Default is [0, 0].
    nu0 : float, optional
        Prior degrees of freedom for covariance. Must be > 3. Default is 4.
    S0 : array-like, optional
        Prior scale matrix. Default is identity matrix.
    w_max : float, optional
        Maximum weight to avoid extreme values. Default is 100.
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'correlation': estimated correlation coefficient
        - 'covariance_matrix': posterior mean covariance matrix
        - 'mean_vector': posterior mean vector
        - 'n_observations': number of observations used
        - 'posterior_params': (kappa, m, nu, S) for further updates
    """
    
    # Extract GPR outputs
    focal_means, focal_variances = focal_gpr_output
    
    # Convert to numpy arrays
    partner_rewards = np.array(partner_rewards)
    partner_locations = np.array(partner_locations, dtype=int)
    focal_means = np.array(focal_means)
    focal_variances = np.array(focal_variances)
    
    # Validate inputs
    if len(partner_rewards) != len(partner_locations):
        raise ValueError("partner_rewards and partner_locations must have same length")
    
    if len(focal_means) != len(focal_variances):
        raise ValueError("focal_means and focal_variances must have same length")
    
    # Set default prior parameters
    if m0 is None:
        m0 = np.array([0.0, 0.0])
    else:
        m0 = np.array(m0)
    
    if S0 is None:
        S0 = np.eye(2)
    else:
        S0 = np.array(S0)
    
    # Initialize posterior parameters with prior
    kappa = kappa0
    m = m0.copy()
    nu = nu0
    S = S0.copy()
    
    # Process each observation
    n_obs = len(partner_rewards)
    
    for t in range(n_obs):
        # Get focal's prediction at partner's location
        if partner_locations[t] >= len(focal_means):
            continue  # Skip if location index is out of bounds
            
        x_t = focal_means[partner_locations[t]]  # Focal's posterior mean
        y_t = partner_rewards[t]  # Partner's observed reward
        
        # Create observation vector
        z_t = np.array([x_t, y_t])
        
        # Calculate weight based on focal's uncertainty
        sigma_focal = np.sqrt(focal_variances[partner_locations[t]])
        if sigma_focal > 0:
            w = min(1.0 / (sigma_focal**2), w_max)
        else:
            w = 1.0  # Default weight if variance is zero
        
        # Update parameters using NIW update equations
        kappa_new = kappa + w
        m_new = (kappa * m + w * z_t) / kappa_new
        nu_new = nu + w
        S_new = S + (kappa * w / kappa_new) * np.outer(z_t - m, z_t - m)
        
        # Update for next iteration
        kappa = kappa_new
        m = m_new
        nu = nu_new
        S = S_new
    
    # Extract posterior mean covariance matrix
    if nu > 3:
        Sigma_bar = S / (nu - 3)
    else:
        # If nu <= 3, posterior mean doesn't exist, use mode instead
        Sigma_bar = S / (nu + 1)
    
    # Calculate correlation
    if Sigma_bar[0, 0] > 0 and Sigma_bar[1, 1] > 0:
        correlation = Sigma_bar[0, 1] / np.sqrt(Sigma_bar[0, 0] * Sigma_bar[1, 1])
    else:
        correlation = 0.0
    
    return {
        'correlation': correlation,
        'covariance_matrix': Sigma_bar,
        'mean_vector': m,
        'n_observations': n_obs,
        'posterior_params': (kappa, m, nu, S)
    }

def get_social_index(focal_agent_idx, all_agents_data, focal_gpr_output, n_agents=4):
    """
    Estimates the social index vector δ_soc for a focal agent based on correlations
    with all partner agents using Bayesian correlation estimation.
    
    Parameters
    ----------
    focal_agent_idx : int
        Index of the focal agent (0-based).
    all_agents_data : dict
        Dictionary containing data for all agents with keys:
        - 'choices': array of shape (n_trials, n_agents) - location choices
        - 'rewards': array of shape (n_trials, n_agents) - observed rewards
    focal_gpr_output : tuple
        (mean, variance) from GPR for all locations for the focal agent.
    n_agents : int, optional
        Total number of agents. Default is 4.
        
    Returns
    -------
    numpy.ndarray
        Social index vector δ_soc of length n_agents where:
        - δ_soc[0] = 0 (focal agent)
        - δ_soc[j] = 1 - ρ₁ⱼ² for j ∈ {1, 2, 3} (partners)
        where ρ₁ⱼ is the correlation between focal and partner j
    """
    
    # Initialize social index vector
    delta_soc = np.zeros(n_agents)
    
    # Extract data
    choices = all_agents_data['choices']
    rewards = all_agents_data['rewards']
    n_trials = len(choices)
    
    # For each partner agent (excluding focal)
    for partner_idx in range(n_agents):
        if partner_idx == focal_agent_idx:
            delta_soc[partner_idx] = 0.0  # Focal agent has no social index
            continue
            
        # Get partner's choices and rewards
        partner_choices = choices[:, partner_idx]
        partner_rewards = rewards[:, partner_idx]
        
        # Estimate correlation between focal and this partner
        correlation_result = estimate_correlation_bayesian(
            focal_gpr_output=focal_gpr_output,
            partner_rewards=partner_rewards,
            partner_locations=partner_choices
        )
        
        # Extract correlation coefficient
        rho_1j = correlation_result['correlation']
        
        # Calculate δ_soc(j) = 1 - ρ₁ⱼ²
        # This represents the "unexplained variance" - how much that partner inflates noise
        delta_soc[partner_idx] = 1.0 - (rho_1j ** 2)
        
        # Ensure the value is in [0, 1] range
        delta_soc[partner_idx] = np.clip(delta_soc[partner_idx], 0.0, 1.0)
    
    return delta_soc

def get_adaptive_social_noise(delta_soc_vector, base_eps_soc=1.0, max_eps_soc=10.0):
    """
    Converts social index vector to adaptive social noise values.
    
    Parameters
    ----------
    delta_soc_vector : numpy.ndarray
        Social index vector δ_soc for one agent.
    base_eps_soc : float, optional
        Base social noise parameter. Default is 1.0.
    max_eps_soc : float, optional
        Maximum social noise parameter. Default is 10.0.
        
    Returns
    -------
    numpy.ndarray
        Adaptive social noise values for each partner.
    """
    
    # Convert social index to noise inflation factors
    # Higher δ_soc (lower correlation) → higher noise
    # Lower δ_soc (higher correlation) → lower noise
    adaptive_noise = base_eps_soc * (1.0 + delta_soc_vector)
    
    # Clip to reasonable range
    adaptive_noise = np.clip(adaptive_noise, base_eps_soc, max_eps_soc)
    
    return adaptive_noise

def test_social_index_estimation():
    """
    Test function to demonstrate the social index estimation functions.
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create example data
    n_agents = 4
    n_trials = 20
    grid_size = 25
    
    # Generate example choices and rewards
    choices = np.random.randint(0, grid_size, (n_trials, n_agents))
    rewards = np.random.normal(0, 1, (n_trials, n_agents))
    
    # Create some correlation between agents 0 and 1
    rewards[:, 1] = 0.7 * rewards[:, 0] + 0.3 * np.random.normal(0, 1, n_trials)
    
    # Create some correlation between agents 0 and 2 (negative)
    rewards[:, 2] = -0.5 * rewards[:, 0] + 0.5 * np.random.normal(0, 1, n_trials)
    
    # Agent 3 is independent
    rewards[:, 3] = np.random.normal(0, 1, n_trials)
    
    # Create example GPR outputs (mean and variance for all locations)
    focal_means = np.random.normal(0, 1, grid_size)
    focal_variances = np.random.exponential(0.1, grid_size)
    focal_gpr_output = (focal_means, focal_variances)
    
    # Create data dictionary
    all_agents_data = {
        'choices': choices,
        'rewards': rewards
    }
    
    # Test social index estimation for agent 0
    print("Testing social index estimation for agent 0:")
    print(f"Choices shape: {choices.shape}")
    print(f"Rewards shape: {rewards.shape}")
    
    delta_soc = get_social_index(
        focal_agent_idx=0,
        all_agents_data=all_agents_data,
        focal_gpr_output=focal_gpr_output,
        n_agents=n_agents
    )
    
    print(f"Social index vector δ_soc: {delta_soc}")
    print(f"Interpretation:")
    print(f"  - δ_soc[0] = {delta_soc[0]:.3f} (focal agent)")
    print(f"  - δ_soc[1] = {delta_soc[1]:.3f} (partner 1 - should be low due to positive correlation)")
    print(f"  - δ_soc[2] = {delta_soc[2]:.3f} (partner 2 - should be low due to negative correlation)")
    print(f"  - δ_soc[3] = {delta_soc[3]:.3f} (partner 3 - should be high due to independence)")
    
    # Test adaptive social noise
    adaptive_noise = get_adaptive_social_noise(delta_soc, base_eps_soc=1.0, max_eps_soc=5.0)
    print(f"\nAdaptive social noise: {adaptive_noise}")
    
    return delta_soc, adaptive_noise

def test_correlation_learning_model():
    """
    Test function to demonstrate the new Correlation Learning model.
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create a simple test with 4 agents, 1 group, 1 round, 10 trials
    nAgents = 4
    nGroups = 1
    rounds = 1
    shor = 10
    
    # Create simple environment
    gridSize = 9  # 3x3 grid
    # Create multiple environments for each agent
    nEnvs = 3
    envList = [[[{'payoff': np.random.uniform(0, 1)} for _ in range(gridSize)] for _ in range(nEnvs)] for _ in range(nAgents)]
    
    # Generate parameters: 1 CL agent (model 6), 3 AS agents (model 0)
    models = [6, 0, 0, 0]  # Agent 0 is CL, agents 1-3 are AS
    allParams = param_gen(nAgents, nGroups, models=models)
    
    print("Testing Correlation Learning model:")
    print(f"Agent 0 (CL) parameters: {allParams[0][0]}")
    print(f"Agent 1 (AS) parameters: {allParams[0][1]}")
    print(f"Agent 2 (AS) parameters: {allParams[0][2]}")
    print(f"Agent 3 (AS) parameters: {allParams[0][3]}")
    
    # Run simulation
    try:
        result = model_sim(allParams, envList, rounds, shor)
        print(f"Simulation successful! Result shape: {result.shape}")
        print(f"Columns: {result.columns.tolist()}")
        
        # Show some results
        print("\nFirst few rows of results:")
        print(result.head())
        
        return result
    except Exception as e:
        print(f"Simulation failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None

    