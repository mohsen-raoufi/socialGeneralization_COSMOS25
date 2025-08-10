# Adaptive Social Generalization (ASG) Model Extension Project

## 1. Project Overview

### **Introduciton**

This project is based on the research paper, "Humans flexibly integrate social information despite interindividual differences in reward" by Witt, Toyokawa, Lala, Gaissmaier, and Wu (2024). The **Social Generalization (SG) model** introduced in this paper provides a powerful framework for understanding how individuals integrate social information into their learning process. The model assumes a fixed social skepticism parameter, $\epsilon_{soc}$, which modulates the variance of an individual's value function. This is represented by the equation:

$$\sigma_{\epsilon|x}^{2}=\sigma_{\epsilon_{ind}}^{2}+\delta_{soc}(x)\cdot\sigma_{\epsilon_{soc}}^{2}$$

where the total variance ($\sigma_{\epsilon|x}^{2}$) is a function of both individual variance ($\sigma_{\epsilon_{ind}}^{2}$) and social variance ($\sigma_{\epsilon_{soc}}^{2}$).

However, since $\epsilon_{soc}$ represents a participant's skepticism towards social information and in the original model it is a fixed value, this project introduces and analyzes the **Adaptive Social Generalization (ASG) model** to expand on this concept. The ASG model hypothesizes that humans are adaptive learners who continuously adjust their trust in social information over time, based on the rewards they receive. The core of this adaptation is the dynamic update rule for the skepticism parameter:

$$\epsilon_{soc,t} = \max(0, \epsilon_{soc,t-1} - \eta_{\epsilon_{soc}} \cdot r_{obs})$$

where $\eta_{\epsilon_{soc}}$ is the adaptation rate and $r_{soc}(t)$ is the social reward. A decrease in $\epsilon_{soc}$ represents an increase in trust. This updated $\epsilon_{soc}$ value is then used in the main value update equation to modulate the influence of the social prediction error. The $max(0, ...)$ function ensures that skepticism never drops below zero. A negative value for $\epsilon_{soc}$​ would be nonsensical in this context. It would imply a level of "anti-skepticism" beyond complete trust, which is not a valid interpretation of the parameter. The max(0, ...) function acts as a lower bound, preventing the parameter from entering an uninterpretable range.

By making $\epsilon_{soc}$ a dynamic parameter, it is possible that the ASG model provides a more flexible and more realistic representation of this cognitive process. This adaptive nature is crucial for understanding how individuals learn to navigate and optimize their performance in a complex, ever-changing social world.


### **Research Question & Hypothesis**

* **Research Question:** How do individuals integrate and adapt to social information, and what is the relationship between a participant's social learning style and their asocial exploration strategies?

* **Hypothesis:** I hypothesize that participants will not rely on a single, static learning strategy. Instead, they will:
  1.  **Flexibly adjust their exploration-exploitation trade-off** by decreasing individual exploration when social information is available.
  2.  **Exhibit distinct "social learning personalities"** that can be identified by their unique trust trajectories and model parameters.



## 2. Project Steps

### Step 1: Model Development
> Implement ASG model with dynamic `eps_soc` parameter that adapts based on reward feedback

**Files Needed**:
```
project/
├── modelSim.py          # Core model implementation (modify to add ASG)
├── modelFit.py          # Model fitting functions (add ASG case)
```

**Output**: Modified core files with ASG model (index 4) integrated

---

### Step 2: Normative Testing
> Run evolutionary simulations to test if ASG agents outcompete existing models

**Files Needed**:
```
project/
├── evoSim.py            # Evolutionary simulation script (include ASG)
├── run_evolution.sh     # Batch execution script
```

**How to Run**:
```bash
cd project/
python evoSim.py 150       # ASG-only population
python evoSim.py 190       # ASG vs SG   
python evoSim.py 300       # All 5 models competing
```

**Output**: 
```
Data/evoSims/corr09     # Evolutionary simulation results
├── AS.DB.VS.SG.ASG_c09_300_scores.npy  
├── AS.DB.VS.SG.ASG_c09_300.npy         
├── ASG_c09_150_scores.npy              
├── ASG_c09_150.npy                     
├── SG.ASG_c09_190_scores.npy           
└── SG.ASG_c09_190.npy             
```

---

### Step 3: Human Data Fitting  
> Fit ASG model to experimental data and compare with original 4 models

**Files Needed**:
```
project/
├── modelFitting_ASG_group.py         # ASG model fitting (group conditions)
└── modelFitting_ASG_solo.py          # ASG model fitting (solo conditions)
```

**How to Run**:
```bash
cd project/

# Fit ASG model to group conditions (each group)
python modelFitting_ASG_group.py 0 

# Fit ASG model to solo conditions  (each group)
python modelFitting_ASG_solo.py 0

# Alternative: Use batch scripts
sbatch run_ASG_group.sh
sbatch run_ASG_solo.sh
```

**Output**:
```
Data/fitting_data/      
├── ASG_group/          # Group condition parameter estimates (.npy files)
├── ASG_solo/           # Individual condition parameter estimates  
└── logs/               # If run script, it'll produce fitting execution logs
    ├── logs_ASG_group/
    └── logs_ASG_solo/
```

---

### Step 4: Model Comparison
**Objective**: Use Protected Exceedance Probability (PXP) for hierarchical Bayesian model selection

**Files Needed**:
```
project/
├── fitting_synthesis_5_models.py       # Combine original 4 models + ASG, calculate PXP
├── visualization_5_models_PXP.py       # Generate PXP comparison plots and summary
├── boxplot_parameter_comparison.py     # Parameter comparison visualizations
└── visualization_trust_reward.py       # Trust-reward relationship analysis
```

**How to Run**:
```bash
cd project/

# Step 4.1: Synthesize all 5 models and calculate PXP
python fitting_synthesis_5_models.py

# Step 4.2: Create PXP comparison visualizations
python visualization_5_models_PXP.py

# Step 4.3: Generate parameter comparison boxplots
python boxplot_parameter_comparison.py

# Step 4.4: Analyze trust-reward relationships
python visualization_trust_reward.py
```

**Output**:
```
Data/
├── model_fits_5models_social.csv           # Social condition best model per participant
├── model_fits_5models_individual.csv       # Individual condition best model per participant
├── fit+pars_5models_social.csv             # Social condition all model parameters
├── fit+pars_5models_individual.csv         # Individual condition all model parameters
├── pxp_5models_social.csv                  # PXP results for social conditions
├── pxp_5models_individual.csv              # PXP results for individual conditions
├── visualization_5_models_PXP/             # Model comparison visualizations
│   ├── pxp_comparison.png                  # Main PXP comparison plot
│   └── summary_table.csv                   # Model performance summary
├── visualization_parameter_comparison/      # Parameter analysis plots
│   ├── group_vs_solo_parameters.png        # Side-by-side parameter boxplots
│   └── parameter_statistics.csv            # Statistical comparison results
└── visualization_trust_reward/             # Trust-reward analysis
    ├── agent_trust_change_vs_total_reward.png
    ├── trial_eps_change_vs_reward.png
    ├── eps_change_by_reward_quartile.png
    ├── trial_eps_changes.csv
    └── reward_quartile_stats.csv
```


---

### Step 5: Visualization & Analysis
**Objective**: Create comprehensive visualizations of ASG dynamics and trust adaptation patterns

**Files Needed**:
```
project/
├── ASG_dynamics_reconstructor.py              # Reconstruct trust trajectories from parameters
├── visualization_compare_group_vs_solo_trajectories.py  # Group vs solo comparison plots
├── visualization_raw_eps_soc_trajectories.py  # Raw trust evolution visualizations
└── visualization_normalized_trajectories.py   # Normalized trajectory comparisons
```

**How to Run**:
```bash
cd project/

# Step 5.1: Reconstruct ASG trust dynamics from fitted parameters
python ASG_dynamics_reconstructor.py

# Step 5.2: Create group vs solo trajectory comparisons (all groups)
python visualization_compare_group_vs_solo_trajectories.py

# Step 5.3: Generate raw trust trajectory plots
python visualization_raw_eps_soc_trajectories.py

# Step 5.4: Create normalized trajectory comparisons
python visualization_normalized_trajectories.py
```

**Output**:
```
Data/
├── ASG_dynamics_analysis/                      # Reconstructed trust dynamics
│   ├── ASG_dynamics_group.csv                 # Group condition trust trajectories
│   ├── ASG_dynamics_solo.csv                  # Solo condition trust trajectories
│   ├── trust_changes_summary.csv              # Summary of trust changes per agent
│   └── reconstruction_log.txt                 # Processing log
├── visualization_compare_group_vs_solo_trajectories/  # Group vs solo comparisons
│   ├── four_panel_comparison_group_*_vs_solo.png     # Per-group comparison plots
│   └── group_vs_solo_volatility_summary.csv          # Volatility comparison statistics
├── raw_eps_soc_comparison/                     # Raw trust trajectory plots
│   ├── group_vs_solo_raw_comparison.png       # Raw trust value comparisons
│   └── trajectory_statistics.csv              # Raw trajectory statistics
└── normalized_trajectories/                   # Normalized trust comparisons
    ├── normalized_trust_evolution.png         # Normalized trajectory plots
    └── adaptation_patterns.csv                # Trust adaptation pattern analysis
```