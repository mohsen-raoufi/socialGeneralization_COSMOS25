"""
Sanity check for the new model of the detla variable, based on the (perceived) similarity of the rewards by agents to the other agent.

Project 2 @ COSMOS 2025
Author(s): Mohsen Raoufi
"""

import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt


import matplotlib.patheffects as path_effects

grid_n = 11         # width of the grid
k = 16             # number of choices
agent_n = 4         # number of agents/environments

# agent_index_to_plot = 0
env_index_to_plot = 0
# or pick a random index from the list of environments
# env_index_to_plot = np.random.randint(0, len(envList))



def load_env(path = None):
    if path is None:
        path = ('environments_unequal' ) 
    json_files = [file for file in os.listdir(path) if file.endswith('_c01_unequal_demo.json')]
    # sort the files based on their names
    json_files.sort()
    print("Found JSON files:", json_files)
    envList = []
    for file in json_files:
        f=open(os.path.join(path, file))
        envList.append(json.load(f))
    return envList







def env_list_to_array(env_list, grid_n=11):
    arr = np.zeros((grid_n, grid_n), dtype=float)
    for d in env_list:
        i = int(d['x1']) - 1   # 1-indexed in your format
        j = int(d['x2']) - 1
        arr[i, j] = float(d['payoff'])
    return arr

def plot_env(arr, title="Environment", bool_show_vals=False, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6,5))
    ax.imshow(arr.T, origin='lower',
               extent=[0.5, arr.shape[0]+0.5, 0.5, arr.shape[1]+0.5])
    if bool_show_vals:
        # annotate values
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                val = arr[i, j]
                if not np.isnan(val):
                    ax.text(i+1, j+1, f"{val:.2f}",
                            ha="center", va="center",
                            color="white" if val < 0.5 else "black",
                            fontsize=8)
    ax.set_title(title)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_xticks(range(1, arr.shape[0]+1))
    ax.set_yticks(range(1, arr.shape[1]+1))
    # plt.colorbar(label="Payoff", ax=ax)
    # plt.show()

def plot_random_subset(env_list, grid_n=11, k=16, seed=None):
    arr = env_list_to_array(env_list, grid_n)
    rng = np.random.default_rng(seed)
    flat_idx = rng.choice(grid_n*grid_n, size=k, replace=False)
    mask = np.full(arr.shape, np.nan)
    for idx in flat_idx:
        i, j = np.unravel_index(idx, arr.shape)
        mask[i, j] = arr[i, j]
    plot_env(mask, title=f"{k} randomly picked tiles (others white)", bool_show_vals=True)


# plot the given subset from the agent's choices
def plot_agent_choice_subset(env_list, agent_choice_arr, ax = None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    arr = env_list_to_array(env_list, grid_n=11)
    mask = np.full(arr.shape, np.nan)
    for choice in agent_choice_arr:
        i = int(choice['x1']) - 1
        j = int(choice['x2']) - 1
        mask[i, j] = arr[i, j]
    plot_env(mask, title="Agent's chosen tiles (others white)", bool_show_vals=True, ax=ax)


# ## Similarity metrics
# 1. Position overlap (regardless of rewards): Just compare the chosen tile positions.
def jaccard_on_positions(idxA, idxB):
    """Jaccard index: overlap of chosen tiles, ignoring rewards."""
    sA, sB = set(idxA), set(idxB)
    inter = len(sA & sB)
    union = len(sA | sB)
    return 0.0 if union == 0 else inter / union


# 2. Overlap + rewards: Look at tiles both agents visited, and compare the rewards they received.
def reward_corr_on_overlap(idxA, rewA, idxB, rewB, grid_n=11):
    """Pearson correlation of rewards on overlapping tiles."""
    coordsA = {idx: r for idx, r in zip(idxA, rewA)}
    coordsB = {idx: r for idx, r in zip(idxB, rewB)}
    overlap = list(set(coordsA.keys()) & set(coordsB.keys()))
    if not overlap:
        return 0.0
    rA = np.array([coordsA[i] for i in overlap])
    rB = np.array([coordsB[i] for i in overlap])
    return np.corrcoef(rA, rB)[0, 1]

# 3. Spatial correlation (beyond exact overlap): Turn each agent’s choices into a reward map, then compare them globally.
def reward_map_corr(idxA, rewA, idxB, rewB, grid_n=11):
    """Correlation of reward maps across the whole grid."""
    arrA = np.zeros((grid_n, grid_n)); arrB = np.zeros((grid_n, grid_n))
    for idx, r in zip(idxA, rewA):
        i, j = np.unravel_index(idx, (grid_n, grid_n))
        arrA[i, j] = r
    for idx, r in zip(idxB, rewB):
        i, j = np.unravel_index(idx, (grid_n, grid_n))
        arrB[i, j] = r
    # flatten & correlate (ignores empty=0 cells)
    return np.corrcoef(arrA.ravel(), arrB.ravel())[0, 1]


# 4. Spatially tolerant kernel alignment (more advanced): Even if agents don’t pick the same tile, nearby choices still count.
def kernel_alignment_similarity(idxA, rewA, idxB, rewB, grid_n=11, sigma=1.5):
    """Kernel alignment similarity: rewards weighted by spatial closeness."""
    coordsA = np.array([np.unravel_index(idx, (grid_n, grid_n)) for idx in idxA])
    coordsB = np.array([np.unravel_index(idx, (grid_n, grid_n)) for idx in idxB])
    if len(coordsA)==0 or len(coordsB)==0:
        return 0.0
    d2 = ((coordsA[:,None,:] - coordsB[None,:,:])**2).sum(axis=2)
    K = np.exp(-d2 / (2*sigma**2))
    wA = np.array(rewA).reshape(-1,1)
    wB = np.array(rewB).reshape(1,-1)
    num = (K * (wA * wB)).sum()
    return num / (np.linalg.norm(wA)*np.linalg.norm(wB) + 1e-9)



from scipy.ndimage import gaussian_filter
import numpy as np

def smoothed_reward_map_corr(idxA, rewA, idxB, rewB, grid_n=11, sigma=1.5):
    """

    Parameters
    ----------
    idxA : list or array of int
        Flat indices of agent A's chosen tiles (0..grid_n*grid_n-1).
    rewA : list or array of float
        Rewards received by agent A at the chosen tiles.
    idxB : list or array of int
        Flat indices of agent B's chosen tiles.
    rewB : list or array of float
        Rewards received by agent B at the chosen tiles.
    grid_n : int, optional
        Size of one grid dimension (default 11).
    sigma : float, optional
        Standard deviation of the Gaussian smoothing kernel (default 1.5).

    Returns
    -------
    float
        Pearson correlation between the smoothed reward maps of A and B.
    """
    arrA = np.zeros((grid_n, grid_n))
    arrB = np.zeros((grid_n, grid_n))
    for idx, r in zip(idxA, rewA):
        i, j = np.unravel_index(idx, (grid_n, grid_n))
        arrA[i, j] = r
    for idx, r in zip(idxB, rewB):
        i, j = np.unravel_index(idx, (grid_n, grid_n))
        arrB[i, j] = r

    # smooth both maps
    SA = gaussian_filter(arrA, sigma=sigma, mode='nearest')
    SB = gaussian_filter(arrB, sigma=sigma, mode='nearest')

    # flatten & correlate
    return np.corrcoef(SA.ravel(), SB.ravel())[0, 1]





def plot_envs(env_index_to_plot, axs=None):
    if axs is None:
        fig, axs = plt.subplots(1, agent_n, figsize=(18, 5))
    for i in range(agent_n):
        arr = env_list_to_array(envList[i][env_index_to_plot], grid_n=11)
        plot_env(arr, title=f"Environment for Agent {i+1}", bool_show_vals=False, ax=axs[i])
    plt.tight_layout()
    plt.suptitle(f"Environment index {env_index_to_plot}")
    return axs


# plot the choices of all the agents in a single figure
def plot_choices_and_rewards(axs=None):
    if axs is None:
        fig, axs = plt.subplots(1, 4, figsize=(18, 5))
    for index_agent, (ax, choice_arr, title) in enumerate(zip(axs,
                                    [choice_arr_1, choice_arr_2, choice_arr_3, choice_arr_4],
                                    ["Agent 1", "Agent 2", "Agent 3", "Agent 4"])):
        choice_list = []
        for idx in choice_arr:
            i, j = np.unravel_index(idx, (grid_n, grid_n))
            choice_list.append({'x1': i+1, 'x2': j+1})  # convert to 1-indexed

        print(index_agent)
        envTemp = envList[index_agent][env_index_to_plot]
        plot_agent_choice_subset(envTemp, choice_list, ax=ax)
        ax.set_title(title)
    plt.tight_layout()
    plt.suptitle("Agents' Choices and Rewards")
    return axs
# plt.show()



## get the rewards for each agent given choices and the environment
def get_rewards(env, agent_choices):
    arr = env_list_to_array(env, grid_n=11)
    rewards = []
    for choice in agent_choices:
        i = int(choice['x1']) - 1
        j = int(choice['x2']) - 1
        rewards.append(arr[i, j])
    return np.array(rewards)


# make a subplot of 4 figures, and imshow the heatmap of similarity for each metric
def plot_similarity_matrices(similarity_matrix, agent_names, similarity_metrics, axs=None):
    if axs is None:
        fig, axs = plt.subplots(1, 4, figsize=(15, 5))
    for i, metric in enumerate(similarity_metrics):
        im = axs[i].imshow(similarity_matrix[metric], cmap='Blues', interpolation='nearest')
        axs[i].set_title(metric, fontsize=10)
        axs[i].set_xticks(np.arange(len(agent_names)))
        # rotate the xtick labels
        plt.setp(axs[i].get_xticklabels(), rotation=45, ha="right",
                rotation_mode="anchor")
        axs[i].set_yticks(np.arange(len(agent_names)))
        axs[i].set_xticklabels(agent_names)
        
        for (j, k), val in np.ndenumerate(similarity_matrix[metric]):
            txt = axs[i].text(k, j, f"{val:.2f}", ha="center", va="center", color="black", fontsize=8)
            txt.set_path_effects([
                path_effects.Stroke(linewidth=1, foreground="white"),
                path_effects.Normal()
            ])

        # remove the yticks labels
        axs[i].set_yticklabels([])
    # fig.colorbar(im, ax=axs, orientation='vertical')
    axs[0].set_yticklabels(agent_names)

    # set the colormap to magma

    plt.tight_layout()
    plt.suptitle("Similarity Matrices Between Agents")
    return axs


def calculate_similarity_metrics(agents_choices, agents_rewards):
    n_agents = len(agents_choices)
    similarity_metrics = ["Position Overlap", "Reward Corr on Overlap", "Spatial Reward Map Corr", "Kernel-based Spatial Similarity"]
    similarity_matrix = {metric: np.zeros((n_agents, n_agents)) for metric in similarity_metrics}
    for i in range(n_agents):
        for j in range(n_agents):
            if i == j:
                for metric in similarity_metrics:
                    similarity_matrix[metric][i, j] = 1.0
            else:
                sim_pos = jaccard_on_positions(agents_choices[i], agents_choices[j])
                sim_rew = reward_corr_on_overlap(agents_choices[i], agents_rewards[i],
                                                agents_choices[j], agents_rewards[j], grid_n=11)
                sim_field = reward_map_corr(agents_choices[i], agents_rewards[i],
                                           agents_choices[j], agents_rewards[j], grid_n=11)
                sim_kernel = kernel_alignment_similarity(agents_choices[i], agents_rewards[i],
                                                        agents_choices[j], agents_rewards[j],
                                                        grid_n=11, sigma=1.5)
                similarity_matrix["Position Overlap"][i, j] = sim_pos
                similarity_matrix["Reward Corr on Overlap"][i, j] = sim_rew
                similarity_matrix["Spatial Reward Map Corr"][i, j] = sim_field
                similarity_matrix["Kernel-based Spatial Similarity"][i, j] = sim_kernel
    return similarity_matrix, similarity_metrics




## make a default function to run if nothing is called from outside
if __name__ == "__main__":
    print("Running sanity_check_model.py as main...")
    ## load the environments
    envList = load_env()

    # make a random choice for agents
    rng = np.random.default_rng(12345)
    choice_arr_1 = rng.choice(grid_n*grid_n, size=k, replace=False)


    # make agent 2 choices similar to agent 1 and only mutate n_mutate choices
    choice_arr_2 = choice_arr_1.copy()
    bool_mutate_agent_2 = False
    if bool_mutate_agent_2:
        n_mutate = 4
        mutate_indices = rng.choice(k, size=n_mutate, replace=False)
        new_choices = rng.choice(grid_n*grid_n, size=n_mutate, replace=False)
        for idx, new_choice in zip(mutate_indices, new_choices):
            choice_arr_2[idx] = new_choice
        choice_arr_2 = np.unique(choice_arr_2)  # remove duplicates if any
        while len(choice_arr_2) < k:  # ensure we have exactly k choices
            new_choice = rng.choice(grid_n*grid_n)
            if new_choice not in choice_arr_2:
                choice_arr_2 = np.append(choice_arr_2, new_choice)
        choice_arr_2 = choice_arr_2[:k]  # trim to k if we exceeded
        # choice_arr_2.sort()

    choice_arr_3 = rng.choice(grid_n*grid_n, size=k, replace=False)
    choice_arr_4 = rng.choice(grid_n*grid_n, size=k, replace=False)



    # Calculate the rewards
    rewards_agent_1 = get_rewards(envList[0][env_index_to_plot], [{'x1': (idx // grid_n) + 1, 'x2': (idx % grid_n) + 1} for idx in choice_arr_1])
    rewards_agent_2 = get_rewards(envList[1][env_index_to_plot], [{'x1': (idx // grid_n) + 1, 'x2': (idx % grid_n) + 1} for idx in choice_arr_2])
    rewards_agent_3 = get_rewards(envList[2][env_index_to_plot], [{'x1': (idx // grid_n) + 1, 'x2': (idx % grid_n) + 1} for idx in choice_arr_3])
    rewards_agent_4 = get_rewards(envList[3][env_index_to_plot], [{'x1': (idx // grid_n) + 1, 'x2': (idx % grid_n) + 1} for idx in choice_arr_4])

    print("Rewards Agent 1:", rewards_agent_1)
    print("Rewards Agent 2:", rewards_agent_2)
    print("Rewards Agent 3:", rewards_agent_3)
    print("Rewards Agent 4:", rewards_agent_4)



    sim_pos = jaccard_on_positions(choice_arr_1, choice_arr_2)
    print("Position overlap similarity:", sim_pos)


    sim_rew = reward_corr_on_overlap(choice_arr_1, rewards_agent_1,
                                    choice_arr_2, rewards_agent_2, grid_n=11)
    print("Reward correlation on overlapping tiles:", sim_rew)


    sim_field = reward_map_corr(choice_arr_1, rewards_agent_1,
                                choice_arr_2, rewards_agent_2, grid_n=11)
    print("Spatial reward map correlation:", sim_field)


    sim_kernel = kernel_alignment_similarity(choice_arr_1, rewards_agent_1,
                                            choice_arr_2, rewards_agent_2,
                                            grid_n=11, sigma=1.5)
    print("Kernel-based spatial similarity:", sim_kernel)



    ## Calculate the similarity metrics

    # make a table of all similarity metrics between all pairs of agents
    agents_choices = [choice_arr_1, choice_arr_2, choice_arr_3, choice_arr_4]
    agents_rewards = [rewards_agent_1, rewards_agent_2, rewards_agent_3, rewards_agent_4]
    agent_names = ["Agent 1", "Agent 2", "Agent 3", "Agent 4"]  


    similarity_matrix, similarity_metrics = calculate_similarity_metrics(agents_choices, agents_rewards)



    # print the similarity matrices
    for metric in similarity_metrics:
        print(f"\nSimilarity Metric: {metric}")
        print(" " * 15 + "  ".join(f"{name:>10}" for name in agent_names))
        for i, name in enumerate(agent_names):
            row = "  ".join(f"{similarity_matrix[metric][i, j]:10.2f}" for j in range(4))
            print(f"{name:>15}  {row}")



    plot_envs(env_index_to_plot)
    # save the plot
    plt.savefig("new_delta_study/plots/env_plot.png")

    plot_choices_and_rewards()
    plt.savefig("new_delta_study/plots/choices_rewards_plot.png")

    plot_similarity_matrices(similarity_matrix, agent_names, similarity_metrics)
    plt.savefig("new_delta_study/plots/similarity_matrices.png")

    plt.show()






    