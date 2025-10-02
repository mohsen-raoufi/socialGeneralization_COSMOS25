import sanity_check_model as scm
import numpy as np
import matplotlib.pyplot as plt



grid_n = 11         # width of the grid
k = 16             # number of choices
agent_n = 4         # number of agents/environments

# agent_index_to_plot = 0
env_index_to_plot = 0

envList = scm.load_env()

fig, ax = plt.subplots(figsize=(10, 6))

colors = ['r', 'g', 'b', 'm']
## correlation values vs. number of choices
# vary the number of choices from 1 to 120
# calculaute the similarity metrics compared to the other 3 agents, and plot them as a function of number of choices
n_all_choices = grid_n * grid_n
all_choices = np.arange(n_all_choices)
# shuffle the choices for each agent
# rng = np.random.default_rng(12345)
rng = np.random.default_rng()


nMC = 100

MC_corr_arr_all = []


for MC_rep in range(nMC):
    print("MC rep:", MC_rep)
        
    shuffled_choices_agent_1 = rng.permutation(all_choices)
    shuffled_choices_agent_2 = rng.permutation(all_choices)
    shuffled_choices_agent_3 = rng.permutation(all_choices)
    shuffled_choices_agent_4 = rng.permutation(all_choices)


    all_rewards_agent_1 = scm.get_rewards(envList[0][env_index_to_plot], [{'x1': (idx // grid_n) + 1, 'x2': (idx % grid_n) + 1} for idx in shuffled_choices_agent_1])
    all_rewards_agent_2 = scm.get_rewards(envList[1][env_index_to_plot], [{'x1': (idx // grid_n) + 1, 'x2': (idx % grid_n) + 1} for idx in shuffled_choices_agent_2])
    all_rewards_agent_3 = scm.get_rewards(envList[2][env_index_to_plot], [{'x1': (idx // grid_n) + 1, 'x2': (idx % grid_n) + 1} for idx in shuffled_choices_agent_3])
    all_rewards_agent_4 = scm.get_rewards(envList[3][env_index_to_plot], [{'x1': (idx // grid_n) + 1, 'x2': (idx % grid_n) + 1} for idx in shuffled_choices_agent_4])


    num_choice_arr = np.arange(1, 120, 2)
    corr_arr_dict = []
    for n_choices in num_choice_arr:
        choice_arr_1 = shuffled_choices_agent_1[:n_choices]
        choice_arr_2 = shuffled_choices_agent_2[:n_choices]
        choice_arr_3 = shuffled_choices_agent_3[:n_choices]
        choice_arr_4 = shuffled_choices_agent_4[:n_choices]
        rewards_agent_1 = all_rewards_agent_1[:n_choices]
        rewards_agent_2 = all_rewards_agent_2[:n_choices]
        rewards_agent_3 = all_rewards_agent_3[:n_choices]
        rewards_agent_4 = all_rewards_agent_4[:n_choices]

        agents_choices = [choice_arr_1, choice_arr_2, choice_arr_3, choice_arr_4]
        agents_rewards = [rewards_agent_1, rewards_agent_2, rewards_agent_3, rewards_agent_4]

        # print("n choices:", n_choices)
        # print("choices:", agents_choices)
        # print("rewards:", agents_rewards)

        # similarity_metrics = scm.calculate_similarity_metrics(agents_choices, agents_rewards)
        corr_tmp = np.zeros((agent_n, agent_n))
        for i in range(agent_n):
            for j in range(agent_n):
                sim_field = scm.reward_map_corr(agents_choices[i], agents_rewards[i],
                                                    agents_choices[j], agents_rewards[j], grid_n=11)
                corr_tmp[i, j] = sim_field

        corr_arr_dict.append(corr_tmp)


    corr_arr_np = np.array(corr_arr_dict)
    # print(corr_arr_np.shape)   # (num_choice, agent_n, agent_n)
    corr_arr_1_2 = corr_arr_np[:, 0, 1]
    corr_arr_1_3 = corr_arr_np[:, 0, 2]
    corr_arr_1_4 = corr_arr_np[:, 0, 3]
    # corr_arr_2_3 = corr_arr_np[:, 1, 2]
    # corr_arr_2_4 = corr_arr_np[:, 1, 3]
    # corr_arr_3_4 = corr_arr_np[:, 2, 3]

    all_corr_arr = [corr_arr_1_2, corr_arr_1_3, corr_arr_1_4]

    MC_corr_arr_all.append(all_corr_arr)

    # plt.plot(num_choice_arr, corr_arr_1_2, label='Agent 1 vs Agent 2')
    for i, corr_arr in enumerate(all_corr_arr):
        plt.plot(num_choice_arr, corr_arr, label=f'Pair {i+1}', color=colors[i], alpha=0.05)


MC_corr_arr_all_np = np.array(MC_corr_arr_all)

# make the average plot of the MC runs
mean_corr = np.mean(MC_corr_arr_all_np, axis=0)

for i, corr_arr in enumerate(mean_corr):
    plt.plot(num_choice_arr, corr_arr, label=f'Pair {i+1} Mean', color=colors[i], linewidth=2)
plt.xlabel('Number of Choices')
plt.ylabel('Correlation')
plt.title('Correlation vs Number of Choices (Averaged over MC runs)')
plt.legend()
plt.show()