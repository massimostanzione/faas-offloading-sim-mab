import sys
import json
import matplotlib.pyplot as plt
import numpy as np

NODE_TYPE = 'heterogeneous'

UPDATE_INTERVAL = 3600

policy_colors = {"random-lb":"green", "round-robin-lb":"blue", "mama-lb":"orange", "const-hash-lb":"purple", "wrr-speedup-lb":"lawngreen", "wrr-memory-lb":"dodgerblue", "wrr-cost-lb":"fuchsia"}
cloud_nodes_colors = {"cloud1":"blue", "cloud2":"orange", "cloud3":"green", "cloud4":"red", "cloud5":"yellow", "cloud6":"pink", "cloud7":"purple", "cloud8":"lawngreen"}

policy2node2zero = {
    "random-lb":{"cloud1":0, "cloud2":0, "cloud3":0, "cloud4":0, "cloud5":0, "cloud6":0, "cloud7":0, "cloud8":0},
    "round-robin-lb":{"cloud1":0, "cloud2":0, "cloud3":0, "cloud4":0, "cloud5":0, "cloud6":0, "cloud7":0, "cloud8":0},
    "mama-lb":{"cloud1":0, "cloud2":0, "cloud3":0, "cloud4":0, "cloud5":0, "cloud6":0, "cloud7":0, "cloud8":0},
    "const-hash-lb":{"cloud1":0, "cloud2":0, "cloud3":0, "cloud4":0, "cloud5":0, "cloud6":0, "cloud7":0, "cloud8":0},
    "wrr-speedup-lb":{"cloud1":0, "cloud2":0, "cloud3":0, "cloud4":0, "cloud5":0, "cloud6":0, "cloud7":0, "cloud8":0},
    "wrr-memory-lb":{"cloud1":0, "cloud2":0, "cloud3":0, "cloud4":0, "cloud5":0, "cloud6":0, "cloud7":0, "cloud8":0},
    "wrr-cost-lb":{"cloud1":0, "cloud2":0, "cloud3":0, "cloud4":0, "cloud5":0, "cloud6":0, "cloud7":0, "cloud8":0}
}

def _plot_mean_cum_reward(time_frames, rewards):
    cum_rewards = []
    cum_reward = 0
    n = 0
    for i in range(0, len(time_frames)):
        cum_reward += rewards[i]
        n = i+1
        cum_rewards.append(cum_reward / n)
    plt.plot(time_frames, cum_rewards, color="red", label="mean reward")

def plot_rewards(time_frames, rewards, policies, title):
    # Estrai politiche uniche e corrispondenti colori
    unique_policies = list(set(policies))

    """
    if max(rewards) >= 0.0:
        plt.ylim(min(rewards) + min(rewards)*10/100, max(rewards) + 10/100)
    else:
        plt.ylim(min(rewards) + min(rewards)*10/100, max(rewards) - max(rewards)*10/100)
    """
    #plt.ylim(-0.8, 0.2)

    # Creazione del grafico
    for _, policy in enumerate(unique_policies):
        policy_indices = [j for j, p in enumerate(policies) if p == policy]
        policy_time_frames = [time_frames[j] for j in policy_indices]
        policy_rewards = [rewards[j] for j in policy_indices]
        plt.scatter(policy_time_frames, policy_rewards, color=policy_colors[policy], label=policy)

    # Grafico media mobile del reward
    _plot_mean_cum_reward(time_frames, rewards)

    plt.axvline(x=UPDATE_INTERVAL, color='black', linestyle='--', label='weights updated')

    # Aggiunta di etichette agli assi e titolo
    plt.xlabel('Time (s)')
    plt.ylabel('Reward')
    plt.title(f'Rewards with {title} nodes')
    plt.legend() # Aggiunge la legenda con le etichette delle politiche

    # Mostra il grafico
    plt.tight_layout()
    plt.grid(axis="y")
    plt.show()

def plot_time_rewards(time_frames, rewards, policies, title):
    # Estrai politiche uniche e corrispondenti colori
    unique_policies = list(set(policies))
    #policy_colors = plt.cm.tab10.colors[:len(unique_policies)]  # Scegliamo i primi colori dalla tabella colori "tab10"

    # plt.figure(figsize=(5, 8))

    # Collega i punti dello scatter plot con una linea
    #plt.plot(time_frames, rewards, linestyle='-', color='darkgray')
    plt.ylim(min(rewards) + min(rewards)*10/100, max(rewards) - max(rewards)*10/100)

    # Creazione del grafico
    for i, policy in enumerate(unique_policies):
        policy_indices = [j for j, p in enumerate(policies) if p == policy]
        policy_time_frames = [time_frames[j] for j in policy_indices]
        policy_rewards = [rewards[j] for j in policy_indices]
        plt.scatter(policy_time_frames, policy_rewards, color=policy_colors[policy], label=policy)

    # Grafico media mobile del reward
    _plot_mean_cum_reward(time_frames, rewards)

    # plt.axhline(y=1.0, color='r', linestyle='--')

    # Aggiunta delle etichette degli assi x con entrambi i tempi e le politiche
    #combined_labels = [f'{time_frame} ({policy})' for time_frame, policy in zip(time_frames, policies)]
    #plt.xticks(time_frames, combined_labels, rotation=45, ha='right') 
    #plt.xticks(time_frames, rotation=60)

    # Aggiunta di etichette agli assi e titolo
    plt.xlabel('Time (s)')
    plt.ylabel('Reward')
    plt.title(f'Rewards: response time ({title} nodes)')
    plt.legend() # Aggiunge la legenda con le etichette delle politiche

    # Mostra il grafico
    plt.tight_layout()
    plt.grid(axis="y")
    plt.show()

def plot_load_imbalance_rewards(time_frames, rewards, policies, title):
    # Estrai politiche uniche e corrispondenti colori
    unique_policies = list(set(policies))
    #policy_colors = plt.cm.tab10.colors[:len(unique_policies)]  # Scegliamo i primi colori dalla tabella colori "tab10"

    #plt.plot(time_frames, rewards, color="darkgray")

    # Creazione del grafico
    for _, policy in enumerate(unique_policies):
        policy_indices = [j for j, p in enumerate(policies) if p == policy]
        policy_time_frames = [time_frames[j] for j in policy_indices]
        policy_rewards = [rewards[j] for j in policy_indices]
        plt.scatter(policy_time_frames, policy_rewards, color=policy_colors[policy], label=policy)

    _plot_mean_cum_reward(time_frames, rewards)

    #plt.axhline(y=1.0, color='r', linestyle='--')

    # Aggiunta delle etichette degli assi x con entrambi i tempi e le politiche
    #combined_labels = [f'{time_frame} ({policy})' for time_frame, policy in zip(time_frames, policies)]
    #plt.xticks(time_frames, combined_labels, rotation=45, ha='right') 
    #plt.xticks(time_frames, rotation=60)

    # Aggiunta di etichette agli assi e titolo
    plt.xlabel('Time (s)')
    plt.ylabel('Reward')
    plt.title(f'Rewards: load imbalance ({title} nodes)')
    plt.legend() # Aggiunge la legenda con le etichette delle politiche

    # Mostra il grafico
    plt.tight_layout()
    plt.grid(axis="y")
    plt.show()

def plot_dropped_percentage_rewards(time_frames, rewards, policies, title):
    # Estrai politiche uniche e corrispondenti colori
    unique_policies = list(set(policies))

    # Creazione del grafico
    for _, policy in enumerate(unique_policies):
        policy_indices = [j for j, p in enumerate(policies) if p == policy]
        policy_time_frames = [time_frames[j] for j in policy_indices]
        policy_rewards = [rewards[j] for j in policy_indices]
        plt.scatter(policy_time_frames, policy_rewards, color=policy_colors[policy], label=policy)

    _plot_mean_cum_reward(time_frames, rewards)

    # Aggiunta di etichette agli assi e titolo
    plt.xlabel('Time (s)')
    plt.ylabel('Reward')
    plt.title(f'Rewards: dropped percentage ({title} nodes)')
    plt.legend() # Aggiunge la legenda con le etichette delle politiche

    # Mostra il grafico
    plt.tight_layout()
    plt.grid(axis="y")
    plt.show()

def plot_server_loads(time_frames, server_1_reqs, server_2_reqs, server_3_reqs, server_4_reqs, server_5_reqs, server_6_reqs, server_7_reqs, server_8_reqs, title):

    plt.plot(time_frames, server_1_reqs, label='cloud1', color=cloud_nodes_colors["cloud1"])
    plt.plot(time_frames, server_2_reqs, label='cloud2', color=cloud_nodes_colors["cloud2"])
    plt.plot(time_frames, server_3_reqs, label='cloud3', color=cloud_nodes_colors["cloud3"])
    plt.plot(time_frames, server_4_reqs, label='cloud4', color=cloud_nodes_colors["cloud4"])
    plt.plot(time_frames, server_5_reqs, label='cloud5', color=cloud_nodes_colors["cloud5"])
    plt.plot(time_frames, server_6_reqs, label='cloud6', color=cloud_nodes_colors["cloud6"])
    plt.plot(time_frames, server_7_reqs, label='cloud7', color=cloud_nodes_colors["cloud7"])
    plt.plot(time_frames, server_8_reqs, label='cloud8', color=cloud_nodes_colors["cloud8"])

    # Aggiunta di etichette agli assi e titolo
    plt.xlabel('Time (s)')
    plt.ylabel('Number of requests')
    plt.title(f'Load evolution ({title} nodes)')

    # Aggiunge la legenda
    plt.legend()

    # Mostra il grafico
    #plt.grid(True)  # Opzionale: aggiunge la griglia al grafico
    #plt.tight_layout()  # Opzionale: migliora la disposizione degli elementi del grafico
    plt.show()

def plot_server_loads_cum(time_frames, server_1_reqs_cum, server_2_reqs_cum, server_3_reqs_cum, policies):
    plt.plot(time_frames, server_1_reqs_cum, label='cloud1')
    plt.plot(time_frames, server_2_reqs_cum, label='cloud2')
    plt.plot(time_frames, server_3_reqs_cum, label='cloud3')
   
    # Aggiunta delle etichette degli assi x con entrambi i tempi e le politiche
    combined_labels = [f'{time_frame} ({policy})' for time_frame, policy in zip(time_frames, policies)]
    plt.xticks(time_frames, combined_labels, rotation=45, ha='right') 

    # Aggiunta di etichette agli assi e titolo
    plt.xlabel('Time (s)')
    plt.ylabel('N° of requests')
    plt.title('Load evolution')

    # Aggiunge la legenda
    plt.legend()

    # Mostra il grafico
    #plt.grid(True)  # Opzionale: aggiunge la griglia al grafico
    plt.tight_layout()  # Opzionale: migliora la disposizione degli elementi del grafico
    plt.show()

def plot_completion_percentage_reward(time_frames, rewards, policies):
    pass

def plot_number_selected(data: dict, title):
    categories = data.keys()
    heights = data.values() 
    # Grafico dell'istogramma
    plt.bar(categories, heights)

    # Impostazione del valore minimo e massimo dell'asse y
    #plt.ylim(0, 100)  

    # Aggiunta di etichette e titolo
    plt.xlabel('Load Balancing Policies')
    plt.ylabel('Policy Usage Count')
    plt.title(f'Policy Invocation Frequency ({title} nodes)')

    # Mostra il grafico
    plt.show()

def plot_dropped_reqs(time_frames, server_1_dropped_reqs, server_2_dropped_reqs, server_3_dropped_reqs, server_4_dropped_reqs, server_5_dropped_reqs, server_6_dropped_reqs, title):
    
    plt.plot(time_frames, server_1_dropped_reqs, label='cloud1', color=cloud_nodes_colors["cloud1"])
    plt.plot(time_frames, server_2_dropped_reqs, label='cloud2', color=cloud_nodes_colors["cloud2"])
    plt.plot(time_frames, server_3_dropped_reqs, label='cloud3', color=cloud_nodes_colors["cloud3"])
    plt.plot(time_frames, server_4_dropped_reqs, label='cloud4', color=cloud_nodes_colors["cloud4"])
    plt.plot(time_frames, server_5_dropped_reqs, label='cloud5', color=cloud_nodes_colors["cloud5"])
    plt.plot(time_frames, server_6_dropped_reqs, label='cloud6', color=cloud_nodes_colors["cloud6"])
   
    # Aggiunta di etichette agli assi e titolo
    plt.xlabel('Time (s)')
    plt.ylabel('Number of requests')
    plt.title(f'Dropped requests ({title} nodes)')

    # Aggiunge la legenda
    plt.legend()

    # Mostra il grafico
    #plt.grid(True)  # Opzionale: aggiunge la griglia al grafico
    plt.tight_layout()  # Opzionale: migliora la disposizione degli elementi del grafico
    plt.show()

def plot_drop_reqs_bar(data: dict, title):
    # set width of bar 
    barWidth = 0.1
    
    # set height of bar 
    cloud1 = [data[policy]["cloud1"] for policy in data]
    cloud2 = [data[policy]["cloud2"] for policy in data]
    cloud3 = [data[policy]["cloud3"] for policy in data]
    cloud4 = [data[policy]["cloud4"] for policy in data]
    cloud5 = [data[policy]["cloud5"] for policy in data]
    cloud6 = [data[policy]["cloud6"] for policy in data]
    cloud7 = [data[policy]["cloud7"] for policy in data]
    cloud8 = [data[policy]["cloud8"] for policy in data]
    
    # Set position of bar on X axis 
    br1 = np.arange(len(cloud1)) 
    br2 = [x + barWidth for x in br1] 
    br3 = [x + barWidth for x in br2]
    br4 = [x + barWidth for x in br3]
    br5 = [x + barWidth for x in br4] 
    br6 = [x + barWidth for x in br5] 
    br7 = [x + barWidth for x in br6] 
    br8 = [x + barWidth for x in br7] 

    # Make the plot
    plt.bar(br1, cloud1, color = cloud_nodes_colors["cloud1"], width = barWidth, label ='cloud1') 
    plt.bar(br2, cloud2, color = cloud_nodes_colors["cloud2"], width = barWidth, label ='cloud2') 
    plt.bar(br3, cloud3, color = cloud_nodes_colors["cloud3"], width = barWidth, label ='cloud3')
    plt.bar(br4, cloud4, color = cloud_nodes_colors["cloud4"], width = barWidth, label ='cloud4')
    plt.bar(br5, cloud5, color = cloud_nodes_colors["cloud5"], width = barWidth, label ='cloud5')
    plt.bar(br6, cloud6, color = cloud_nodes_colors["cloud6"], width = barWidth, label ='cloud6') 
    plt.bar(br7, cloud7, color = cloud_nodes_colors["cloud7"], width = barWidth, label ='cloud7') 
    plt.bar(br8, cloud8, color = cloud_nodes_colors["cloud8"], width = barWidth, label ='cloud8') 
    
    # Adding plot info
    plt.title(f'Number of requests dropped by cloud nodes ({title})')
    plt.xlabel('Load balancing policies') 
    plt.ylabel('Number of dropped requests') 
    plt.xticks([r + barWidth + 0.15 for r in range(len(cloud1))], ['random-lb', 'round-robin-lb', 'mama-lb', 'const-hash-lb', 'wrr-speedup-lb', 'wrr-memory-lb', 'wrr-cost-lb'])
    plt.legend()
    plt.show()

def plot_drop_reqs_bar_percentage(data: dict, title):
    categories = data.keys()
    heights = data.values() 

    # Creazione delle etichette personalizzate
    custom_labels = [f'{policy} ({percentage:.2f}%)' for policy, percentage in zip(categories, heights)]

    # Creazione dell'istogramma
    fig, ax = plt.subplots()
    ax.bar(categories, heights)

    # Aggiunta dei nomi personalizzati per i tick dell'asse x
    ax.set_xticks(range(len(categories)))  # Imposta la posizione dei tick
    ax.set_xticklabels(custom_labels, rotation=30, ha="center")  # Imposta le etichette personalizzate

    # Aggiunta di etichette e titolo
    ax.set_xlabel('Load Balancing Policies')
    ax.set_ylabel('Percentage of dropped requests (%)')
    ax.set_title('Percentage of dropped requests per policy')

    # Mostra il grafico
    plt.tight_layout()  # Per evitare che le etichette si sovrappongano
    plt.show()

def plot_resp_times(time_frames, resp_times, title):
    
    cum_resp_times = []
    cum_resp_time = 0
    n = 0
    for i in range(0, len(resp_times)):
        cum_resp_time += resp_times[i]
        n = i+1
        cum_resp_times.append(cum_resp_time / n)

    plt.plot(time_frames, cum_resp_times, color="blue", label="mean response time")
    plt.plot(time_frames, resp_times, color="green", label="response time")
    plt.xlim(right=28800)
    plt.axvline(x=UPDATE_INTERVAL, color='black', linestyle='--', label='weights updated')

    plt.ylim(0.05, 0.4)

    # Aggiunta di etichette agli assi e titolo
    plt.xlabel('Time (s)')
    plt.ylabel('Response Time')
    plt.title(f'Response Time with {title} nodes')
    plt.legend()

    # Mostra il grafico
    plt.tight_layout()
    plt.show()

def plot_cost(time_frames, costs, title):
    
    cum_costs = []
    cum_cost = 0
    n = 0
    for i in range(0, len(costs)):
        cum_cost += costs[i]
        n = i+1
        cum_costs.append(cum_cost / n)

    plt.plot(time_frames, cum_costs, color="red", label="mean cost")
    plt.plot(time_frames, costs, color="green", label="cost")
    plt.xlim(right=28800)
    plt.axvline(x=UPDATE_INTERVAL, color='black', linestyle='--', label='weights updated')

    plt.ylim(0.05, 0.4)

    # Aggiunta di etichette agli assi e titolo
    plt.xlabel('Time (s)')
    plt.ylabel('Cost')
    plt.title(f'Cost with {title} nodes')
    plt.legend()

    # Mostra il grafico
    plt.tight_layout()
    plt.show()

def plot_utility(time_frames, utilities, title):
    
    cum_utilities = []
    cum_utility = 0
    n = 0
    for i in range(0, len(utilities)):
        cum_utility += utilities[i]
        n = i+1
        cum_utilities.append(cum_utility / n)

    plt.plot(time_frames, cum_utilities, color="blue", label="mean utility")
    plt.plot(time_frames, utilities, color="green", label="utility")
    plt.axvline(x=UPDATE_INTERVAL, color='black', linestyle='--', label='weights updated')

    plt.ylim(7000, 21000)

    # Aggiunta di etichette agli assi e titolo
    plt.xlabel('Time (s)')
    plt.ylabel('Utility')
    plt.title(f'Utility with {title} nodes')
    plt.legend()

    # Mostra il grafico
    plt.tight_layout()
    plt.show()


def plot_load_imbalance_rewards_ax(ax, time_frames, rewards, policies):
    # Estrai politiche uniche e corrispondenti colori
    unique_policies = list(set(policies))
    #policy_colors = plt.cm.tab10.colors[:len(unique_policies)]  # Scegliamo i primi colori dalla tabella colori "tab10"

    #plt.plot(time_frames, rewards, color="darkgray")

    # Creazione del grafico
    for _, policy in enumerate(unique_policies):
        policy_indices = [j for j, p in enumerate(policies) if p == policy]
        policy_time_frames = [time_frames[j] for j in policy_indices]
        policy_rewards = [rewards[j] for j in policy_indices]
        ax.scatter(policy_time_frames, policy_rewards, color=policy_colors[policy], label=policy)

    _plot_mean_cum_reward(ax, time_frames, rewards)

    #plt.axhline(y=1.0, color='r', linestyle='--')

    # Aggiunta delle etichette degli assi x con entrambi i tempi e le politiche
    #combined_labels = [f'{time_frame} ({policy})' for time_frame, policy in zip(time_frames, policies)]
    #plt.xticks(time_frames, combined_labels, rotation=45, ha='right') 
    #plt.xticks(time_frames, rotation=60)

    # Aggiunta di etichette agli assi e titolo
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Reward')
    ax.set_title('Rewards: load imbalance (homogeneous nodes)')
    ax.legend() # Aggiunge la legenda con le etichette delle politiche
    ax.grid(axis="y")

def plot_server_loads_ax(ax, time_frames, server_1_reqs, server_2_reqs, server_3_reqs, policies):

    ax.plot(time_frames, server_1_reqs, label='cloud1')
    ax.plot(time_frames, server_2_reqs, label='cloud2')
    ax.plot(time_frames, server_3_reqs, label='cloud3')
   
    # Aggiunta delle etichette degli assi x con entrambi i tempi e le politiche
    combined_labels = [f'{time_frame} ({policy})' for time_frame, policy in zip(time_frames, policies)]
    # plt.xticks(time_frames, combined_labels, rotation=45, ha='right') 

    # Aggiunta di etichette agli assi e titolo
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Number of requests')
    ax.set_title('Load evolution (homogenous nodes)')

    # Aggiunge la legenda
    ax.legend()

def test_double_plot(time_frames, rewards, server_1_reqs, server_2_reqs, server_3_reqs, policies):
    # Creazione di due assi (subplot) sulla stessa figura
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))  # 2 righe, 1 colonna

    # Grafico 1
    plot_load_imbalance_rewards_ax(ax1, time_frames, rewards, policies)
    
    # Grafico 2
    plot_server_loads_ax(ax2, time_frames, server_1_reqs, server_2_reqs, server_3_reqs, policies)

    # Ottimizza la disposizione dei subplot
    plt.tight_layout()

    # Mostra la figura con i due grafici
    plt.show()



if __name__ == '__main__':

    """
    if len(sys.argv) < 3:
        print("Uso: nome_programma tipo_di_grafici[response-time, load-imbalance, dropped-percentage, all] tipo_di_nodi[homogeneous, heterogeneous]")
        exit(1)
    """

    with open('mab_stats.json', 'r') as f:
        data = json.load(f)
    
    time_frames = []
    policies = []
    policies_frequency = {"random-lb":0, "round-robin-lb":0, "mama-lb":0, "const-hash-lb":0, "wrr-speedup-lb":0, "wrr-memory-lb":0, "wrr-cost-lb":0}
    pol_fr_prev = {"random-lb":0, "round-robin-lb":0, "mama-lb":0, "const-hash-lb":0, "wrr-speedup-lb":0, "wrr-memory-lb":0, "wrr-cost-lb":0}
    pol_fr_post = {"random-lb":0, "round-robin-lb":0, "mama-lb":0, "const-hash-lb":0, "wrr-speedup-lb":0, "wrr-memory-lb":0, "wrr-cost-lb":0}
    server_loads = []
    server_loads_cum = []
    dropped_reqs = []
    rewards = []
    cum_reward = 0
    cum_rewards = []
    resp_times = []
    costs = []
    utilities = []
    total_cost = 0
    total_utility = 0
    
    for d in data:
        time_frames.append(d['time'])
        policies.append(d['policy'])
        policies_frequency[d['policy']] += 1 
        if d['time'] < 9000:
            pol_fr_prev[d['policy']] += 1
        else:
            pol_fr_post[d['policy']] += 1
        server_loads.append(d['server_loads'])
        server_loads_cum.append(d['server_loads_cum'])
        dropped_reqs.append(d['dropped_reqs'])
        rewards.append(d['reward'])
        cum_reward += d['reward']
        cum_rewards.append(cum_reward)
        resp_times.append(d['avg_resp_time'])
        costs.append(d['cost'])
        utilities.append(d['utility'])
        total_cost += d['cost']
        total_utility += d['utility']
    
    data_drops = {
        "random-lb":{"cloud1":0, "cloud2":0, "cloud3":0, "cloud4":0, "cloud5":0, "cloud6":0, "cloud7":0, "cloud8":0},
        "round-robin-lb":{"cloud1":0, "cloud2":0, "cloud3":0, "cloud4":0, "cloud5":0, "cloud6":0, "cloud7":0, "cloud8":0},
        "mama-lb":{"cloud1":0, "cloud2":0, "cloud3":0, "cloud4":0, "cloud5":0, "cloud6":0, "cloud7":0, "cloud8":0},
        "const-hash-lb":{"cloud1":0, "cloud2":0, "cloud3":0, "cloud4":0, "cloud5":0, "cloud6":0, "cloud7":0, "cloud8":0},
        "wrr-speedup-lb":{"cloud1":0, "cloud2":0, "cloud3":0, "cloud4":0, "cloud5":0, "cloud6":0, "cloud7":0, "cloud8":0},
        "wrr-memory-lb":{"cloud1":0, "cloud2":0, "cloud3":0, "cloud4":0, "cloud5":0, "cloud6":0, "cloud7":0, "cloud8":0},
        "wrr-cost-lb":{"cloud1":0, "cloud2":0, "cloud3":0, "cloud4":0, "cloud5":0, "cloud6":0, "cloud7":0, "cloud8":0}
    }
    for i in range(0, len(dropped_reqs)):
        data_drops[policies[i]]["cloud1"] = data_drops.get(policies[i], 0)["cloud1"] + dropped_reqs[i][0] 
        data_drops[policies[i]]["cloud2"] = data_drops.get(policies[i], 0)["cloud2"] + dropped_reqs[i][1] 
        data_drops[policies[i]]["cloud3"] = data_drops.get(policies[i], 0)["cloud3"] + dropped_reqs[i][2]
        data_drops[policies[i]]["cloud4"] = data_drops.get(policies[i], 0)["cloud4"] + dropped_reqs[i][3]
        data_drops[policies[i]]["cloud5"] = data_drops.get(policies[i], 0)["cloud5"] + dropped_reqs[i][4]
        data_drops[policies[i]]["cloud6"] = data_drops.get(policies[i], 0)["cloud6"] + dropped_reqs[i][5] 
        data_drops[policies[i]]["cloud7"] = data_drops.get(policies[i], 0)["cloud7"] + dropped_reqs[i][6] 
        data_drops[policies[i]]["cloud8"] = data_drops.get(policies[i], 0)["cloud8"] + dropped_reqs[i][7] 
    #print("drops: ", data_drops)

    data_arrivals = {
        "random-lb":{"cloud1":0, "cloud2":0, "cloud3":0, "cloud4":0, "cloud5":0, "cloud6":0, "cloud7":0, "cloud8":0},
        "round-robin-lb":{"cloud1":0, "cloud2":0, "cloud3":0, "cloud4":0, "cloud5":0, "cloud6":0, "cloud7":0, "cloud8":0},
        "mama-lb":{"cloud1":0, "cloud2":0, "cloud3":0, "cloud4":0, "cloud5":0, "cloud6":0, "cloud7":0, "cloud8":0},
        "const-hash-lb":{"cloud1":0, "cloud2":0, "cloud3":0, "cloud4":0, "cloud5":0, "cloud6":0, "cloud7":0, "cloud8":0},
        "wrr-speedup-lb":{"cloud1":0, "cloud2":0, "cloud3":0, "cloud4":0, "cloud5":0, "cloud6":0, "cloud7":0, "cloud8":0},
        "wrr-memory-lb":{"cloud1":0, "cloud2":0, "cloud3":0, "cloud4":0, "cloud5":0, "cloud6":0, "cloud7":0, "cloud8":0},
        "wrr-cost-lb":{"cloud1":0, "cloud2":0, "cloud3":0, "cloud4":0, "cloud5":0, "cloud6":0, "cloud7":0, "cloud8":0}
    }
    for i in range(0, len(server_loads)):
        data_arrivals[policies[i]]["cloud1"] = data_arrivals.get(policies[i], 0)["cloud1"] + server_loads[i][0]
        data_arrivals[policies[i]]["cloud2"] = data_arrivals.get(policies[i], 0)["cloud2"] + server_loads[i][1]
        data_arrivals[policies[i]]["cloud3"] = data_arrivals.get(policies[i], 0)["cloud3"] + server_loads[i][2]
        data_arrivals[policies[i]]["cloud4"] = data_arrivals.get(policies[i], 0)["cloud4"] + server_loads[i][3]
        data_arrivals[policies[i]]["cloud5"] = data_arrivals.get(policies[i], 0)["cloud5"] + server_loads[i][4]
        data_arrivals[policies[i]]["cloud6"] = data_arrivals.get(policies[i], 0)["cloud6"] + server_loads[i][5]
        data_arrivals[policies[i]]["cloud7"] = data_arrivals.get(policies[i], 0)["cloud7"] + server_loads[i][6]
        data_arrivals[policies[i]]["cloud8"] = data_arrivals.get(policies[i], 0)["cloud8"] + server_loads[i][7]
    #print("arrivals: ", data_arrivals)

    """
    data_drops_percentage = policy2node2zero.copy()
    for key, value in data_drops.items():
        #print(key, value)
        if data_arrivals.get(key, 0)["cloud1"] != 0:
            data_drops_percentage[key]["cloud1"] = data_drops.get(key, 0)["cloud1"] / data_arrivals.get(key, 0)["cloud1"] 
        else: 
            data_drops_percentage[key]["cloud1"] = 0
        if data_arrivals.get(key, 0)["cloud2"] != 0:
            data_drops_percentage[key]["cloud2"] = data_drops.get(key, 0)["cloud2"] / data_arrivals.get(key, 0)["cloud2"] 
        else:
            data_drops_percentage[key]["cloud2"] = 0
        if data_arrivals.get(key, 0)["cloud3"] != 0:
            data_drops_percentage[key]["cloud3"] = data_drops.get(key, 0)["cloud3"] / data_arrivals.get(key, 0)["cloud3"] 
        else:
            data_drops_percentage[key]["cloud3"] = 0
        if data_arrivals.get(key, 0)["cloud4"] != 0:
            data_drops_percentage[key]["cloud4"] = data_drops.get(key, 0)["cloud4"] / data_arrivals.get(key, 0)["cloud4"] 
        else:
            data_drops_percentage[key]["cloud4"] = 0
        if data_arrivals.get(key, 0)["cloud5"] != 0:
            data_drops_percentage[key]["cloud5"] = data_drops.get(key, 0)["cloud5"] / data_arrivals.get(key, 0)["cloud5"] 
        else:
            data_drops_percentage[key]["cloud5"] = 0
        if data_arrivals.get(key, 0)["cloud6"] != 0:
            data_drops_percentage[key]["cloud6"] = data_drops.get(key, 0)["cloud6"] / data_arrivals.get(key, 0)["cloud6"] 
        else:
            data_drops_percentage[key]["cloud6"] = 0
        if data_arrivals.get(key, 0)["cloud7"] != 0:
            data_drops_percentage[key]["cloud7"] = data_drops.get(key, 0)["cloud7"] / data_arrivals.get(key, 0)["cloud7"] 
        else:
            data_drops_percentage[key]["cloud7"] = 0
        if data_arrivals.get(key, 0)["cloud8"] != 0:
            data_drops_percentage[key]["cloud8"] = data_drops.get(key, 0)["cloud8"] / data_arrivals.get(key, 0)["cloud8"] 
        else:
            data_drops_percentage[key]["cloud8"] = 0
    print(data_drops_percentage)
    """
    
    policy2arrivals = {}
    for policy, clouds in data_arrivals.items():
        policy2arrivals[policy] = sum(clouds.values())
    #print("pol2arvs: ", policy2arrivals)

    policy2drops = {}
    for policy, clouds in data_drops.items():
        policy2drops[policy] = sum(clouds.values())
    #print("pol2drops: ", policy2drops)

    policy2drops_percentage = {}
    # Calcolo della percentuale di drops sugli arrivi per ciascuna policy
    for policy in policy2arrivals.keys():
        if policy in policy2drops:
            arrivals = policy2arrivals[policy]
            drops = policy2drops[policy]
            drop_percentage = (drops / arrivals) * 100  # Calcola la percentuale
            policy2drops_percentage[policy] = drop_percentage

    server_1_reqs = []
    server_2_reqs = []
    server_3_reqs = []
    server_4_reqs = []
    server_5_reqs = []
    server_6_reqs = []
    server_7_reqs = []
    server_8_reqs = []
    for loads in server_loads:
        server_1_reqs.append(loads[0])
        server_2_reqs.append(loads[1])
        server_3_reqs.append(loads[2])
        server_4_reqs.append(loads[3])
        server_5_reqs.append(loads[4])
        server_6_reqs.append(loads[5])
        server_7_reqs.append(loads[6])
        server_8_reqs.append(loads[7])
    
    server_1_reqs_cum = []
    server_2_reqs_cum = []
    server_3_reqs_cum = []
    for loads in server_loads_cum:
        server_1_reqs_cum.append(loads[0])
        server_2_reqs_cum.append(loads[1])
        server_3_reqs_cum.append(loads[2])

    server_1_dropped_reqs = []
    server_2_dropped_reqs = []
    server_3_dropped_reqs = []
    server_4_dropped_reqs = []
    server_5_dropped_reqs = []
    server_6_dropped_reqs = []
    server_7_dropped_reqs = []
    server_8_dropped_reqs = []
    for dropped in dropped_reqs:
        server_1_dropped_reqs.append(dropped[0])
        server_2_dropped_reqs.append(dropped[1])
        server_3_dropped_reqs.append(dropped[2])
        server_4_dropped_reqs.append(dropped[3])
        server_5_dropped_reqs.append(dropped[4])
        server_6_dropped_reqs.append(dropped[5])
        server_7_dropped_reqs.append(dropped[6])
        server_8_dropped_reqs.append(dropped[7])

    total_drops = 0
    for dropped in dropped_reqs:
        total_drops += sum(dropped)
    print("total drops: ", total_drops)
    print("total cost: ", total_cost)
    print("total utility: ", total_utility)

    """
    if (sys.argv[1] == "response-time"):
        plot_time_rewards(time_frames, rewards, policies, title=sys.argv[2])
    elif (sys.argv[1] == "load-imbalance"):
        plot_load_imbalance_rewards(time_frames, rewards, policies, title=sys.argv[2])
    elif (sys.argv[1] == "dropped-percentage"):
        plot_dropped_percentage_rewards(time_frames, rewards, policies, title=sys.argv[2])
    else:
        RuntimeError(f"Unknown {sys.argv[1]}")
    """

    # plot_time_rewards(time_frames, rewards, policies, title=sys.argv[2])
    # plot_rewards(time_frames, rewards, policies, title=NODE_TYPE)
    # plot_server_loads(time_frames, server_1_reqs, server_2_reqs, server_3_reqs, server_4_reqs, server_5_reqs, server_6_reqs, server_7_reqs, server_8_reqs, title=NODE_TYPE)
    # plot_number_selected(policies_frequency, title=NODE_TYPE)
    plot_number_selected(pol_fr_prev, title=NODE_TYPE)
    plot_number_selected(pol_fr_post, title=NODE_TYPE)
    # plot_drop_reqs_bar(data_drops, title=NODE_TYPE)
    # plot_drop_reqs_bar_percentage(policy2drops_percentage, title=NODE_TYPE)
    plot_resp_times(time_frames, resp_times, title=NODE_TYPE)
    plot_cost(time_frames, costs, title=NODE_TYPE)
    # plot_utility(time_frames, utilities, title=NODE_TYPE)
