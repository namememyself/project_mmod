
import numpy as np
import networkx as nx
import random
import pickle
import os
import json

N_SAMPLES = 500
N_NODES = 2000
MAX_DEGREE = 200
EMB_DIM = 32
BASE_BETA = 0.08
BASE_GAMMA = 0.04
T_MAX = 100

def cap_degrees(G, max_deg):
    for n in G.nodes():
        neighbors = list(G.neighbors(n))
        if len(neighbors) > max_deg:
            to_remove = random.sample(neighbors, len(neighbors) - max_deg)
            G.remove_edges_from([(n, nbr) for nbr in to_remove])

def simulate_sir(G, node_features, content, base_beta, base_gamma, t_max):
    N = G.number_of_nodes()
    beta = base_beta * content['virality'] * node_features['susceptibility']
    gamma = base_gamma * (1.0 + 0.2 * (1 - node_features['activity']))
    infected = np.zeros(N, dtype=bool)
    recovered = np.zeros(N, dtype=bool)
    S_time, I_time, R_time = [], [], []
    events = []
    time_grid = np.arange(t_max)
    # Patient zero
    patient_zero = random.randint(0, N-1)
    infected[patient_zero] = True
    events.append((0, patient_zero))
    for t in time_grid:
        new_infected = []
        for i in range(N):
            if infected[i] and not recovered[i]:
                # Try to infect neighbors
                for j in G.neighbors(i):
                    if not infected[j] and not recovered[j]:
                        p = 1 - np.exp(-beta[j])
                        if random.random() < p:
                            new_infected.append(j)
                # Try to recover
                if random.random() < gamma[i]:
                    recovered[i] = True
        for j in new_infected:
            infected[j] = True
            events.append((t+1, j))
        S_time.append(np.sum(~infected & ~recovered))
        I_time.append(np.sum(infected & ~recovered))
        R_time.append(np.sum(recovered))
        if np.sum(infected & ~recovered) == 0:
            break
    return {
        'events': events,
        'time_grid': time_grid[:len(I_time)].tolist(),
        'I_time': I_time,
        'S_time': S_time,
        'R_time': R_time,
        'final_spread_ratio': R_time[-1] / N if R_time else 0.0
    }


# Save each sample as a JSON file in a directory
out_dir = 'synthetic_pheme'
os.makedirs(out_dir, exist_ok=True)
for sample_idx in range(N_SAMPLES):
    G = nx.barabasi_albert_graph(N_NODES, 5)
    cap_degrees(G, MAX_DEGREE)
    # Node features (keep as np arrays for simulation)
    followers = np.random.randint(10, 10000, N_NODES)
    activity = np.random.beta(2, 5, N_NODES)
    susceptibility = np.random.uniform(0.05, 0.5, N_NODES)
    node_features = {
        'followers': followers,
        'activity': activity,
        'susceptibility': susceptibility
    }
    # Content
    sentiment = np.random.uniform(-1, 1)
    embedding = np.random.normal(0, 1, EMB_DIM)
    virality = np.random.uniform(0.5, 1.5)
    content = {
        'sentiment': sentiment,
        'embedding': embedding.tolist(),
        'virality': virality
    }
    # Simulate cascade
    sir = simulate_sir(G, node_features, content, BASE_BETA, BASE_GAMMA, T_MAX)
    # Convert node_features to lists for JSON
    node_features_json = {
        'followers': followers.tolist(),
        'activity': activity.tolist(),
        'susceptibility': susceptibility.tolist()
    }
    # Convert SIR output to native Python types
    def to_native(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.generic, np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, dict):
            return {k: to_native(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [to_native(v) for v in obj]
        else:
            return obj

    sample = {
        'edges': [list(map(int, e)) for e in G.edges()],
        'node_features': node_features_json,
        'content': content,
        **sir
    }
    sample = to_native(sample)
    # Save as JSON
    with open(os.path.join(out_dir, f'sample_{sample_idx:04d}.json'), 'w') as f:
        json.dump(sample, f)
    if (sample_idx+1) % 10 == 0:
        print(f"Generated {sample_idx+1}/{N_SAMPLES}")
print(f'Saved synthetic dataset to {out_dir}/')
