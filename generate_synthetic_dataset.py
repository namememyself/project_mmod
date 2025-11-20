import os
import json
import random
import torch
import numpy as np
import networkx as nx
from transformers import BertTokenizer, BertModel

# Settings
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

NUM_SAMPLES = 300
N_RANGE = (200, 1000)
M_RANGE = (3, 5)
T = 48
EMB_DIM = 768
SUMMARY_PATH = 'synthetic/summary.json'
DATA_DIR = 'synthetic'

os.makedirs(DATA_DIR, exist_ok=True)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert = BertModel.from_pretrained('bert-base-uncased')
bert.eval()

# Helper functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def random_text():
    # Generate a random sentence of 8-16 words
    vocab = ["rumor", "news", "event", "breaking", "update", "alert", "report", "claim", "viral", "fact", "false", "true", "witness", "source", "official", "statement", "public", "media", "spread", "denial", "confirmation", "investigation", "incident", "response", "community", "online", "platform", "account", "tweet", "post"]
    n = random.randint(8, 16)
    return " ".join(random.choices(vocab, k=n))

def get_bert_embedding(text):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=32)
        outputs = bert(**inputs)
        emb = outputs.last_hidden_state[:, 0, :].squeeze(0)  # [CLS] token
        return emb

def generate_sample():
    MAX_ATTEMPTS = 100
    for attempt in range(MAX_ATTEMPTS):
        N = random.randint(*N_RANGE)
        m = random.randint(*M_RANGE)
        G = nx.barabasi_albert_graph(N, m)
        edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous()
        edge_weight = torch.ones(edge_index.size(1), dtype=torch.float32)

        # Node features
        followers = (torch.rand(N) ** -2.5) * 5000
        followers = torch.clamp(followers, 1, 1e5)
        activity = torch.randint(1, 50, (N,), dtype=torch.float32)
        noise = torch.randn(N) * 0.1
        susceptibility = torch.sigmoid(0.4 * torch.log(followers + 1) + 0.2 * activity + noise)
        susceptibility = torch.clamp(susceptibility, 0, 1)

        # Content features
        sentiment = random.uniform(-1, 1)
        text = random_text()
        embedding = get_bert_embedding(text)
        if torch.isnan(embedding).any() or torch.isinf(embedding).any():
            continue

        # Virality
        base_beta = random.uniform(0.01, 0.1)
        controversy = random.uniform(-1, 1)
        w_s, w_c = 1.0, 0.7
        content_virality = sigmoid(w_s * sentiment + w_c * controversy)
        beta = base_beta * (0.5 + susceptibility) * content_virality
        beta = torch.clamp(beta, 1e-4, 1.0)
        gamma = torch.FloatTensor(N).uniform_(0.02, 0.08)
        gamma = torch.clamp(gamma, 1e-4, 1.0)

        # SIR simulation
        S = torch.ones(N)
        I = torch.zeros(N)
        R = torch.zeros(N)
        patient_zero = random.randint(0, N-1)
        I[patient_zero] = 1.0
        S[patient_zero] = 0.0
        I_norm = []
        adj = torch.zeros((N, N))
        for u, v in G.edges:
            adj[u, v] = 1
            adj[v, u] = 1
        nan_flag = False
        for t in range(T):
            force = torch.matmul(adj, I)
            new_infections = beta * S * force
            recoveries = gamma * I
            S = S - new_infections
            I = I + new_infections - recoveries
            R = R + recoveries
            S = torch.clamp(S, 0, 1)
            I = torch.clamp(I, 0, 1)
            R = torch.clamp(R, 0, 1)
            if torch.isnan(S).any() or torch.isinf(S).any() or torch.isnan(I).any() or torch.isinf(I).any() or torch.isnan(R).any() or torch.isinf(R).any():
                nan_flag = True
                break
            I_norm.append(I.sum().item() / N)
            if I.sum().item() < 1e-3:
                break
        I_norm = torch.tensor(I_norm, dtype=torch.float32)
        # Final check for NaN/Inf in all outputs
        if (torch.isnan(followers).any() or torch.isinf(followers).any() or
            torch.isnan(activity).any() or torch.isinf(activity).any() or
            torch.isnan(susceptibility).any() or torch.isinf(susceptibility).any() or
            torch.isnan(beta).any() or torch.isinf(beta).any() or
            torch.isnan(gamma).any() or torch.isinf(gamma).any() or
            torch.isnan(I_norm).any() or torch.isinf(I_norm).any() or
            nan_flag):
            continue
        data = {
            "N": N,
            "edge_index": edge_index,
            "edge_weight": edge_weight,
            "followers": followers,
            "activity": activity,
            "susceptibility": susceptibility,
            "embedding": embedding,
            "I_norm": I_norm
        }
        return data, N, edge_index.size(1)
    
    # If we exhaust max attempts, raise error
    raise RuntimeError(f"Failed to generate valid sample after {MAX_ATTEMPTS} attempts")

# Main generation loop
all_N = []
all_deg = []
for i in range(NUM_SAMPLES):
    data, N, num_edges = generate_sample()
    torch.save(data, os.path.join(DATA_DIR, f"{i+1:04d}.pt"))
    all_N.append(N)
    all_deg.append(num_edges / N)
    if (i+1) % 100 == 0:
        print(f"Generated {i+1}/{NUM_SAMPLES}")

summary = {
    "num_samples": NUM_SAMPLES,
    "avg_nodes": float(np.mean(all_N)),
    "avg_degree": float(np.mean(all_deg)),
    "description": "Fully synthetic rumor diffusion dataset using heterogeneous SIR on BA networks with BERT embeddings"
}
with open(SUMMARY_PATH, "w") as f:
    json.dump(summary, f, indent=2)
print(f"Saved summary to {SUMMARY_PATH}")
