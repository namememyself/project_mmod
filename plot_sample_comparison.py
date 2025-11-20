#!/usr/bin/env python3
"""
Plot I(t) curves for a random test sample comparing NGEN and baseline models.
"""

import os
import sys
import json
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from pathlib import Path

from ngen_dataset import load_thread, PhemeSyntheticDataset
from ngen_model import TextEnc, NodeMLP, GNNBlock, ParamNet, ODEIntegrator, GRUCorrector


def build_adj_matrix(edge_index, edge_weight, N):
    if edge_index.numel()==0:
        return None
    idx = edge_index.long()
    mat = torch.zeros((N,N), dtype=torch.float32)
    src = idx[0]
    dst = idx[1]
    w = edge_weight
    mat[dst, src] = w
    return mat


def global_sir_baseline(N, initial_infected, beta, gamma, T, adj=None):
    """Simple global SIR without network structure."""
    S = np.ones(T)
    I = np.zeros(T)
    R = np.zeros(T)
    I[0] = initial_infected / N
    S[0] = 1.0 - I[0]
    
    for t in range(1, T):
        new_inf = beta.mean() * S[t-1] * I[t-1]
        new_rec = gamma.mean() * I[t-1]
        S[t] = np.clip(S[t-1] - new_inf, 0, 1)
        I[t] = np.clip(I[t-1] + new_inf - new_rec, 0, 1)
        R[t] = np.clip(R[t-1] + new_rec, 0, 1)
    
    return I


def degree_weighted_sir_baseline(N, initial_infected, beta, gamma, T, adj=None):
    """SIR with degree-weighted transmission."""
    S = np.ones(T)
    I = np.zeros(T)
    R = np.zeros(T)
    I[0] = initial_infected / N
    S[0] = 1.0 - I[0]
    
    # Compute degree weights if adj provided
    if adj is not None:
        adj_np = adj.cpu().numpy() if isinstance(adj, torch.Tensor) else adj
        degrees = adj_np.sum(axis=1)
        avg_degree = degrees.mean()
    else:
        avg_degree = 1.0
    
    for t in range(1, T):
        new_inf = beta.mean() * avg_degree * S[t-1] * I[t-1]
        new_rec = gamma.mean() * I[t-1]
        S[t] = np.clip(S[t-1] - new_inf, 0, 1)
        I[t] = np.clip(I[t-1] + new_inf - new_rec, 0, 1)
        R[t] = np.clip(R[t-1] + new_rec, 0, 1)
    
    return I


def constant_baseline(N, initial_infected, T):
    """Dummy: constant infected proportion."""
    return np.ones(T) * (initial_infected / N)


def run_ngen_inference(data, model_dict, device):
    """Run NGEN inference on a single sample."""
    textenc, nodemlp, gnn, paramnet, odeint, corrector = model_dict
    
    N = data['N']
    edge_index = data['edge_index'].to(device)
    edge_weight = data['edge_weight'].to(device)
    adj = build_adj_matrix(edge_index, edge_weight, N)
    if adj is not None:
        adj = adj.to(device)
    
    followers = data['followers'].unsqueeze(1).to(device)
    activity = data['activity'].unsqueeze(1).to(device)
    susceptibility = data['susceptibility'].unsqueeze(1).to(device)
    node_feat = torch.cat([followers, activity, susceptibility], dim=1)
    
    # Handle embedding
    emb = data['embedding']
    if isinstance(emb, torch.Tensor):
        emb = emb.to(device)
    else:
        emb = torch.tensor(emb, dtype=torch.float32, device=device)
    
    with torch.no_grad():
        if emb.dim() == 1:
            emb = emb.unsqueeze(0)
        emb = torch.clamp(emb, -1e3, 1e3)
        c = textenc(emb)
        if c.dim() > 1:
            c = c.squeeze(0)
        
        node_feat = nodemlp(node_feat)
        
        S0 = torch.ones(N, device=device)
        I0 = torch.zeros(N, device=device)
        infected_count = int(data['I_norm'][0].item() * N) if len(data['I_norm']) > 0 else 1
        if infected_count > 0:
            I0[:infected_count] = 1.0
        S0 = torch.ones(N, device=device) - I0
        R0 = torch.zeros(N, device=device)
        
        T_steps = len(data['I_norm'])
        I_ode_seq, beta, gamma = odeint(S0, I0, R0, list(range(T_steps)), node_feat, 
                                        edge_index=edge_index, edge_weight=edge_weight, 
                                        adj_mat=adj, content_c=c)
        
        pooled_h_seq = I_ode_seq.mean(dim=1)
        I_corr, corr = corrector(I_ode_seq, pooled_h_seq)
        
        pred = I_corr.mean(dim=1).cpu().numpy()
    
    return pred


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Load synthetic dataset
    ds = PhemeSyntheticDataset(root='synthetic')
    files = ds.files
    random.shuffle(files)
    
    n = len(files)
    test_size = int(n * 0.15)
    test_files = files[:test_size]
    
    # Pick a random test sample
    sample_idx = random.randint(0, len(test_files) - 1)
    sample_path = test_files[sample_idx]
    print(f"Selected random test sample: {sample_path}")
    print(f"(Index {sample_idx} of {len(test_files)} test samples)\n")
    
    # Load NGEN model
    print("=== Loading NGEN Model ===\n")
    ckpt_path = 'ngen_checkpoint_cpu.pth'
    if not os.path.exists(ckpt_path):
        print(f"ERROR: Checkpoint not found at {ckpt_path}")
        return
    
    ckpt = torch.load(ckpt_path, map_location=device)
    sample = load_thread(files[0])
    emb_dim = sample['embedding'].shape[0] if isinstance(sample['embedding'], torch.Tensor) else len(sample['embedding'])
    
    textenc = TextEnc(in_dim=emb_dim, out_dim=32).to(device)
    nodemlp = NodeMLP(in_dim=3, out_dim=32).to(device)
    gnn = GNNBlock(in_dim=33, out_dim=32).to(device)
    paramnet = ParamNet(h_dim=32, c_dim=32).to(device)
    odeint = ODEIntegrator(gnn, paramnet, dt=0.5).to(device)
    corrector = GRUCorrector(node_dim=32, hidden_dim=32).to(device)
    
    try:
        textenc.load_state_dict(ckpt['textenc'])
        nodemlp.load_state_dict(ckpt['nodemlp'])
        odeint.load_state_dict(ckpt['odeint'])
        corrector.load_state_dict(ckpt['corrector'])
        print("✓ Model weights loaded successfully\n")
    except Exception as e:
        print(f"ERROR loading model weights: {e}")
        return
    
    textenc.eval()
    nodemlp.eval()
    odeint.eval()
    corrector.eval()
    
    model_dict = (textenc, nodemlp, gnn, paramnet, odeint, corrector)
    
    # Load the sample
    print("=== Loading Sample Data ===\n")
    data = load_thread(sample_path)
    true_seq = data['I_norm'].numpy()
    
    N = data['N']
    edge_index = data['edge_index']
    edge_weight = data['edge_weight']
    adj = build_adj_matrix(edge_index, edge_weight, N)
    beta = data.get('beta', np.ones(N) * 0.05)
    gamma = data.get('gamma', np.ones(N) * 0.05)
    
    if isinstance(beta, torch.Tensor):
        beta = beta.numpy()
    if isinstance(gamma, torch.Tensor):
        gamma = gamma.numpy()
    
    initial_infected = int(true_seq[0] * N)
    T = len(true_seq)
    
    print(f"Sample Statistics:")
    print(f"  N (nodes): {N}")
    print(f"  T (timesteps): {T}")
    print(f"  Initial infected: {initial_infected} ({initial_infected/N*100:.2f}%)")
    print(f"  Peak infected: {true_seq.max()*100:.2f}% at t={np.argmax(true_seq)}")
    print(f"  Final infected: {true_seq[-1]*100:.2f}%")
    print(f"  Avg beta: {beta.mean():.6f}")
    print(f"  Avg gamma: {gamma.mean():.6f}\n")
    
    # Run baselines
    print("=== Running Baselines ===\n")
    gs_pred = global_sir_baseline(N, initial_infected, beta, gamma, T, adj)
    ds_pred = degree_weighted_sir_baseline(N, initial_infected, beta, gamma, T, adj)
    const_pred = constant_baseline(N, initial_infected, T)
    
    print("✓ GlobalSIR")
    print("✓ DegreeSIR")
    print("✓ Constant\n")
    
    # Run NGEN
    print("=== Running NGEN ===\n")
    try:
        ngen_pred = run_ngen_inference(data, model_dict, device)
        ngen_pred = np.clip(ngen_pred, 0, 1)
        print("✓ NGEN inference complete\n")
    except Exception as e:
        print(f"ERROR in NGEN inference: {e}")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))
    
    time_steps = np.arange(T)
    
    # Plot predictions
    ax.plot(time_steps, true_seq, 'ko-', linewidth=3, markersize=6, label='Ground Truth', zorder=5)
    ax.plot(time_steps, ngen_pred, 's-', linewidth=2.5, markersize=5, label='NGEN', color='#2E7D32', zorder=4)
    ax.plot(time_steps, ds_pred, '^-', linewidth=2, markersize=4.5, label='Degree-weighted SIR', color='#1976D2', zorder=3)
    ax.plot(time_steps, gs_pred, 'v-', linewidth=2, markersize=4.5, label='Global SIR', color='#D32F2F', zorder=2)
    ax.plot(time_steps, const_pred, 'D-', linewidth=1.5, markersize=4, label='Constant', color='#FF6F00', zorder=1, alpha=0.7)
    
    ax.set_xlabel('Time (hours)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Infected Proportion I(t)', fontsize=12, fontweight='bold')
    ax.set_title(f'Rumor Cascade I(t) Prediction Comparison\n(N={N}, T={T}, Initial Infected={initial_infected})', 
                 fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=11, framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim([0, max(true_seq.max(), ngen_pred.max(), ds_pred.max()) * 1.1])
    
    plt.tight_layout()
    
    # Save figure
    fig_path = 'sample_prediction_comparison.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"✓ Figure saved to {fig_path}\n")
    
    # Compute and display metrics
    print("="*60)
    print("PREDICTION METRICS FOR THIS SAMPLE")
    print("="*60 + "\n")
    
    def rmse(pred, true):
        return np.sqrt(np.mean((pred - true) ** 2))
    
    def mae_final(pred, true):
        return np.abs(pred[-1] - true[-1])
    
    def mae_all(pred, true):
        return np.mean(np.abs(pred - true))
    
    models = {
        'NGEN': ngen_pred,
        'Degree-SIR': ds_pred,
        'Global-SIR': gs_pred,
        'Pure GNN': const_pred
    }
    
    for model_name, pred in models.items():
        r = rmse(pred, true_seq)
        mae_f = mae_final(pred, true_seq)
        mae_a = mae_all(pred, true_seq)
        print(f"{model_name:15} | RMSE: {r:.6f} | MAE(final): {mae_f:.6f} | MAE(all): {mae_a:.6f}")
    
    plt.show()


if __name__ == '__main__':
    main()
