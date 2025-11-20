#!/usr/bin/env python3
"""
Run baseline models on synthetic dataset and compare with NGEN.
Baselines: GlobalSIR, DegreeSIR, RandomSIR
"""

import os
import sys
import json
import numpy as np
import torch
import random
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


def compute_metrics(pred, true, rmse_weight=1.0, mae_weight=1.0):
    """Compute RMSE and MAE."""
    rmse = np.sqrt(np.mean((pred - true) ** 2))
    mae_final = np.abs(pred[-1] - true[-1])
    return rmse, mae_final


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load synthetic dataset
    ds = PhemeSyntheticDataset(root='synthetic')
    files = ds.files
    random.shuffle(files)
    
    n = len(files)
    test_size = int(n * 0.15)
    test_files = files[:test_size]
    
    print(f"Testing on {len(test_files)} samples from {n} total")
    
    # Load NGEN model
    print("\n=== Loading NGEN Model ===")
    ckpt_path = 'ngen_checkpoint_cpu.pth'
    if not os.path.exists(ckpt_path):
        print(f"ERROR: Checkpoint not found at {ckpt_path}")
        return
    
    ckpt = torch.load(ckpt_path, map_location=device)
    sample = load_thread(files[0])
    emb_dim = sample['embedding'].shape[0] if isinstance(sample['embedding'], torch.Tensor) else len(sample['embedding'])
    
    print(f"Embedding dim: {emb_dim}")
    
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
        print("✓ Model weights loaded successfully")
    except Exception as e:
        print(f"ERROR loading model weights: {e}")
        return
    
    textenc.eval()
    nodemlp.eval()
    odeint.eval()
    corrector.eval()
    
    model_dict = (textenc, nodemlp, gnn, paramnet, odeint, corrector)
    
    # Initialize result storage
    results = {
        'global_sir': {'rmse': [], 'mae_final': []},
        'degree_sir': {'rmse': [], 'mae_final': []},
        'constant': {'rmse': [], 'mae_final': []},
        'ngen': {'rmse': [], 'mae_final': []}
    }
    
    print(f"\n=== Evaluating {len(test_files)} Test Samples ===\n")
    
    for idx, path in enumerate(test_files):
        try:
            data = load_thread(path)
            true_seq = data['I_norm'].numpy()
            
            if len(true_seq) == 0:
                continue
            
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
            
            # Run baselines
            gs_pred = global_sir_baseline(N, initial_infected, beta, gamma, T, adj)
            ds_pred = degree_weighted_sir_baseline(N, initial_infected, beta, gamma, T, adj)
            const_pred = constant_baseline(N, initial_infected, T)
            
            # Run NGEN
            try:
                ngen_pred = run_ngen_inference(data, model_dict, device)
                ngen_pred = np.clip(ngen_pred, 0, 1)
            except Exception as e:
                print(f"  NGEN inference failed on sample {idx}: {e}")
                ngen_pred = np.ones(T) * 0.5  # fallback
            
            # Compute metrics
            gs_rmse, gs_mae = compute_metrics(gs_pred, true_seq)
            ds_rmse, ds_mae = compute_metrics(ds_pred, true_seq)
            const_rmse, const_mae = compute_metrics(const_pred, true_seq)
            ngen_rmse, ngen_mae = compute_metrics(ngen_pred, true_seq)
            
            results['global_sir']['rmse'].append(gs_rmse)
            results['global_sir']['mae_final'].append(gs_mae)
            results['degree_sir']['rmse'].append(ds_rmse)
            results['degree_sir']['mae_final'].append(ds_mae)
            results['constant']['rmse'].append(const_rmse)
            results['constant']['mae_final'].append(const_mae)
            results['ngen']['rmse'].append(ngen_rmse)
            results['ngen']['mae_final'].append(ngen_mae)
            
            if (idx + 1) % 10 == 0:
                print(f"Processed {idx + 1}/{len(test_files)} samples")
        
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            continue
    
    # Print summary
    print("\n" + "="*80)
    print("BASELINE COMPARISON RESULTS")
    print("="*80 + "\n")
    
    for model_name in ['global_sir', 'degree_sir', 'constant', 'ngen']:
        rmse_vals = results[model_name]['rmse']
        mae_vals = results[model_name]['mae_final']
        
        if len(rmse_vals) > 0:
            print(f"{model_name.upper()}")
            print(f"  RMSE: {np.mean(rmse_vals):.6f} ± {np.std(rmse_vals):.6f}")
            print(f"  MAE:  {np.mean(mae_vals):.6f} ± {np.std(mae_vals):.6f}")
            print()
    
    # Save results
    results_json = {
        'global_sir': {
            'rmse': {'mean': float(np.mean(results['global_sir']['rmse'])), 
                    'std': float(np.std(results['global_sir']['rmse']))},
            'mae_final': {'mean': float(np.mean(results['global_sir']['mae_final'])), 
                         'std': float(np.std(results['global_sir']['mae_final']))}
        },
        'degree_sir': {
            'rmse': {'mean': float(np.mean(results['degree_sir']['rmse'])), 
                    'std': float(np.std(results['degree_sir']['rmse']))},
            'mae_final': {'mean': float(np.mean(results['degree_sir']['mae_final'])), 
                         'std': float(np.std(results['degree_sir']['mae_final']))}
        },
        'constant': {
            'rmse': {'mean': float(np.mean(results['constant']['rmse'])), 
                    'std': float(np.std(results['constant']['rmse']))},
            'mae_final': {'mean': float(np.mean(results['constant']['mae_final'])), 
                         'std': float(np.std(results['constant']['mae_final']))}
        },
        'ngen': {
            'rmse': {'mean': float(np.mean(results['ngen']['rmse'])), 
                    'std': float(np.std(results['ngen']['rmse']))},
            'mae_final': {'mean': float(np.mean(results['ngen']['mae_final'])), 
                         'std': float(np.std(results['ngen']['mae_final']))}
        },
        'num_samples': len(test_files)
    }
    
    with open('baseline_results.json', 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"Results saved to baseline_results.json")


if __name__ == '__main__':
    main()
