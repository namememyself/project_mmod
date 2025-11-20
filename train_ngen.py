import os
import argparse
import random
import time
import json
import math

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from ngen_dataset import PhemeSyntheticDataset, load_thread
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


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds = PhemeSyntheticDataset(root=args.root)
    files = ds.files
    random.shuffle(files)
    n = len(files)
    ntrain = int(n*0.7)
    nval = int(n*0.15)
    train_files = files[:ntrain]
    val_files = files[ntrain:ntrain+nval]
    test_files = files[ntrain+nval:]


    # build model
    sample = load_thread(train_files[0])
    emb_dim = sample['embedding'].numel()
    textenc = TextEnc(in_dim=emb_dim, out_dim=32).to(device)
    nodemlp = NodeMLP(in_dim=3, out_dim=32).to(device)
    gnn = GNNBlock(in_dim=33, out_dim=32).to(device)  # node feat + I
    paramnet = ParamNet(h_dim=32, c_dim=32).to(device)
    odeint = ODEIntegrator(gnn, paramnet, dt=args.dt).to(device)
    corrector = GRUCorrector(node_dim=32, hidden_dim=32).to(device)

    params = list(textenc.parameters()) + list(nodemlp.parameters()) + list(odeint.parameters()) + list(corrector.parameters())
    opt = optim.Adam(params, lr=args.lr, weight_decay=1e-5)

    # --- Load weights from latest checkpoint if present ---
    ckpt_path = 'ngen_checkpoint_cpu.pth'
    if os.path.exists(ckpt_path):
        print(f"Loading weights from {ckpt_path} before training...")
        ckpt = torch.load(ckpt_path, map_location=device)
        textenc.load_state_dict(ckpt['textenc'])
        nodemlp.load_state_dict(ckpt['nodemlp'])
        odeint.load_state_dict(ckpt['odeint'])
        corrector.load_state_dict(ckpt['corrector'])

    best_val = 1e9
    patience = 10
    no_improve = 0

    def process_one(path):
        data = load_thread(path)
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
        emb = data['embedding'].to(device)
        c = textenc(emb).squeeze(0)
        # initial S,I,R
        T = data['I_norm'].shape[0]
        if T == 0:
            return None
        I0 = torch.zeros(N, dtype=torch.float32, device=device)
        # Use true I(0) as initial infected
        infected_count = int(data['I_norm'][0].item() * N)
        if infected_count > 0:
            I0[:infected_count] = 1.0
        S0 = torch.ones(N, device=device) - I0
        R0 = torch.zeros(N, device=device)
        I_norm = data['I_norm'].to(device)
        return S0, I0, R0, I_norm, node_feat, edge_index, edge_weight, adj, c, N

    def compute_loss(pred_seq, true_seq, beta, gamma, lam1=1.0, lam2=2.0, lam3=1e-5):
        # pred_seq: (T, N) ; true_seq: (T,)
        pred_agg = pred_seq.mean(dim=1)  # (T,)
        # ensure same device/dtype
        true_seq = true_seq.to(pred_agg.device)
        rmse = torch.sqrt(((pred_agg - true_seq)**2).mean())
        R_pred = pred_agg.sum()
        R_true = true_seq.sum()
        mae_final = torch.abs(R_pred - R_true)
        reg = (beta.mean()**2 + gamma.mean()**2)
        loss = lam1 * rmse + lam2 * mae_final + lam3 * reg
        return loss, rmse.item(), mae_final.item()

    for epoch in range(args.epochs):
        t0 = time.time()
        random.shuffle(train_files)
        textenc.train(); nodemlp.train(); odeint.train(); corrector.train()
        running_loss = 0.0
        for i, path in enumerate(train_files):
            res = process_one(path)
            if res is None:
                continue
            S0, I0, R0, I_norm, node_feat_raw, edge_index, edge_weight, adj, c, N = res
            node_feat = nodemlp(node_feat_raw.to(device))
            opt.zero_grad()
            I_ode_seq, beta, gamma = odeint(S0, I0, R0, list(range(len(I_norm))), node_feat, edge_index=edge_index, edge_weight=edge_weight, adj_mat=adj, content_c=c)
            # Debug: check for NaNs/Infs in model outputs
            if torch.isnan(I_ode_seq).any() or torch.isinf(I_ode_seq).any():
                print(f"[NaN/Inf] I_ode_seq in sample {path}")
                continue
            if torch.isnan(beta).any() or torch.isinf(beta).any():
                print(f"[NaN/Inf] beta in sample {path}")
                continue
            if torch.isnan(gamma).any() or torch.isinf(gamma).any():
                print(f"[NaN/Inf] gamma in sample {path}")
                continue
            pooled_h_seq = I_ode_seq.mean(dim=1)
            I_corr, corr = corrector(I_ode_seq, pooled_h_seq)
            true_seq = I_norm.to(device)
            loss, rmse, mae_final = compute_loss(I_corr, true_seq, beta, gamma)
            # Debug: check for NaNs/Infs in loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"[NaN/Inf] loss in sample {path}")
                print(f"  loss={loss}, rmse={rmse}, mae_final={mae_final}")
                print(f"  true_seq: min={true_seq.min().item()}, max={true_seq.max().item()}, mean={true_seq.mean().item()}")
                print(f"  I_corr: min={I_corr.min().item()}, max={I_corr.max().item()}, mean={I_corr.mean().item()}")
                continue
            loss.backward()
            # Debug: check for NaNs/Infs in gradients
            nan_grad = False
            for name, param in list(textenc.named_parameters()) + list(nodemlp.named_parameters()) + list(odeint.named_parameters()) + list(corrector.named_parameters()):
                if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                    print(f"[NaN/Inf] grad in {name} for sample {path}")
                    nan_grad = True
            if nan_grad:
                continue
            # Add gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
            opt.step()
            running_loss += loss.item()
        # validation
        textenc.eval(); nodemlp.eval(); odeint.eval(); corrector.eval()
        val_losses = []
        with torch.no_grad():
            for path in val_files:
                res = process_one(path)
                if res is None:
                    continue
                S0, I0, R0, I_norm, node_feat_raw, edge_index, edge_weight, adj, c, N = res
                node_feat = nodemlp(node_feat_raw.to(device))
                I_ode_seq, beta, gamma = odeint(S0, I0, R0, list(range(len(I_norm))), node_feat, edge_index=edge_index, edge_weight=edge_weight, adj_mat=adj, content_c=c)
                pooled_h_seq = I_ode_seq.mean(dim=1)
                I_corr, corr = corrector(I_ode_seq, pooled_h_seq)
                loss, rmse, mae_final = compute_loss(I_corr, I_norm.to(device), beta, gamma)
                val_losses.append(rmse)
        val_rmse = float(np.mean(val_losses)) if val_losses else 1e9
        print(f'Epoch {epoch+1}/{args.epochs} train_loss={running_loss/len(train_files):.4f} val_rmse={val_rmse:.4f} time={(time.time()-t0):.1f}s')
        if val_rmse < best_val:
            best_val = val_rmse
            no_improve = 0
            torch.save({'textenc': textenc.state_dict(), 'nodemlp': nodemlp.state_dict(), 'odeint': odeint.state_dict(), 'corrector': corrector.state_dict()}, args.save)
        else:
            no_improve += 1
        if no_improve >= patience:
            print('Early stopping')
            break

    print('Training complete. Best val RMSE:', best_val)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='pheme_synthetic')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--dt', type=float, default=0.5)
    parser.add_argument('--save', default='ngen_checkpoint.pth')
    args = parser.parse_args()
    train(args)
