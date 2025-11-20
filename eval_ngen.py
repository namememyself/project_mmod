import os
import argparse
import json
import numpy as np
import torch
import matplotlib.pyplot as plt

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


def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds = PhemeSyntheticDataset(root=args.root)
    files = ds.files
    # load model
    ckpt = torch.load(args.checkpoint, map_location=device)
    sample = load_thread(files[0])
    emb_dim = sample['embedding'].numel()
    textenc = TextEnc(in_dim=emb_dim, out_dim=32).to(device)
    nodemlp = NodeMLP(in_dim=3, out_dim=32).to(device)
    gnn = GNNBlock(in_dim=33, out_dim=32).to(device)
    paramnet = ParamNet(h_dim=32, c_dim=32).to(device)
    odeint = ODEIntegrator(gnn, paramnet, dt=args.dt).to(device)
    corrector = GRUCorrector(node_dim=32, hidden_dim=32).to(device)
    textenc.load_state_dict(ckpt['textenc'])
    nodemlp.load_state_dict(ckpt['nodemlp'])
    odeint.load_state_dict(ckpt['odeint'])
    corrector.load_state_dict(ckpt['corrector'])
    textenc.eval(); nodemlp.eval(); odeint.eval(); corrector.eval()

    metrics = {'rmse_list': [], 'mae_final': [], 'tpe': []}
    preds = []
    trues = []

    for path in files:
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
        S0 = torch.ones(N, device=device)
        I0 = torch.zeros(N, device=device)
        # Use true I(0) as initial infected
        infected_count = int(data['I_norm'][0].item() * N)
        if infected_count > 0:
            I0[:infected_count] = 1.0
        R0 = torch.zeros(N, device=device)
        # skip samples with empty observed curve
        if len(data['I_norm']) == 0:
            continue
        node_h = nodemlp(node_feat)
        I_ode_seq, beta, gamma = odeint(S0, I0, R0, list(range(len(data['I_norm']))), node_h, edge_index=edge_index, edge_weight=edge_weight, adj_mat=adj, content_c=c)
        pooled = I_ode_seq.mean(dim=1)
        I_corr, corr = corrector(I_ode_seq, pooled)
        true_seq = data['I_norm'].numpy()
        pred_seq = I_corr.cpu().detach().numpy()
        # aggregate predicted curve across nodes to match true_seq shape
        pred_agg = pred_seq.mean(axis=1)  # (T,)
        # metrics
        rmse = np.sqrt(((pred_agg - true_seq)**2).mean())
        R_pred = pred_agg.sum()
        R_true = true_seq.sum()
        mae_final = abs(R_pred - R_true)
        t_pred = int(pred_agg.argmax()) * args.dt
        t_true = int(np.argmax(true_seq)) * args.dt
        tpe = abs(t_pred - t_true)
        metrics['rmse_list'].append(rmse)
        metrics['mae_final'].append(mae_final)
        metrics['tpe'].append(tpe)
        preds.append(R_pred)
        trues.append(R_true)

    print('RMSE_curve:', float(np.mean(metrics['rmse_list'])))
    print('MAE_final:', float(np.mean(metrics['mae_final'])))
    print('time_to_peak_error:', float(np.mean(metrics['tpe'])))

    # scatter
    plt.figure(figsize=(6,6))
    plt.scatter(trues, preds, alpha=0.6)
    plt.xlabel('R_true^inf'); plt.ylabel('R_pred^inf'); plt.plot([0,1],[0,1], '--', color='k')
    plt.title('Final spread: predicted vs true')
    plt.savefig('final_spread_scatter.png')
    print('Saved scatter to final_spread_scatter.png')

    # save metrics
    with open('eval_metrics.json','w',encoding='utf8') as f:
        json.dump({'rmse': float(np.mean(metrics['rmse_list'])), 'mae_final': float(np.mean(metrics['mae_final'])), 'tpe': float(np.mean(metrics['tpe']))}, f)
    print('Saved eval_metrics.json')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='pheme_synthetic')
    p.add_argument('--checkpoint', default='ngen_checkpoint.pth')
    p.add_argument('--dt', type=float, default=0.5)
    args = p.parse_args()
    evaluate(args)
