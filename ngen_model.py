import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import warnings

try:
    from torch_geometric.nn import SAGEConv, GATConv
    has_pyg = True
except Exception:
    has_pyg = False
    warnings.warn('PyTorch Geometric not available; using adjacency-mean fallback GNN.')

class TextEnc(nn.Module):
    def __init__(self, in_dim, out_dim=32):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim),
            nn.Tanh()
        )
        # Xavier init
        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, emb):
        # emb: (D,) or (D,) tensor
        if emb.dim()==1:
            emb = emb.unsqueeze(0)
        # Clamp to avoid extreme inputs
        emb = torch.clamp(emb, -1e3, 1e3)
        return self.fc(emb)

class NodeMLP(nn.Module):
    def __init__(self, in_dim=3, out_dim=32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim)
        )
        # Xavier init
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, x):
        # x: (N, 3)
        # Clamp to avoid extreme inputs
        x = torch.clamp(x, -1e3, 1e3)
        return self.mlp(x)

class GNNBlock(nn.Module):
    def __init__(self, in_dim, out_dim, agg='sage'):
        super().__init__()
        self.agg = agg
        self.in_dim = in_dim
        self.out_dim = out_dim
        if has_pyg:
            if agg=='gat':
                self.conv1 = GATConv(in_dim, out_dim//2, heads=2)
                self.conv2 = GATConv(out_dim, out_dim//2, heads=2)
            else:
                self.conv1 = SAGEConv(in_dim, out_dim)
                self.conv2 = SAGEConv(out_dim, out_dim)
        else:
            # fallback: stack 2 linear layers after neighbor mean
            self.lin1 = nn.Linear(in_dim, out_dim)
            self.lin2 = nn.Linear(out_dim, out_dim)

    def forward(self, x, edge_index=None, edge_weight=None, adj_mat=None):
        # x: (N, D)
        # Clamp input to avoid extreme values
        x = torch.clamp(x, -1e3, 1e3)
        if has_pyg:
            h = F.relu(self.conv1(x, edge_index))
            # Clamp after each layer to prevent explosion
            h = torch.clamp(h, -1e3, 1e3)
            h = F.relu(self.conv2(h, edge_index))
            h = torch.clamp(h, -1e3, 1e3)
            return h
        else:
            # adj_mat: (N,N) dense tensor
            if adj_mat is None:
                neigh = torch.zeros_like(x)
            else:
                deg = adj_mat.sum(dim=1, keepdim=True).clamp(min=1e-8)
                neigh = (adj_mat @ x) / deg
                neigh = torch.clamp(neigh, -1e3, 1e3)
            h = x + neigh
            h = torch.clamp(h, -1e3, 1e3)
            h = F.relu(self.lin1(h))
            h = torch.clamp(h, -1e3, 1e3)
            h = F.relu(self.lin2(h))
            h = torch.clamp(h, -1e3, 1e3)
            return h

class ParamNet(nn.Module):
    def __init__(self, h_dim, c_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(h_dim + c_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
        # Xavier init for layers
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        # Initialize final layer bias to produce reasonable rates
        self.net[-1].bias.data = torch.tensor([-2.0, -2.5], dtype=torch.float32)

    def forward(self, h, c):
        # h: (N, h_dim), c: (1, c_dim) or (c_dim,)
        if c.dim()==1:
            c = c.unsqueeze(0)
        c_rep = c.expand(h.size(0), -1)
        inp = torch.cat([h, c_rep], dim=-1)
        # Clamp input
        inp = torch.clamp(inp, -1e3, 1e3)
        out = self.net(inp)
        # Clamp before softplus to avoid overflow
        out = torch.clamp(out, -10, 10)
        # enforce positivity via softplus + eps
        out = F.softplus(out) + 1e-6
        # Scale to reasonable ranges: beta in [1e-4, 0.1], gamma in [1e-4, 0.1]
        beta = torch.clamp(out[:,0:1] * 0.1, min=1e-6, max=0.1)
        gamma = torch.clamp(out[:,1:2] * 0.05, min=1e-6, max=0.1)
        return beta.squeeze(-1), gamma.squeeze(-1)

class ODEIntegrator(nn.Module):
    def __init__(self, gnn_block, param_net, dt=0.5, use_diffeq=False):
        super().__init__()
        self.gnn = gnn_block
        self.param_net = param_net
        self.dt = dt
        self.use_diffeq = use_diffeq
        try:
            from torchdiffeq import odeint
            self.odeint = odeint
            self.has_diffeq = True
        except Exception:
            self.has_diffeq = False

    def forward(self, S0, I0, R0, time_grid, node_feat, edge_index=None, edge_weight=None, adj_mat=None, content_c=None):
        # S0,I0,R0: (N,)
        T = len(time_grid)
        N = S0.size(0)
        S = S0.clone()
        I = I0.clone()
        R = R0.clone()
        I_seq = []
        for t_idx in range(T):
            # prepare node inputs: concatenate base node features with current I
            x_in = torch.cat([node_feat, I.unsqueeze(-1)], dim=-1)
            h = self.gnn(x_in, edge_index=edge_index, edge_weight=edge_weight, adj_mat=adj_mat)
            beta, gamma = self.param_net(h, content_c)
            # compute force: A_ij * I_j sum
            if adj_mat is not None:
                force = (adj_mat * I.unsqueeze(0)).sum(dim=1)
            elif edge_index is not None and edge_index.numel()>0:
                # build dense product (slow) or aggregate edges manually
                src = edge_index[0]
                dst = edge_index[1]
                w = edge_weight if edge_weight is not None else torch.ones_like(src, dtype=torch.float32)
                force = torch.zeros(N, dtype=I.dtype, device=I.device)
                force = force.index_add(0, dst, w * I[src])
            else:
                force = torch.zeros_like(I)

            # Clamp force to avoid explosion
            force = torch.clamp(force, 0, 1e3)
            dS = - beta * S * force
            dI = beta * S * force - gamma * I
            dR = gamma * I

            S = S + dS * self.dt
            I = I + dI * self.dt
            R = R + dR * self.dt
            # Clamp to [0, 1] after each step
            S = torch.clamp(S, min=0.0, max=1.0)
            I = torch.clamp(I, min=0.0, max=1.0)
            R = torch.clamp(R, min=0.0, max=1.0)
            I_seq.append(I.unsqueeze(0))

        I_seq = torch.cat(I_seq, dim=0)  # (T, N)
        return I_seq, beta, gamma

class GRUCorrector(nn.Module):
    def __init__(self, node_dim, hidden_dim=32):
        super().__init__()
        # input per time-step: [mean_I, pooled_h]
        self.gru = nn.GRU(input_size=2, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())

    def forward(self, I_ode_seq, pooled_h_seq):
        # I_ode_seq: (T, N) ; pooled_h_seq: (T, h_dim) or (T,)
        T, N = I_ode_seq.shape
        mean_I = I_ode_seq.mean(dim=1, keepdim=True)  # (T,1)
        if pooled_h_seq is None:
            ph = torch.zeros((T,1), device=I_ode_seq.device)
        else:
            ph = pooled_h_seq.view(T,1)
        inp = torch.cat([mean_I, ph], dim=1).unsqueeze(0)  # (1, T, 2)
        out, _ = self.gru(inp)
        corr = self.fc(out).squeeze(0).squeeze(-1)  # (T,)
        # apply correction multiplicatively to I_ode_seq across nodes
        I_corr = I_ode_seq * corr.unsqueeze(1)
        return I_corr, corr
