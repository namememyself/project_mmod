import os
import json
import random
import numpy as np
import torch

class PhemeSyntheticDataset:
    def __init__(self, root='pheme_synthetic', split='all', seed=42):
        self.root = root
        # Support both .json and .pt files
        files = [os.path.join(root, f) for f in os.listdir(root) if f.endswith('.json') or f.endswith('.pt')]
        random.seed(seed)
        self.files = sorted(files)

    def load_all(self):
        items = []
        for f in self.files:
            if f.endswith('.json'):
                with open(f, 'r', encoding='utf8') as fh:
                    j = json.load(fh)
                N = j.get('network', {}).get('N', j.get('N', None))
                I_time = np.array(j.get('I_time', j.get('I_time', [])), dtype=float)
                if N is None or N <= 0:
                    N = float(j.get('N', max(1, len(j.get('node_features', [])))))
                I_norm = (I_time / float(N)).astype(float)
                items.append({'json': j, 'I_norm': I_norm, 'N': int(N)})
            elif f.endswith('.pt'):
                d = torch.load(f)
                N = int(d['N']) if 'N' in d else len(d['followers'])
                I_norm = d['I_norm'].cpu().numpy() if 'I_norm' in d else np.zeros(1)
                items.append({'pt': d, 'I_norm': I_norm, 'N': N})
        return items

    def split(self, ratios=(0.7,0.15,0.15), seed=42):
        files = self.files.copy()
        random.Random(seed).shuffle(files)
        n = len(files)
        n1 = int(n * ratios[0])
        n2 = int(n * (ratios[0]+ratios[1]))
        return files[:n1], files[n1:n2], files[n2:]

def load_thread(path):
    if path.endswith('.pt'):
        d = torch.load(path)
        # Use directly as expected by the model
        return {
            'edge_index': d['edge_index'],
            'edge_weight': d['edge_weight'],
            'followers': d['followers'],
            'activity': d['activity'],
            'susceptibility': d['susceptibility'],
            'embedding': d['embedding'],
            'I_norm': d['I_norm'],
            'N': int(d['N'])
        }
    else:
        with open(path, 'r', encoding='utf8') as f:
            j = json.load(f)
        N = j.get('network', {}).get('N', j.get('N', None))
        if N is None:
            N = max(1, len(j.get('node_features', [])))
        I_time = np.array(j.get('I_time', []), dtype=float)
        I_norm = I_time / float(N)
        # edges: may be (u,v) or (u,v,w)
        edges = j.get('network', {}).get('edges', j.get('edges', []))
        edge_index = []
        edge_weight = []
        for e in edges:
            if len(e) >= 3:
                u,v,w = e[0],e[1],e[2]
            else:
                u,v = e[0],e[1]
                w = 1.0
            edge_index.append([int(u), int(v)])
            edge_weight.append(float(w))
        if len(edge_index)==0:
            edge_index = torch.zeros((2,0), dtype=torch.long)
            edge_weight = torch.tensor([], dtype=torch.float32)
        else:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_weight = torch.tensor(edge_weight, dtype=torch.float32)

        # node features
        node_feats = j.get('node_features', [])
        # create array ordered by node id
        max_node = max([nf.get('node',0) for nf in node_feats]) if node_feats else -1
        N_nodes = max(max_node+1, int(N))
        followers = np.zeros(N_nodes, dtype=float)
        activity = np.zeros(N_nodes, dtype=float)
        susceptibility = np.zeros(N_nodes, dtype=float)
        for nf in node_feats:
            i = int(nf.get('node',0))
            followers[i] = float(nf.get('followers',0.0))
            activity[i] = float(nf.get('activity',0.0))
            susceptibility[i] = float(nf.get('susceptibility',0.0))

        content = j.get('content', {})
        embedding = torch.tensor(content.get('embedding', []), dtype=torch.float32)
        sentiment = float(content.get('sentiment', 0.0))
        length = int(content.get('length', 0)) if 'length' in content else len(content.get('embedding', []))

        return {
            'edge_index': edge_index,
            'edge_weight': torch.tensor(edge_weight, dtype=torch.float32),
            'followers': torch.tensor(followers, dtype=torch.float32),
            'activity': torch.tensor(activity, dtype=torch.float32),
            'susceptibility': torch.tensor(susceptibility, dtype=torch.float32),
            'embedding': embedding,
            'sentiment': sentiment,
            'length': length,
            'I_norm': torch.tensor(I_norm, dtype=torch.float32),
            'N': int(N)
        }
