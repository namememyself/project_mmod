import os
import json
import math
from datetime import datetime
from collections import defaultdict

import numpy as np
import networkx as nx

from sklearn.feature_extraction.text import TfidfVectorizer

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except Exception:
    SentimentIntensityAnalyzer = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

PARSER_OUTPUT = 'pheme_parsed_threads.json'
OUTDIR = 'pheme_synthetic'
os.makedirs(OUTDIR, exist_ok=True)

# time grid settings (hours)
TIME_WINDOW = 48.0
DT = 0.5
time_bins = np.arange(0, TIME_WINDOW + 1e-9, DT)

def parse_twitter_date(s):
    # Example: 'Wed Jan 07 11:06:08 +0000 2015'
    try:
        return datetime.strptime(s, '%a %b %d %H:%M:%S %z %Y')
    except Exception:
        try:
            return datetime.strptime(s, '%a %b %d %H:%M:%S +0000 %Y')
        except Exception:
            return None

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

def compute_user_participation(threads):
    freq = defaultdict(int)
    for t in threads:
        # source user
        src = t['source_tweet'].get('user', {}).get('id')
        if src:
            freq[src] += 1
        for r in t['reactions']:
            uid = r.get('user', {}).get('id')
            if uid:
                freq[uid] += 1
    return freq

def main():
    print('Loading parsed threads...')
    with open(PARSER_OUTPUT, encoding='utf8') as f:
        threads = json.load(f)

    # Prepare corpus of source texts for TF-IDF and embeddings
    docs = []
    for t in threads:
        text = t['source_tweet'].get('text', '')
        docs.append(text)

    print('Computing TF-IDF keywords...')
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X = vectorizer.fit_transform(docs)
    feature_names = np.array(vectorizer.get_feature_names_out())

    print('Preparing sentiment analyzer and embedding model (may require installation)...')
    analyzer = SentimentIntensityAnalyzer() if SentimentIntensityAnalyzer is not None else None
    embed_model = SentenceTransformer('all-MiniLM-L6-v2') if SentenceTransformer is not None else None

    user_freq = compute_user_participation(threads)

    outputs = []
    for idx, t in enumerate(threads):
        src = t['source_tweet']
        src_text = src.get('text', '')
        # sentiment
        if analyzer is not None:
            sent = analyzer.polarity_scores(src_text)['compound']
        else:
            sent = 0.0
        # embedding
        if embed_model is not None:
            emb = embed_model.encode(src_text).tolist()
        else:
            emb = []
        # length
        length = len(src_text)
        # keywords: top 5 TF-IDF for this doc
        row = X[idx].toarray().ravel()
        topk_idx = row.argsort()[-5:][::-1]
        keywords = feature_names[topk_idx].tolist()

        # cascade trajectory: replies over time
        src_time = parse_twitter_date(src.get('created_at', ''))
        counts = np.zeros(len(time_bins), dtype=int)
        user_times = {}
        for r in t['reactions']:
            rt = parse_twitter_date(r.get('created_at', ''))
            if rt is None or src_time is None:
                continue
            delta = (rt - src_time).total_seconds() / 3600.0
            if delta < 0 or delta > TIME_WINDOW:
                continue
            bin_idx = int(delta // DT)
            if bin_idx < len(counts):
                counts[bin_idx] += 1
            uid = r.get('user', {}).get('id')
            if uid and uid not in user_times:
                user_times[uid] = delta

        total_replies = sum(counts)
        unique_users = set([src.get('user', {}).get('id')]) | set(user_times.keys())
        n_users = len([u for u in unique_users if u is not None])
        final_spread_ratio = float(total_replies) / max(1, n_users)

        # synthetic network: BA graph
        extra_neighbors = 50
        N_nodes = max(10, n_users + extra_neighbors)
        m = min(5, max(1, N_nodes // 100))
        G = nx.barabasi_albert_graph(N_nodes, m, seed=42 + idx)
        # map real users to nodes
        real_users = [u for u in unique_users if u is not None]
        node_map = {}
        nodes = list(G.nodes())
        np.random.seed(42 + idx)
        assigned = np.random.choice(nodes, size=min(len(real_users), len(nodes)), replace=False).tolist()
        for u, n in zip(real_users, assigned):
            node_map[u] = int(n)

        # compute per-user features for susceptibility
        # reply speed (inverse of time), participation freq, sentiment of reaction
        speeds = []
        freqs = []
        sents = []
        for u in real_users:
            delta = user_times.get(u, TIME_WINDOW)
            speed = 1.0 / (1.0 + delta)
            freq = float(user_freq.get(u, 0))
            # find user's first reaction text to get sentiment
            user_sent = 0.0
            for r in t['reactions']:
                if r.get('user', {}).get('id') == u:
                    if analyzer is not None:
                        user_sent = analyzer.polarity_scores(r.get('text', ''))['compound']
                    break
            speeds.append(speed)
            freqs.append(freq)
            sents.append(user_sent)

        # normalize
        def znormalize(xs):
            a = np.array(xs, dtype=float)
            if a.size == 0:
                return a
            mu = a.mean()
            sd = a.std() if a.std() > 0 else 1.0
            return ((a - mu) / sd).tolist()

        z_speeds = znormalize(speeds)
        z_freqs = znormalize(freqs)
        z_sents = znormalize(sents)

        node_features = []
        for i, u in enumerate(real_users):
            n_id = node_map.get(u, None)
            if n_id is None:
                continue
            a = 0.6 * (z_freqs[i] if i < len(z_freqs) else 0.0)
            b = 0.8 * (z_speeds[i] if i < len(z_speeds) else 0.0)
            c = 0.4 * (z_sents[i] if i < len(z_sents) else 0.0)
            sus = sigmoid(a + b + c)
            node_features.append({'user_id': u, 'node': int(n_id), 'susceptibility': float(sus)})

        edges = [(int(u), int(v)) for u, v in G.edges()]

        out = {
            'event': t.get('event'),
            'label': t.get('label'),
            'thread_id': t.get('thread_id'),
            'context': {
                'sentiment': float(sent),
                'embedding': emb,
                'length': int(length),
                'keywords': keywords
            },
            'I_time': counts.tolist(),
            'time_grid': time_bins.tolist(),
            'total_replies': int(total_replies),
            'final_spread_ratio': float(final_spread_ratio),
            'network': {
                'N': N_nodes,
                'm': m,
                'edges': edges
            },
            'node_features': node_features
        }

        out_path = os.path.join(OUTDIR, f"thread_{idx:04d}_{t.get('thread_id')}.json")
        with open(out_path, 'w', encoding='utf8') as fo:
            json.dump(out, fo, ensure_ascii=False)
        outputs.append(out)

        if (idx + 1) % 50 == 0:
            print(f'Processed {idx+1} threads...')

    # save summary
    with open(os.path.join(OUTDIR, 'summary.json'), 'w', encoding='utf8') as f:
        json.dump({'n_threads': len(outputs)}, f)

    print('Done. Synthetic dataset pieces saved to', OUTDIR)

if __name__ == '__main__':
    main()
