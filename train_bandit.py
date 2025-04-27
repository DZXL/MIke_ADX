"""

python train_bandit.py --proc_dir processed --out_file trained_bandit.pkl

"""

import os, argparse, pickle, csv
import numpy as np
from collections import defaultdict
from thompson_util import ThompsonSampling
from path_utils import path_from_local_root

# how frequently to sample: 1 means use all, N means use 1 in N rows
SAMPLE_RATE = 1

# window size for average price
K_DAYS = 3  # window size for average price


def feat_vec_camp(r):
    return np.array([
        float(r['reach']),
        float(r['duration']),
        float(r['segment_pop_frac']),
        float(r['reach_per_day']),
        float(r['avg_price_last_3_days']),
        float(r['current_Q']),
        float(r['active_campaigns_load']),
        float(r['cash_buffer_ratio']),
        float(r['bidding_day']),
    ], dtype=float)

def feat_vec_impr(r):
    return np.array([
        float(r['num_bidders']),
        float(r['price']),
        float(r['reach']),
        float(r['duration']),
        float(r['segment_pop_frac']),
        float(r['reach_per_day']),
        float(r[f'avg_price_last_{K_DAYS}_days']),
        float(r['seg_mean']),
        float(r['seg_std']),
        float(r['seg_max']),
        float(r['current_Q']),
        float(r['active_campaigns_load']),
        float(r['cash_buffer_ratio']),
        float(r['progress_rate']),
        float(r['bidding_day']),
    ], dtype=float)


def train_bandits(proc_dir, lam, v):
    camp_bandit = ThompsonSampling(dim=9,  lam=lam, v=v)
    impr_bandit = ThompsonSampling(dim=15, lam=lam, v=v)

    with open(os.path.join(proc_dir, "campaign_features.csv"), newline='') as f:
        reader = csv.DictReader(f)
        for i, r in enumerate(reader):
            if i % SAMPLE_RATE != 0:
                continue
            x = feat_vec_camp(r)
            camp_bandit.update(x, float(r['won']))

    pos_cnt = 0
    neg_cnt = 0
    TARGET_RATIO = 5  # adjust as needed: ratio of negative to positive samples

    with open(os.path.join(proc_dir, "impression_features.csv"), newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            won = int(r['won'])
            if won == 1:
                pos_cnt += 1
            else:
                # allow up to TARGET_RATIO negatives per positive
                if pos_cnt > 0:
                    if neg_cnt >= TARGET_RATIO * pos_cnt:
                        continue
                else:
                    # allow initial negatives until first positive
                    if neg_cnt >= 100:
                        continue
                neg_cnt += 1

            x = feat_vec_impr(r)
            reward = float(r['price']) if won == 1 else 0.0
            impr_bandit.update(x, reward)

    return camp_bandit, impr_bandit

def serialize_thompson(bandit):
    return {
        "a": bandit.A,
        "b": bandit.b,
        "dim": bandit.d,
        "v": bandit.v
    }

# ─────────────────────────────────── main ─────────────────────────────────
if __name__ == "__main__":
    pa = argparse.ArgumentParser()
    pa.add_argument("--proc_dir", type=str, default="processed",
                    help="build_dataset.py The generated directory")
    pa.add_argument("--out_file", type=str, default="trained_bandit_ts.pkl",
                    help="outpit pickle")
    pa.add_argument("--lam", type=float, default=1.0,
                    help="regularization parameter lambda for ThompsonSampling")
    pa.add_argument("--v", type=float, default=2.0,
                    help="noise parameter v for ThompsonSampling")
    args = pa.parse_args()

    camp_bandit, impr_bandit = train_bandits(args.proc_dir, args.lam, args.v)

    out_path = path_from_local_root(args.out_file)
    with open(out_path, "wb") as f:
        pickle.dump({
            "campaign": serialize_thompson(camp_bandit),
            "impression": serialize_thompson(impr_bandit)
        }, f)

    print(f"✅ The training is completed!")