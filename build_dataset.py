"""
python build_dataset.py --in_dir data --out_dir processed
"""

import os
import csv
import argparse
import ast
from collections import defaultdict, deque
from statistics import mean
from adx.adx_game_simulator import CONFIG

K_DAYS = 3   #

MARKET_POP = CONFIG['market_segment_pop']
TOTAL_POP   = sum(MARKET_POP.values())

POP_BY_FROZ = {frozenset(k): v for k, v in MARKET_POP.items()}

def parse_segment(seg_str: str):

    inside = seg_str.strip()

    if inside.startswith("MarketSegment("):
        inside = inside[len("MarketSegment("):-1]
    # Parse into Python set using ast
    return frozenset(ast.literal_eval(inside))

def load_q_log(q_log_path: str):
    q_data = {}
    with open(q_log_path, newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rnd = int(r['round'])
            day = int(r['day'])
            q = float(r['current_Q'])
            load = float(r['active_campaigns_load'])
            cash_buf = float(r['cash_buffer_ratio'])
            q_data[(rnd, day)] = (q, load, cash_buf)
    return q_data

def build_campaign_features(in_dir: str, q_log_path: str, out_path: str):
    inp = os.path.join(in_dir, "campaign_log.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Read all rows
    rows = []
    with open(inp, newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            if int(r['day']) < 0:  # Skip the initial batch
                continue
            rows.append({
                'round':        int(r['round']),
                'day':          int(r['day']),
                'campaign_uid': int(r['campaign_uid']),
                'reach':        int(r['reach']),
                'start_day':    int(r['start_day']),
                'end_day':      int(r['end_day']),
                'segment':      parse_segment(r['segment']),
                'winner':       r['winner'],
                'second_price': float(r['second_price']),
                'num_bidders':  int(r['num_bidders']),
            })

    # Sort by (round, day)
    rows.sort(key=lambda x: (x['round'], x['day']))

    # Maintain sliding window for each segment
    history = defaultdict(lambda: deque(maxlen=K_DAYS))
    curr_round = None

    # Read Q_log
    q_log = load_q_log(q_log_path)

    # Write to file
    header = [
        'round','day','campaign_uid',
        # 9 features
        'reach','duration','segment_pop_frac','reach_per_day',
        f'avg_price_last_{K_DAYS}_days',
        'current_Q','active_campaigns_load','cash_buffer_ratio','bidding_day',
        # Label
        'won'
    ]
    with open(out_path, 'w', newline='') as fo:
        writer = csv.writer(fo)
        writer.writerow(header)

        for rec in rows:
            rnd = rec['round']
            day = rec['day']
            seg = rec['segment']

            # Reset history for new round
            if rnd != curr_round:
                curr_round = rnd
                history.clear()

            # Calculate features
            reach   = rec['reach']
            dur     = rec['end_day'] - rec['start_day'] + 1
            pop_frac= POP_BY_FROZ.get(seg, 0) / TOTAL_POP
            rpd     = reach / dur if dur>0 else 0.0

            past = list(history[seg])
            avg_p = mean(past) if past else 0.0
            history[seg].append(rec['second_price'])

            current_Q, load_active, cash_buf = q_log.get((rnd, day), (-1.0, -1.0, -1.0))
            won = 1 if rec['winner']=="ProfitAgent" else 0

            writer.writerow([
                rnd, day, rec['campaign_uid'],
                reach, dur, pop_frac, rpd,
                f"{avg_p:.6f}",
                f"{current_Q:.6f}", f"{load_active:.6f}", f"{cash_buf:.6f}",
                day,
                won
            ])

def build_impression_features(in_dir: str, camp_feat_path: str, out_path: str):
    imp_in = os.path.join(in_dir, "impression_log.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # First read campaign_features.csv, use (round, campaign_uid) as key
    camp_feats = {}
    with open(camp_feat_path, newline='') as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            key = (int(r['round']), int(r['campaign_uid']))
            camp_feats[key] = [
                r['reach'], r['duration'], r['segment_pop_frac'], r['reach_per_day'],
                r[f'avg_price_last_{K_DAYS}_days'],
                r['current_Q'], r['active_campaigns_load'], r['cash_buffer_ratio']
            ]

    # Read impression_log.csv
    with open(out_path, 'w', newline='') as fo, open(imp_in, newline='') as fi:
        reader = csv.DictReader(fi)
        header = [
            'round','day','campaign_uid',
            'user_segment','num_bidders','price','won',
            'reach','duration','segment_pop_frac','reach_per_day',
            f'avg_price_last_{K_DAYS}_days',
            'seg_mean','seg_std','seg_max',
            'current_Q','active_campaigns_load','cash_buffer_ratio',
            'progress_rate','bidding_day'
        ]
        writer = csv.writer(fo)
        writer.writerow(header)

        for r in reader:
            if r['campaign_uid'].lower()=='n/a':
                continue
            rnd = int(r['round'])
            day = int(r['day'])
            try:
                cid = int(r['campaign_uid'])
            except ValueError:
                continue
            key = (rnd, cid)
            cf = camp_feats.get(key)
            if cf is None:
                continue

            writer.writerow([
                rnd, day, cid,
                r['user_segment'], r['num_bidders'], r['price'], r['won'],
                *cf,
                r['seg_mean'], r['seg_std'], r['seg_max'],
                r['current_Q'], r['active_campaigns_load'], r['cash_buffer_ratio'],
                r['progress_rate'], day
            ])

if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--in_dir",  type=str, default="data",      help="Output directory of collect_data")
    p.add_argument("--out_dir", type=str, default="processed", help="Output feature directory")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    build_campaign_features(
        in_dir=args.in_dir,
        q_log_path=os.path.join(args.in_dir, "q_log.csv"),
        out_path=os.path.join(args.out_dir, "campaign_features.csv")
    )
    build_impression_features(
        in_dir=args.in_dir,
        camp_feat_path=os.path.join(args.out_dir, "campaign_features.csv"),
        out_path=os.path.join(args.out_dir, "impression_features.csv")
    )

    print("✅ Build complete:", args.out_dir)