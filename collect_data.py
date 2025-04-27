#!/usr/bin/env python
"""
    python collect_data.py --episodes 300 --out_dir data
"""
import os
import csv
import argparse
import random
import pathlib
from typing import List, Dict, Set, Tuple
from collections import defaultdict
import numpy as np

# Import the Thompson-Sampling agent
from my_ndays_ncampaign_agent import MyNDaysNCampaignsAgent


# Open-source opponent

from adx.adx_game_simulator import CONFIG as SIM_CONF
SEG_POP   = SIM_CONF['market_segment_pop']
TOTAL_POP = sum(SEG_POP.values())


from adx.adx_game_simulator import AdXGameSimulator
from adx.tier1_ndays_ncampaign_agent import Tier1NDaysNCampaignsAgent
from adx.structures import Campaign, MarketSegment, Bid, BidBundle
from adx.agents import NDaysNCampaignsAgent

# ───────────────────────────── ExploreAgent ────────────────────────────────────
# ───────────────────────────── ExploreAgent ────────────────────────────────────
# class ExploreAgent(NDaysNCampaignsAgent):
#
#     def __init__(self, name="ProfitAgent", eps: float = 0.1):
#         super().__init__()
#         self.name = name
#         self.eps = eps  
#         self.total_days = 10
#
#     def on_new_game(self) -> None:
#         
#         pass
#
#     def get_campaign_bids(self, campaigns_for_auction: Set[Campaign]) -> Dict[Campaign, float]:
#         bids: Dict[Campaign, float] = {}
#         for c in campaigns_for_auction:
#             R = c.reach
#             if random.random() < self.eps:
#                 raw = random.uniform(0.1 * R, R)
#             else:
#                 raw = 0.5 * R
#             bids[c] = self.clip_campaign_bid(c, raw)
#         return bids
#
#     def get_ad_bids(self) -> Set[BidBundle]:
#         bundles: Set[BidBundle] = set()
#         for c in self.get_active_campaigns():
#             remain_reach  = max(1, c.reach - self.get_cumulative_reach(c))
#             remain_budget = max(0.1, c.budget - self.get_cumulative_cost(c))
#             # ε-greedy impression 
#             bid_price   = 0.15 * (remain_budget / remain_reach)
#             
#             daily_limit = remain_budget / max(1, (c.end_day - self.current_day + 1))
#             bid = Bid(self, c.target_segment, bid_price, daily_limit)
#             bundles.add(BidBundle(campaign_id=c.uid, limit=daily_limit, bid_entries={bid}))
#         return bundles

# ──────────────────────────── UCBBidder Opponent ────────────────────────────
import math
class UCBBidder(NDaysNCampaignsAgent):
    def __init__(self, name="UCBBidder", alpha=1.0):
        super().__init__()
        self.name = name
        self.alpha = alpha
        self.counts = {}
        self.values = {}

    def on_new_game(self):
        self.counts.clear()
        self.values.clear()

    def get_campaign_bids(self, campaigns_for_auction):
        return {c: 0.5 * c.reach for c in campaigns_for_auction}

    def get_ad_bids(self):
        bundles = set()
        day = self.current_day
        total_counts = sum(self.counts.values()) or 1
        for c in self.get_active_campaigns():
            seg = c.target_segment
            n = self.counts.get(seg, 0)
            v = self.values.get(seg, 0.0)
            bonus = self.alpha * math.sqrt(math.log(total_counts) / (n + 1))
            price = min(c.budget / c.reach, v + bonus)
            limit = (c.budget - self.get_cumulative_cost(c)) / max(1, (c.end_day - day + 1))
            b = Bid(self, seg, price, limit)
            bundles.add(BidBundle(campaign_id=c.uid, limit=limit, bid_entries={b}))
        return bundles

# ──────────────────────────── Epsilon-Greedy Opponent ────────────────────────────
class EpsGreedyBidder(NDaysNCampaignsAgent):
    def __init__(self, name="EpsBidder", eps=0.1):
        super().__init__()
        self.name = name
        self.eps = eps
        self.values = {}

    def on_new_game(self):
        self.values.clear()

    def get_campaign_bids(self, campaigns_for_auction):
        return {c: 0.5 * c.reach for c in campaigns_for_auction}

    def get_ad_bids(self):
        bundles = set()
        day = self.current_day
        for c in self.get_active_campaigns():
            seg = c.target_segment
            base = c.budget / c.reach
            if random.random() < self.eps:
                price = random.uniform(0.1 * c.reach, base)
            else:
                price = self.values.get(seg, base)
            limit = (c.budget - self.get_cumulative_cost(c)) / max(1, (c.end_day - day + 1))
            b = Bid(self, seg, price, limit)
            bundles.add(BidBundle(campaign_id=c.uid, limit=limit, bid_entries={b}))
        return bundles

# ──────────────────────────── Static-Fraction Opponent ────────────────────────────

class StaticBidder(NDaysNCampaignsAgent):
    def __init__(self, frac=0.7, name=None):
        name = name or f"Static_{int(frac*100)}"
        super().__init__()
        self.name = name
        self.frac = frac

    def on_new_game(self):
        pass

    def get_campaign_bids(self, campaigns_for_auction):
        return {c: self.frac * c.reach for c in campaigns_for_auction}

    def get_ad_bids(self):
        bundles = set()
        day = self.current_day
        for c in self.get_active_campaigns():
            price = self.frac * (c.budget / c.reach)
            limit = (c.budget - self.get_cumulative_cost(c)) / max(1, (c.end_day - day + 1))
            b = Bid(self, c.target_segment, price, limit)
            bundles.add(BidBundle(campaign_id=c.uid, limit=limit, bid_entries={b}))
        return bundles

# ──────────────────────────── Aggressive Bidder Opponent ────────────────────────────
class AggressiveBidder(NDaysNCampaignsAgent):
    def __init__(self, name="AggressiveBidder"):
        super().__init__()
        self.name = name

    def on_new_game(self):
        pass

    def get_campaign_bids(self, campaigns_for_auction):
        return {c: c.reach for c in campaigns_for_auction}

    def get_ad_bids(self):
        bundles = set()
        day = self.current_day
        for c in self.get_active_campaigns():
            price = c.budget / c.reach if c.reach > 0 else 0.0
            limit = (c.budget - self.get_cumulative_cost(c)) / max(1, (c.end_day - day + 1))
            b = Bid(self, c.target_segment, price, limit)
            bundles.add(BidBundle(campaign_id=c.uid, limit=limit, bid_entries={b}))
        return bundles

# ──────────────────────────── Pacing Bidder Opponent ────────────────────────────
class PacingBidder(NDaysNCampaignsAgent):
    def __init__(self, name="PacingBidder"):
        super().__init__()
        self.name = name

    def on_new_game(self):
        pass

    def get_campaign_bids(self, campaigns_for_auction):
        # bid half of reach for new campaign auctions
        bids = {}
        for c in campaigns_for_auction:
            raw = 0.5 * c.reach
            bids[c] = self.clip_campaign_bid(c, raw)
        return bids

    def get_ad_bids(self):
        bundles = set()
        day = self.current_day
        for c in self.get_active_campaigns():
            rem_budget = max(0.0, c.budget - self.get_cumulative_cost(c))
            rem_days = max(1, c.end_day - day + 1)
            daily_budget = rem_budget / rem_days
            rem_reach = max(1, c.reach - self.get_cumulative_reach(c))
            price = daily_budget / rem_reach
            limit = daily_budget
            b = Bid(self, c.target_segment, price, limit)
            bundles.add(BidBundle(campaign_id=c.uid, limit=limit, bid_entries={b}))
        return bundles

# ──────────────────────────── Logging Simulator ────────────────────────────────
class LoggingSimulator(AdXGameSimulator):

    def __init__(self, camp_writer: csv.writer, impr_writer: csv.writer, q_writer: csv.writer, *a, **kw):
        super().__init__(*a, **kw)
        self._camp_w = camp_writer
        self._impr_w = impr_writer
        self._q_w = q_writer
        self._round  = 0

    def run_simulation(self, *args, num_simulations=1, **kwargs):
        for r in range(num_simulations):
            self._round = r + 1
            super().run_simulation(*args, num_simulations=1, **kwargs)

    def run_ad_auctions(
        self,
        bid_bundles: List[BidBundle],
        users: List[MarketSegment],
        day: int,
        *args, **kwargs
    ) -> None:
        self.current_day = day
        bidder_states = self.states
        seg2bids = defaultdict(set)
        bid2bundle = {}
        bid2spend  = {}
        limits     = {}

        for bdl in bid_bundles:
            limits[bdl.campaign_id] = bdl.limit
            if bdl.campaign_id not in self.campaigns: continue
            camp = self.campaigns[bdl.campaign_id]
            if not (camp.start_day <= day <= camp.end_day): continue
            for bid in bdl.bid_entries:
                if self.is_valid_bid(bid):
                    seg2bids[bid.item].add(bid)
                    bid2bundle[bid] = bdl
                    bid2spend[bid]  = 0.0

        for user_seg in users:
            prices = [b.bid_per_item for seg in self.sub_segments[user_seg] for b in seg2bids[seg]]
            seg_mean = float(np.mean(prices)) if prices else 0.0
            seg_std  = float(np.std(prices))  if prices else 0.0
            seg_max  = float(max(prices))     if prices else 0.0

            bids = []
            for seg in self.sub_segments[user_seg]:
                bids.extend(seg2bids[seg])
            bids.sort(key=lambda b: b.bid_per_item, reverse=True)
            num = len(bids)

            winning_bid = None
            for idx, bid in enumerate(bids):
                price = bids[idx+1].bid_per_item if idx+1 < num else 0.0
                bundle = bid2bundle[bid]
                agent  = bid.bidder
                cid    = bundle.campaign_id

                if bid2spend[bid] + price > bid.bid_limit or \
                   bidder_states[agent].spend[cid] + price > limits[cid]:
                    seg2bids[bid.item].remove(bid)
                    continue

                bid2spend[bid] += price
                st = bidder_states[agent]
                st.spend[cid] += price
                camp = st.campaigns[cid]
                camp.cumulative_cost += price
                won = False
                if camp.target_segment.issubset(user_seg):
                    st.impressions[cid] += 1
                    camp.cumulative_reach += 1
                    won = True
                # Only allow a single winner per impression
                winning_bid = bid
                break

            # ------------- WRITE LOG -------------
            for bid in bids:
                bundle = bid2bundle[bid]
                agent  = bid.bidder
                cid    = bundle.campaign_id
                st = bidder_states[agent]
                camp = st.campaigns[cid]
                reach = camp.reach
                duration = camp.end_day - camp.start_day + 1
                seg_frac = SEG_POP.get(camp.target_segment, 0) / TOTAL_POP if camp.target_segment in SEG_POP else 0
                reach_per_day = reach / duration if duration > 0 else 0
                # average price last 3 days (=avg_last3)
                last3_prices = []
                for past_day in range(max(camp.start_day, day - 2), day + 1):
                    impressions = st.impressions.get(cid, 0)
                    spend = st.spend.get(cid, 0)
                    if impressions > 0:
                        avg_price = spend / impressions
                    else:
                        avg_price = 0.0
                    last3_prices.append(avg_price)
                avg_last3 = sum(last3_prices) / len(last3_prices) if last3_prices else 0.0
                won_flag = int(bid is winning_bid)
                cur_q = st.quality_score
                progress_rate = st.impressions[cid] / reach if reach > 0 else 0
                active_campaigns_load = 0.0
                for c2 in st.campaigns.values():
                    remain2 = max(0, c2.reach - st.impressions[c2.uid])
                    days2 = max(1, c2.end_day - self.current_day + 1)
                    active_campaigns_load += remain2 / days2
                total_bud2 = sum(st.budgets[c2.uid] for c2 in st.campaigns.values())
                rem_cash2 = sum(st.budgets[c2.uid] - st.spend[c2.uid] for c2 in st.campaigns.values())
                cash_buffer_ratio = max(0.0001, rem_cash2 / max(1e-6, total_bud2))
                self._impr_w.writerow([
                    self._round,
                    day,
                    cid,
                    str(user_seg),
                    num,
                    bid.bid_per_item,
                    won_flag,
                    reach,
                    duration,
                    seg_frac,
                    reach_per_day,
                    seg_mean,
                    seg_std,
                    seg_max,
                    avg_last3,
                    cur_q,
                    progress_rate,
                    active_campaigns_load,
                    cash_buffer_ratio,
                    day
                ])

    def run_campaign_auctions(
        self,
        agent_bids: Dict[NDaysNCampaignsAgent, Dict[Campaign, float]],
        new_campaigns: List[Campaign],
        campaigns_won_counter=None,
        *args, **kwargs
    ) -> None:
        for camp in set(new_campaigns):
            entries = []
            for ag in self.agents:
                if camp in agent_bids[ag]:
                    raw = agent_bids[ag][camp]
                    if self.states[ag].quality_score > 0 and self.is_valid_campaign_bid(raw, camp.reach):
                        eff = raw / self.states[ag].quality_score
                        entries.append((ag, eff, raw))

            winner_name, sec_price = "None", 0.0
            if entries:
                entries.sort(key=lambda x: x[1])
                winner, _, raw_min = entries[0]
                winner_name = winner.name
                if len(entries) == 1:
                    q_low = sum(sorted([self.states[a].quality_score for a in self.agents])[:3]) / 3
                    budget = camp.reach / q_low * self.states[winner].quality_score
                    sec_price = budget / self.states[winner].quality_score
                else:
                    eff2 = entries[1][1]
                    sec_price = eff2 * self.states[winner].quality_score
                    budget   = sec_price

                camp.budget = budget
                winner.my_campaigns.add(camp)
                self.states[winner].add_campaign(camp)
                self.campaigns[camp.uid] = camp
                if campaigns_won_counter is not None:
                    campaigns_won_counter[winner.name] += 1

            self._camp_w.writerow([
                self._round,
                self.current_day,
                camp.uid,
                camp.reach,
                camp.start_day,
                camp.end_day,
                str(camp.target_segment),
                winner_name,
                sec_price,
                len(entries)
            ])

        for ag in self.agents:
            st = self.states[ag]
            load = 0.0
            for camp in st.campaigns.values():
                cid = camp.uid
                remain_impressions = max(0, camp.reach - st.impressions[cid])
                remain_days = max(1, camp.end_day - self.current_day + 1)
                load += remain_impressions / remain_days
            total_budget = sum(st.budgets[cid] for cid in st.campaigns)
            remain_cash = sum(st.budgets[cid] - st.spend[cid] for cid in st.campaigns)
            cash_buffer_ratio = max(0.0001, remain_cash / max(1e-6, total_budget))
            self._q_w.writerow([self._round, self.current_day, ag.name, st.quality_score, load, cash_buffer_ratio])


def main(out_dir: str, episodes: int):
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)

    camp_csv = os.path.join(out_dir, "campaign_log.csv")
    impr_csv = os.path.join(out_dir, "impression_log.csv")
    q_csv = os.path.join(out_dir, "q_log.csv")
    write_c = not os.path.exists(camp_csv)
    write_i = not os.path.exists(impr_csv)
    write_q = not os.path.exists(q_csv)

    with open(camp_csv,   "a", newline="") as fc, \
         open(impr_csv,   "a", newline="") as fi, \
         open(q_csv,      "a", newline="") as fq:
        wc = csv.writer(fc)
        wi = csv.writer(fi)
        wq = csv.writer(fq)
        if write_c:
            wc.writerow([
                "round","day","campaign_uid","reach",
                "start_day","end_day","segment",
                "winner","second_price","num_bidders"
            ])
        if write_i:
            wi.writerow([
                "round","day","campaign_uid","user_segment","num_bidders","price","won",
                "reach","duration","segment_pop_frac","reach_per_day",
                "seg_mean","seg_std","seg_max",
                "avg_price_last_3_days","current_Q","progress_rate","active_campaigns_load","cash_buffer_ratio","bidding_day"
            ])
        if write_q:
            wq.writerow([
                "round","day","agent","current_Q","active_campaigns_load","cash_buffer_ratio"
            ])

        agents = [
            MyNDaysNCampaignsAgent(),
            StaticBidder(frac=0.2, name="Static_20"),
            StaticBidder(frac=0.5, name="Static_50"),
            StaticBidder(frac=0.8, name="Static_80"),
            EpsGreedyBidder(eps=0.05, name="Eps_05"),
            EpsGreedyBidder(eps=0.2,  name="Eps_20"),
            UCBBidder(alpha=0.5,   name="UCB_05"),
            UCBBidder(alpha=2.0,   name="UCB_2"),
            PacingBidder(name="Pacing"),
            AggressiveBidder(name="Aggressive"),
        ]

        sim = LoggingSimulator(wc, wi, wq)
        sim.run_simulation(agents=agents, num_simulations=episodes)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=10, help="Simulation Rounds")
    p.add_argument("--out_dir",  type=str, default="data", help="Output directory")
    args = p.parse_args()
    main(args.out_dir, args.episodes)