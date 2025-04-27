import os
import pickle
import pathlib
import numpy as np
import sys
import io
from collections import defaultdict

from thompson_util import ThompsonSampling
from path_utils import path_from_local_root

from adx.agents import NDaysNCampaignsAgent
from adx.structures import Bid, BidBundle, MarketSegment, Campaign
from adx.adx_game_simulator import CONFIG
from adx.tier1_ndays_ncampaign_agent import Tier1NDaysNCampaignsAgent


def deserialize_thompson(data):
    bandit = ThompsonSampling(dim=data["dim"], v=data["v"])
    bandit.A = data["a"]
    bandit.b = data["b"]
    return bandit


K_DAYS     = 3  
SEG_POP    = CONFIG["market_segment_pop"]
TOTAL_POP  = sum(SEG_POP.values())

class MyNDaysNCampaignsAgent(NDaysNCampaignsAgent):
    def __init__(self, model_file="ts5.pkl"):
        super().__init__()
        self.name = "ProfitAgent"
        
        model_path = pathlib.Path(path_from_local_root(model_file))
        with model_path.open("rb") as f:
            mdl = pickle.load(f)
        self.campaign_bandit = deserialize_thompson(mdl["campaign"])
        self.impression_bandit = deserialize_thompson(mdl["impression"])
        self.price_history = defaultdict(list)

    def on_new_game(self) -> None:
        
        self.price_history.clear()

    def get_campaign_bids(self, campaigns_for_auction):
        """
        Calculate a 9-dimensional feature vector for each campaign to be auctioned, use the UCB of the campaign-Bandit to output λ_c, and then use λ_c * reach as the raw bid.
        """
        bids = {}
        day = self.get_current_day()
        # First calculate the agent's overall load and cash_buffer
        load = 0.0
        total_bud, remain_cash = 0.0, 0.0
        for c in self.get_active_campaigns():
            rem_reach = max(0, c.reach - self.get_cumulative_reach(c))
            rem_days  = max(1, c.end_day - day + 1)
            load += rem_reach / rem_days
            total_bud  += c.budget
            remain_cash += c.budget - self.get_cumulative_cost(c)
        cash_buf = remain_cash / (total_bud + 1e-6)
        # Current quality score
        cur_Q = self.get_quality_score()

        for c in campaigns_for_auction:
            # Basic campaign features
            reach    = c.reach
            duration = c.end_day - c.start_day + 1
            seg_frac = SEG_POP.get(c.target_segment, 0) / TOTAL_POP
            rpd      = reach / duration
            # avg price of this segment over the past K_DAYS
            past = self.price_history[c.target_segment]
            avg_p = float(np.mean(past[-K_DAYS:])) if past else 0.0

            # 9-dimensional context vector
            x = np.array([
                reach,
                duration,
                seg_frac,
                rpd,
                avg_p,
                cur_Q,
                load,
                cash_buf,
                day
            ], dtype=float)

            # UCB as λ_c
            lam_c = self.campaign_bandit.ucb(x)
            raw   = lam_c * reach
            bid   = self.clip_campaign_bid(c, raw)
            bids[c] = bid

        return bids

    def get_ad_bids(self):
        """
        Bid for each active campaign impression:
        We use campaign-related features + timestamp to form an 11-dimensional context, use the UCB of the impression-Bandit to output λ_i, and then combine the remaining budget/reach to get the impression bid.
        """
        bundles = set()
        day = self.get_current_day()

        # Recompute global load and cash_buf 
        total_bud, remain_cash, load = 0.0, 0.0, 0.0
        for c in self.get_active_campaigns():
            total_bud  += c.budget
            remain_cash += c.budget - self.get_cumulative_cost(c)
            rem_r = max(0, c.reach - self.get_cumulative_reach(c))
            rem_d = max(1, c.end_day - day + 1)
            load += rem_r / rem_d
        cash_buf = remain_cash / (total_bud + 1e-6)
        cur_Q    = self.get_quality_score()

        for c in self.get_active_campaigns():
            # Remaining reach / budget
            done      = self.get_cumulative_reach(c)
            cost      = self.get_cumulative_cost(c)
            rem_r     = max(1, c.reach - done)
            rem_b     = max(0.0, c.budget - cost)
            duration  = c.end_day - c.start_day + 1
            seg_frac  = SEG_POP.get(c.target_segment, 0) / TOTAL_POP
            rpd       = c.reach / duration
            avg_p     = float(np.mean(self.price_history[c.target_segment][-K_DAYS:])) if self.price_history[c.target_segment] else 0.0

            # 15-dimensional context vector
            past = self.price_history[c.target_segment]
            seg_mean = float(np.mean(past)) if past else 0.0
            seg_std  = float(np.std(past))  if past else 0.0
            seg_max  = float(max(past))     if past else 0.0
            progress_rate = done / c.reach if c.reach > 0 else 0.0

            x_imp = np.array([
                0.0,            # num_bidders placeholder
                0.0,            # price placeholder
                c.reach,
                duration,
                seg_frac,
                rpd,
                avg_p,          # avg_price_last_K days
                seg_mean,
                seg_std,
                seg_max,
                cur_Q,
                load,
                cash_buf,
                progress_rate,
                day
            ], dtype=float)

            lam_i = self.impression_bandit.ucb(x_imp)
            # pacing base price
            base = rem_b / rem_r
            bid_price = max(0.1, min(base, lam_i))
            daily_lim = rem_b / max(1, (c.end_day - day + 1))
            # clamp bid_price to not exceed spending limit
            bid_price = min(bid_price, daily_lim)

            b = Bid(self, c.target_segment, bid_price, daily_lim)
            bundles.add(BidBundle(campaign_id=c.uid,
                                  limit=daily_lim,
                                  bid_entries={b}))
        return bundles


if __name__ == "__main__":
    from collect_data import StaticBidder, EpsGreedyBidder, UCBBidder, PacingBidder, AggressiveBidder
    from adx.adx_game_simulator import AdXGameSimulator

    # ensure results directory exists
    results_dir = pathlib.Path("results")
    results_dir.mkdir(exist_ok=True)

    
    models = [
        
        "ts_l2_v1.pkl",   "ts_l2_v2.pkl",   "ts_l2_v5.pkl",
    ]
    sims = 100
    for mf in models:
        print("\n" + "="*60)
        print(f" MODEL: {mf} ".center(60, "="))
        print("="*60)

        # Prepare statistics
        total_profits = defaultdict(float)
        win_counts = defaultdict(int)

        for i in range(sims):
            # Initialize a fresh simulator each run
            sim = AdXGameSimulator()
            # Instantiate bots for this run
            bots = [
                MyNDaysNCampaignsAgent(model_file=mf),
                Tier1NDaysNCampaignsAgent(name="Tier1"),
                StaticBidder(frac=0.1, name="Static_10"),
                StaticBidder(frac=0.3, name="Static_30"),
                EpsGreedyBidder(eps=0.01, name="Eps_01"),
                EpsGreedyBidder(eps=0.10, name="Eps_10"),
                UCBBidder(alpha=0.2, name="UCB_02"),
                UCBBidder(alpha=1.0, name="UCB_10"),
                PacingBidder(name="Pacing"),
                AggressiveBidder(name="Aggressive"),
            ]
            # Suppress simulator logs, run one simulation, then restore stdout
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            sim.run_simulation(agents=bots, num_simulations=1)
            sys.stdout = old_stdout

            # Capture results from each agent
            res = {agent.name: agent.get_cumulative_profit() for agent in bots}

            # Accumulate
            max_profit = max(res.values())
            for agent_name, profit in res.items():
                total_profits[agent_name] += profit
                if profit == max_profit:
                    win_counts[agent_name] += 1

        # Print summary
        print("\nFinal summary over", sims, "simulations:")
        print("Agent           | Avg Profit | Wins")
        print("------------------------------------------")
        for agent_name, profit_sum in total_profits.items():
            avg = profit_sum / sims
            w = win_counts.get(agent_name, 0)
            print(f"{agent_name.ljust(15)}| {avg:10.2f} | {w}")
        print("-"*42)

        # Write summary to a file
        out_fname = results_dir / mf.replace('.pkl', '_results.txt')
        with open(out_fname, 'w') as f:
            f.write(f"MODEL: {mf}\n")
            f.write("Agent           | Avg Profit | Wins\n")
            f.write("--------------------------------\n")
            for agent_name, profit_sum in total_profits.items():
                avg = profit_sum / sims
                w = win_counts.get(agent_name, 0)
                f.write(f"{agent_name.ljust(15)}| {avg:10.2f} | {w}\n")