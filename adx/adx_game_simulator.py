import random
from typing import List, Dict, Optional
from collections import defaultdict
from math import isfinite, atan
import pathlib

from adx.structures import Campaign, Bid, BidBundle, MarketSegment
from adx.pmfs import PMF
from adx.tier1_ndays_ncampaign_agent import Tier1NDaysNCampaignsAgent
from adx.states import CampaignBidderState
from adx.agents import NDaysNCampaignsAgent

CONFIG = {
    'num_agents': 10,
    'num_days': 10,
    'quality_score_alpha': 0.5,
    'campaigns_per_day': 5,
    'campaign_reach_dist': [0.3, 0.5, 0.7],
    'campaign_length_dist': [1, 2, 3],
    'market_segment_dist': [
        MarketSegment(("Male", "Young")),
        MarketSegment(("Male", "Old")),
        MarketSegment(("Male", "LowIncome")),
        MarketSegment(("Male", "HighIncome")),
        MarketSegment(("Female", "Young")),
        MarketSegment(("Female", "Old")),
        MarketSegment(("Female", "LowIncome")),
        MarketSegment(("Female", "HighIncome")),
        MarketSegment(("Young", "LowIncome")),
        MarketSegment(("Young", "HighIncome")),
        MarketSegment(("Old", "LowIncome")),
        MarketSegment(("Old", "HighIncome")),
        MarketSegment(("Male", "Young", "LowIncome")),
        MarketSegment(("Male", "Young", "HighIncome")),
        MarketSegment(("Male", "Old", "LowIncome")),
        MarketSegment(("Male", "Old", "HighIncome")),
        MarketSegment(("Female", "Young", "LowIncome")),
        MarketSegment(("Female", "Young", "HighIncome")),
        MarketSegment(("Female", "Old", "LowIncome")),
        MarketSegment(("Female", "Old", "HighIncome"))
    ],
    'market_segment_pop': {
        MarketSegment(("Male", "Young")): 2353,
        MarketSegment(("Male", "Old")): 2603,
        MarketSegment(("Male", "LowIncome")): 3631,
        MarketSegment(("Male", "HighIncome")): 1325,
        MarketSegment(("Female", "Young")): 2236,
        MarketSegment(("Female", "Old")): 2808,
        MarketSegment(("Female", "LowIncome")): 4381,
        MarketSegment(("Female", "HighIncome")): 663,
        MarketSegment(("Young", "LowIncome")): 3816,
        MarketSegment(("Young", "HighIncome")): 773,
        MarketSegment(("Old", "LowIncome")): 4196,
        MarketSegment(("Old", "HighIncome")): 1215,
        MarketSegment(("Male", "Young", "LowIncome")): 1836,
        MarketSegment(("Male", "Young", "HighIncome")): 517,
        MarketSegment(("Male", "Old", "LowIncome")): 1795,
        MarketSegment(("Male", "Old", "HighIncome")): 808,
        MarketSegment(("Female", "Young", "LowIncome")): 1980,
        MarketSegment(("Female", "Young", "HighIncome")): 256,
        MarketSegment(("Female", "Old", "LowIncome")): 2401,
        MarketSegment(("Female", "Old", "HighIncome")): 407
    },
    'user_segment_pmf': {
        MarketSegment(("Male", "Young", "LowIncome")): 0.1836,
        MarketSegment(("Male", "Young", "HighIncome")): 0.0517,
        MarketSegment(("Male", "Old", "LowIncome")): 0.1795,
        MarketSegment(("Male", "Old", "HighIncome")): 0.0808,
        MarketSegment(("Female", "Young", "LowIncome")): 0.1980,
        MarketSegment(("Female", "Young", "HighIncome")): 0.0256,
        MarketSegment(("Female", "Old", "LowIncome")): 0.2401,
        MarketSegment(("Female", "Old", "HighIncome")): 0.0407
    }
}


def calculate_effective_reach(x: int, R: int) -> float:
    return (2.0 / 4.08577) * (atan(4.08577 * (x / R) - 3.08577) - atan(-3.08577))


class AdXGameSimulator:
    def __init__(self, config: Optional[Dict] = None):
        if config is None:
            config = CONFIG
        self.num_agents = config['num_agents']
        self.num_days = config['num_days']
        self.α = config['quality_score_alpha']
        self.campaigns_per_day = config['campaigns_per_day']
        self.agents: List[NDaysNCampaignsAgent] = []
        self.campaign_reach_dist = config['campaign_reach_dist']
        self.campaign_length_dist = config['campaign_length_dist']
        self.market_segment_dist = config['market_segment_dist']
        self.market_segment_pop = config['market_segment_pop']
        self.user_segment_dist = PMF(config['user_segment_pmf'])
        self.sub_segments = defaultdict(list)
        for user_seg in config['user_segment_pmf']:
            for market_seg in config['market_segment_dist']:
                if market_seg.issubset(user_seg):
                    self.sub_segments[user_seg].append(market_seg)

    def init_agents(self, agent_types: List[NDaysNCampaignsAgent]) -> Dict[NDaysNCampaignsAgent, CampaignBidderState]:
        states = {}
        self.agents = []
        for i, agent in enumerate(agent_types):
            agent.init()
            self.agents.append(agent)
            agent.agent_num = i
            states[agent] = CampaignBidderState(i)
        return states

    def generate_campaign(self, start_day: int, end_day: Optional[int] = None) -> Campaign:
        delta = random.choice(self.campaign_reach_dist)
        length = random.choice(self.campaign_length_dist)
        mkt_segment = random.choice(self.market_segment_dist)
        reach = int(self.market_segment_pop[mkt_segment] * delta)
        if end_day is None:
            end_day = start_day + length - 1
        return Campaign(reach=reach,
                        target=mkt_segment,
                        start_day=start_day,
                        end_day=end_day)

    def is_valid_campaign_bid(self, bid: float, reach: int) -> bool:
        return isfinite(bid) and 0.1 * reach <= bid <= reach

    def is_valid_bid(self, bid: Bid) -> bool:
        return isfinite(bid.bid_per_item) and bid.bid_per_item > 0

    def run_ad_auctions(self, bid_bundles: List[BidBundle], users: List[MarketSegment], day: int,
                        profitagent_bid_segments: set, impressions_won_counter: Dict[str, int]) -> None:
        bidder_states = self.states
        seg_to_bid = defaultdict(set)
        daily_limits = {}
        bid_to_bundle = {}
        bid_to_spend = {}

        for bundle in bid_bundles:
            daily_limits[bundle.campaign_id] = bundle.limit
            if bundle.campaign_id not in self.campaigns:
                continue
            camp = self.campaigns[bundle.campaign_id]
            if not (camp.start_day <= day <= camp.end_day):
                continue
            for bid in bundle.bid_entries:
                if self.is_valid_bid(bid):
                    bid_to_bundle[bid] = bundle
                    seg_to_bid[bid.item].add(bid)
                    bid_to_spend[bid] = 0
                    if bid.bidder.name == "ProfitAgent":
                        profitagent_bid_segments.add(bid.item)

        for user_seg in users:
            bids = []
            for seg in self.sub_segments[user_seg]:
                bids.extend(seg_to_bid[seg])
            bids.sort(key=lambda b: b.bid_per_item, reverse=True)

            for i, bid in enumerate(bids):
                price = bids[i + 1].bid_per_item if i + 1 < len(bids) else 0
                bundle = bid_to_bundle[bid]
                bidder_states[bid.bidder].spend[bundle.campaign_id] += price

                over_bid = bid_to_spend[bid] + price > bid.bid_limit
                over_bundle = bidder_states[bid.bidder].spend[bundle.campaign_id] > daily_limits[bundle.campaign_id]
                if over_bid or over_bundle:
                    seg_to_bid[bid.item].remove(bid)
                    continue
                bid_to_spend[bid] += price
                st = bidder_states[bid.bidder]
                camp = st.campaigns[bundle.campaign_id]
                if camp:
                    camp.cumulative_cost += price
                    if camp.target_segment.issubset(user_seg):
                        st.impressions[bundle.campaign_id] += 1
                        camp.cumulative_reach += 1
                        if bid.bidder.name == "ProfitAgent":
                            impressions_won_counter["ProfitAgent"] += 1
                break

    def run_campaign_auctions(self, agent_bids: Dict[NDaysNCampaignsAgent, Dict[Campaign, float]],
                              new_campaigns: List[Campaign], campaigns_won_counter: Dict[str, int]) -> None:
        for camp in set(new_campaigns):
            bids = []
            for agent in self.agents:
                if camp in agent_bids[agent]:
                    val = agent_bids[agent][camp]
                    if self.states[agent].quality_score > 0 and self.is_valid_campaign_bid(val, camp.reach):
                        eff = val / self.states[agent].quality_score
                        bids.append((agent, eff))
            if not bids:
                continue
            winner, _ = min(bids, key=lambda x: x[1])
            if len(bids) == 1:
                q_low = sum(sorted([self.states[a].quality_score for a in self.agents])[:3]) / 3
                budget = camp.reach / q_low * self.states[winner].quality_score
            else:
                second = sorted(bids, key=lambda x: x[1])[1][1]
                budget = second * self.states[winner].quality_score
            camp.budget = budget
            winner.my_campaigns.add(camp)
            self.states[winner].add_campaign(camp)
            self.campaigns[camp.uid] = camp
            if winner.name == "ProfitAgent":
                campaigns_won_counter["ProfitAgent"] += 1

    def generate_auction_items(self, num_items: int) -> List[MarketSegment]:
        return self.user_segment_dist.draw_n(num_items, replace=True)

    def print_game_results(self) -> None:
        print("\n\t################# SIMULATION RESULTS ##################")
        print("\n\t#### Agent \t\t# Profit \t###")
        print("\n\t###########################################")
        for agent in sorted(self.agents, key=lambda a: self.states[a].profits, reverse=True):
            st = self.states[agent]
            print(f"\n\t### {agent.name:12} \t# {st.profits:8.2f} \t###")
        print("\n\t###########################################")
        winner = max(self.agents, key=lambda a: self.states[a].profits)
        print(f"\n\t@@@ WINNER: {winner.name} @@@")

    def run_simulation(self, agents: List[NDaysNCampaignsAgent], num_simulations: int) -> None:
        total_profits = {agent.name: 0.0 for agent in agents}
        win_counts = defaultdict(int)

        # Prepare daily report file
        report_file = pathlib.Path("daily_report.txt")
        # Clear previous contents
        report_file.write_text("")

        for i in range(num_simulations):
            self.states = self.init_agents(agents)
            self.campaigns = {}
            for agent in self.agents:
                agent.current_game = i + 1
                agent.my_campaigns = set()
                agent.on_new_game()
                rc = self.generate_campaign(start_day=1)
                rc.budget = rc.reach
                self.states[agent].add_campaign(rc)
                agent.my_campaigns.add(rc)
                self.campaigns[rc.uid] = rc

            campaigns_won_counter = defaultdict(int)
            impressions_won_counter = defaultdict(int)

            for day in range(1, self.num_days + 1):
                # from collections import defaultdict  # Removed inner import to avoid shadowing
                daily_campaigns_won = defaultdict(int)
                daily_impressions_won = defaultdict(int)
                for agent in self.agents:
                    agent.current_day = day

                if day < self.num_days:
                    new_camps = [self.generate_campaign(start_day=day+1) for _ in range(self.campaigns_per_day)]
                    new_camps = [c for c in new_camps if c.end_day <= self.num_days]
                    bids = {agent: agent.get_campaign_bids(new_camps) for agent in self.agents}
                else:
                    new_camps = []
                    bids = {agent: {} for agent in self.agents}

                ad_bids = []
                profitagent_bid_segments = set()
                for agent in self.agents:
                    bundles = agent.get_ad_bids()
                    ad_bids.extend(bundles)
                    if agent.name == "ProfitAgent":
                        for bundle in bundles:
                            for bid in bundle.bid_entries:
                                profitagent_bid_segments.add(bid.item)

                users = self.generate_auction_items(10000)
                targetable_users = sum(
                    1 for u in users
                    if any(seg.issubset(u) for seg in profitagent_bid_segments)
                )

                self.run_ad_auctions(ad_bids, users, day,
                                     profitagent_bid_segments,
                                     daily_impressions_won)

                for agent in self.agents:
                    st = self.states[agent]
                    todays_profit = 0.0
                    qs_count = 0
                    qs_sum = 0.0
                    for camp in st.campaigns.values():
                        if camp.start_day <= day <= camp.end_day and day == camp.end_day:
                            imp = st.impressions[camp.uid]
                            cost = st.spend[camp.uid]
                            er = calculate_effective_reach(imp, camp.reach)
                            todays_profit += er * st.budgets[camp.uid] - cost
                            qs_count += 1
                            qs_sum += er
                    if qs_count > 0:
                        avg_er = qs_sum / qs_count
                        new_q = (1 - self.α) * st.quality_score + self.α * avg_er
                        self.states[agent].quality_score = new_q
                        agent.quality_score = new_q
                    st.profits += todays_profit
                    agent.profit += todays_profit

                self.run_campaign_auctions(bids, new_camps, daily_campaigns_won)

                # Collect daily summary lines
                daily_lines = []

             
                profitagent = next(a for a in self.agents if a.name == "ProfitAgent")
                qscore = self.states[profitagent].quality_score

                line = f"Day {day} Summary for ProfitAgent:"
                print(f"\n📅 {line}")
                daily_lines.append(line)
                line = f"🧾 Campaigns up for auction: {len(new_camps)}"
                print(line)
                daily_lines.append(line)
                line = f"🎯 Campaigns won: {daily_campaigns_won['ProfitAgent']}"
                print(line)
                daily_lines.append(line)
                line = f"👥 Users auctioned: {len(users)}"
                print(line)
                daily_lines.append(line)
                line = f"🎯 Targetable users (in your segments): {targetable_users}"
                print(line)
                daily_lines.append(line)
                line = f"📢 Impressions won: {daily_impressions_won['ProfitAgent']}"
                print(line)
                daily_lines.append(line)
                rate = (daily_impressions_won['ProfitAgent'] / targetable_users * 100) \
                    if targetable_users > 0 else 0.0
                line = f"✅ Impression success rate: {rate:.2f}%"
                print(line)
                daily_lines.append(line)
                line = f"📊 Quality Score (Q): {qscore:.4f}"
                print(line)
                daily_lines.append(line)
                print("-" * 60)
                daily_lines.append("-" * 60)

                # —— dayily report—— #
                profit_state = self.states[profitagent]
                line = f"📝 Current tasks for {profitagent.name}:"
                print(line)
                daily_lines.append(line)
                for camp in profit_state.campaigns.values():
                    impressions = profit_state.impressions[camp.uid]
                    cost = profit_state.spend[camp.uid]
                    budget = profit_state.budgets[camp.uid]
                    progress = (impressions / camp.reach * 100) if camp.reach > 0 else 0.0
                    line = (f"  - Campaign {camp.uid}: segment={camp.target_segment}, "
                            f"days={camp.start_day}-{camp.end_day}, reach={camp.reach}, "
                            f"budget={budget:.2f}, done={impressions}/{camp.reach} ({progress:.1f}%), "
                            f"cost={cost:.2f}")
                    print(line)
                    daily_lines.append(line)
                print("-" * 60)
                daily_lines.append("-" * 60)

                # Append this day's summary to the report file
                with report_file.open("a") as f:
                    for l in daily_lines:
                        f.write(l + "\n")

            # final result
            for agent in self.agents:
                total_profits[agent.name] += self.states[agent].profits
            winner = max(self.agents, key=lambda a: self.states[a].profits)
            win_counts[winner.name] += 1
            self.print_game_results()

        
        print("\n================ FINAL SUMMARY OVER", num_simulations, "SIMULATIONS ================")
        print(f"{'Agent':<15} | {'Avg Profit':>10} | {'Wins':>5}")
        print("-" * 42)
        for name in sorted(total_profits, key=lambda n: total_profits[n], reverse=True):
            avg = total_profits[name] / num_simulations
            wins = win_counts[name]
            print(f"{name:<15} | {avg:>10.2f} | {wins:>5}")
        print("-" * 42)