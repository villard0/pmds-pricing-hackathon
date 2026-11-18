import sys
import unittest
from copy import deepcopy
from pathlib import Path
from dme_simulator.agents import AuctionLot, Observation, SalesPrice
from dme_simulator.agents.loader import load_agent_bundle_class
from dme_simulator.config import GoodsConfig, RecipeConfig
from dme_simulator.runner import _validate_decision

ROOT=Path(__file__).resolve().parents[1]


class GrowthAgentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.agent_class=load_agent_bundle_class(ROOT/'src/dme_simulator/agents/CompetitiveAgent')
        namespace=cls.agent_class.__module__
        cls.model_class=sys.modules[namespace+'.demand'].DemandEstimator
        cls.auction_class=sys.modules[namespace+'.auctions'].AuctionModel
        cls.plan=staticmethod(sys.modules[namespace+'.planning'].joint_plan)
        cls.goods=GoodsConfig(['A','B'],['X'],{'X':RecipeConfig('X',{'A':1,'B':2})})

    def observation(self,cash=1000,stock=0,turn=1):
        inventory={'A':0,'B':0,'X':stock}
        return Observation(turn,cash,inventory,False,[AuctionLot('A',12,4),AuctionLot('B',20,2)],
                           {'self':dict(inventory),'rival':{'A':1,'B':0,'X':0}},
                           {'self':cash,'rival':1000})

    def test_initial_exposure_is_controlled_without_97_percent_lock(self):
        agent=self.agent_class(); obs=self.observation(cash=1000)
        action=agent.decide(obs,self.goods)
        spend=sum(b.quantity*b.price_per_unit for b in action.bids)
        self.assertLessEqual(spend,350)
        self.assertGreater(agent._investment_budget(obs,self.goods),30)

    def test_profitable_sales_increase_smoothed_investment_budget(self):
        agent=self.agent_class(); obs=self.observation(cash=1000)
        agent._capital_anchor=1000; agent._capital_peak=1000
        initial=agent._investment_budget(obs,self.goods)
        for _ in range(6):
            agent._recent_spend.append(100)
            agent._recent_revenue.append(180)
            agent._recent_sold.append(5)
            agent._recent_offered.append(6)
            agent._recent_capital_change.append(80)
        grown=agent._investment_budget(self.observation(cash=1480),self.goods)
        self.assertGreater(grown,initial)

    def test_drawdown_and_inventory_reduce_exposure(self):
        agent=self.agent_class(); agent._capital_anchor=1000; agent._capital_peak=1600
        for _ in range(6):
            agent._recent_spend.append(100)
            agent._recent_revenue.append(40)
            agent._recent_sold.append(0)
            agent._recent_offered.append(4)
            agent._recent_capital_change.append(-60)
        healthy=agent._investment_budget(self.observation(cash=1600),self.goods)
        burdened=self.observation(cash=1100,stock=20)
        reduced=agent._investment_budget(burdened,self.goods)
        self.assertLess(reduced,healthy)

    def test_zero_demand_allows_no_restock(self):
        model=self.model_class()
        for _ in range(10): model.update(100,1,0,censored=False)
        self.assertEqual(model.target_stock(100,.35,12),0)
        self.assertLess(model.sale_probability(1,100,.35),.1)

    def test_dormant_product_is_periodically_reopened(self):
        model=self.model_class()
        for _ in range(4): model.update(100,1,0,censored=False)
        self.assertTrue(any(model.should_reprobe(turn,3) for turn in range(1,20)))

    def test_repeated_sales_raise_target_above_probe_quantity(self):
        model=self.model_class()
        for _ in range(5): model.update(100,5,4,censored=False)
        self.assertGreater(model.target_stock(100,.35,64),1)

    def test_one_zero_does_not_create_hard_price_boundary(self):
        model=self.model_class()
        model.update(100,3,0,censored=False)
        self.assertGreater(model.expected_demand(100,.35),.08)

    def test_candidate_demand_curve_is_non_increasing(self):
        model=self.model_class()
        model.update(80,3,2,censored=False)
        model.update(160,3,0,censored=False)
        values=model.demand_curve([40,80,120,160,240],.35)
        self.assertEqual(values,sorted(values,reverse=True))

    def test_marginal_probability_decreases_with_stock(self):
        model=self.model_class()
        probabilities=[model.sale_probability(k,100,.35) for k in range(1,5)]
        self.assertEqual(probabilities,sorted(probabilities,reverse=True))
        self.assertLess(probabilities[1],probabilities[0])

    def test_no_supply_prevents_incomplete_new_portfolio(self):
        plan=self.plan(self.goods.recipes,{}, {'A':4,'B':2},{},{'X':3},{'X':100},100,4,
                       available={'A':10,'B':0},marginal_values={'X':[100,50,10]})
        self.assertEqual(sum(plan.quantities.values()),0)

    def test_marginal_revenue_stops_unprofitable_second_unit(self):
        plan=self.plan(self.goods.recipes,{}, {'A':4,'B':2},{},{'X':3},{'X':100},100,4,
                       available={'A':10,'B':10},marginal_values={'X':[20,2,1]})
        self.assertEqual(plan.quantities['X'],1)

    def test_low_joint_fill_probability_avoids_stranded_complements(self):
        plan=self.plan(self.goods.recipes,{}, {'A':4,'B':2},{},{'X':1},{'X':12},100,1,
                       available={'A':10,'B':10},marginal_values={'X':[12]},
                       fill_probabilities={'A':.9,'B':.1})
        self.assertEqual(sum(plan.quantities.values()),0)

    def test_alternative_recipes_share_one_product_target(self):
        recipes={'one':RecipeConfig('X',{'A':1}),'two':RecipeConfig('X',{'B':1})}
        plan=self.plan(recipes,{}, {'A':1,'B':1},{},{'X':1},{'X':100},100,4,
                       available={'A':10,'B':10},marginal_values={'X':[100,100]})
        self.assertEqual(sum(plan.quantities.values()),1)

    def test_auction_rejects_reserve_above_value(self):
        model=self.auction_class()
        rows=model.build_tiers({'A':1},{},[AuctionLot('A',10,10)],{'A':5},100,4,False)
        self.assertEqual(rows,[])

    def test_budget_reduction_never_splits_complements(self):
        model=self.auction_class()
        rows=model.build_tiers({'A':1,'B':2},{},[AuctionLot('A',10,4),AuctionLot('B',10,2)],
                               {'A':100,'B':100},1,4,False)
        self.assertEqual(rows,[])

    def test_zero_reserve_produces_finite_valid_bid(self):
        model=self.auction_class()
        rows=model.build_tiers({'A':1},{},[AuctionLot('A',10,0)],{'A':1},1,4,False)
        self.assertTrue(rows)
        self.assertGreater(rows[0][2],0)

    def test_inputs_unchanged_and_single_price_per_good(self):
        agent=self.agent_class(); obs=self.observation(stock=5)
        snapshot=deepcopy(obs)
        action=agent.decide(obs,self.goods)
        _validate_decision(action,obs,self.goods)
        self.assertEqual(obs,snapshot)
        self.assertEqual(len(action.listings),1)

    def test_stale_inventory_can_be_sold_below_replacement_cost(self):
        agent=self.agent_class(); model=agent.demand.model('X')
        for _ in range(6): model.update(1,3,0,censored=False)
        price=agent.pricing.quote('X',10,3,model,.35,0,7)
        self.assertLess(price,10)

    def test_pricing_never_uses_extreme_legacy_multiplier(self):
        agent=self.agent_class(); model=agent.demand.model('X')
        price=agent.pricing.quote('X',10,0,model,.35,0,1)
        self.assertLessEqual(price,120)

    def test_cumulative_sales_are_not_counted_twice(self):
        agent=self.agent_class(); agent.decide(self.observation(stock=1),self.goods)
        price=agent._last_listings['X'][1]
        state=self.observation(cash=1100,turn=2)
        state.sales_transactions.append(SalesPrice('X',1,price,'self'))
        agent._consume_feedback(state)
        first=agent.demand.model('X').last.sold
        # Same cumulative ledger in the next turn means zero additional sales.
        agent._consume_feedback(state)
        self.assertEqual(first,1)
        self.assertEqual(agent._sales_count,1)


if __name__=='__main__': unittest.main()
