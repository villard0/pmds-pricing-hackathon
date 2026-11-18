from __future__ import annotations

import time
import unittest
from pathlib import Path

from dme_simulator.agents import AuctionLot, Observation
from dme_simulator.agents.loader import load_agent_bundle_class
from dme_simulator.config import GoodsConfig, RecipeConfig
from dme_simulator.runner import _validate_decision


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "src" / "dme_simulator" / "agents" / "CompetitiveAgent"


class CompetitiveAgentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.agent_class = load_agent_bundle_class(BUNDLE)
        cls.goods = GoodsConfig(
            raw_materials=["A", "B"],
            finished_goods=["X"],
            recipes={"MAKE_X": RecipeConfig("X", {"A": 1, "B": 2})},
        )

    def observation(self, *, turn=1, capital=1000.0, inventory=None):
        inventory = inventory or {"A": 0, "B": 0, "X": 0}
        return Observation(
            turn=turn,
            capital=capital,
            inventory=inventory,
            bankrupt=False,
            auction_lots=[AuctionLot("A", 12, 4.0), AuctionLot("B", 20, 2.0)],
            public_inventories={"self": dict(inventory), "rival": {"A": 0, "B": 0, "X": 0}},
            public_capital={"self": capital, "rival": 1000.0},
        )

    def test_bundle_has_one_constructible_agent(self):
        self.assertIsNotNone(self.agent_class())

    def test_decision_is_valid_and_budget_safe(self):
        agent = self.agent_class()
        state = self.observation()
        decision = agent.decide(state, self.goods)
        _validate_decision(decision, state, self.goods)
        committed = sum(bid.quantity * bid.price_per_unit for bid in decision.bids)
        self.assertLessEqual(committed, state.capital * 0.56 + 1e-7)

    def test_empty_capital_is_safe(self):
        agent = self.agent_class()
        state = self.observation(capital=0.0)
        decision = agent.decide(state, self.goods)
        _validate_decision(decision, state, self.goods)
        self.assertEqual(sum(bid.quantity for bid in decision.bids), 0)

    def test_decision_runtime_has_large_headroom(self):
        durations = []
        for turn in range(1, 31):
            agent = self.agent_class()
            state = self.observation(turn=turn)
            started = time.perf_counter()
            decision = agent.decide(state, self.goods)
            durations.append(time.perf_counter() - started)
            _validate_decision(decision, state, self.goods)
        self.assertLess(max(durations), 0.05)


if __name__ == "__main__":
    unittest.main()
