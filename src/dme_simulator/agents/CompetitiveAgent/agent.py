"""AIRIC competitive agent for the official Dynamic Market Exchange API."""

from __future__ import annotations

from math import ceil

from dme_simulator.agents import (
    AssemblyOrder,
    BaseAgent,
    Bid,
    Decision,
    Listing,
    Observation,
)
from dme_simulator.config import GoodsConfig

from .auctions import AuctionModel
from .demand import DemandBook
from .planning import greedy_plan, joint_plan
from .pricing import PricingEngine


class CompetitiveAgent(BaseAgent):
    """Adaptive joint planner using only public observations.

    Optional constructor arguments exist solely for reproducible ablations in
    the local benchmark.  The official entry point remains constructible with
    no arguments.
    """

    def __init__(
        self,
        pricing_mode: str = "hybrid",
        ladder_levels: int = 3,
        bid_aggressiveness: float = 1.0,
        price_factor: float = 1.0,
        use_joint_planning: bool = True,
        use_demand_learning: bool = True,
        tiered_bids: bool = True,
        adaptive_budget: bool = True,
    ) -> None:
        if pricing_mode not in {"adaptive", "margin", "ucb", "gp", "demand", "hybrid"}:
            raise ValueError(f"unsupported pricing mode: {pricing_mode}")
        if ladder_levels not in {0, 1, 2, 3}:
            raise ValueError("ladder_levels must be 0, 1, 2, or 3")
        self.demand = DemandBook()
        self._adaptive_pricing = pricing_mode == "adaptive"
        self._adaptive_ladder = ladder_levels == 0
        self.pricing = PricingEngine(
            "hybrid" if self._adaptive_pricing else pricing_mode,
            1 if self._adaptive_ladder else ladder_levels,
            price_factor,
        )
        self.auctions = AuctionModel(bid_aggressiveness)
        self.use_joint_planning = bool(use_joint_planning)
        self.use_demand_learning = bool(use_demand_learning)
        self.tiered_bids = bool(tiered_bids)
        self.adaptive_budget = bool(adaptive_budget)
        self._team_id: str | None = None
        self._auction_count = 0
        self._sales_count = 0
        self._last_listings: dict[str, tuple[int, float]] = {}
        self._capital_anchor: float | None = None

    # ------------------------------------------------------------------
    # Observation and feedback
    # ------------------------------------------------------------------

    def _resolve_team_id(self, state: Observation) -> None:
        if self._team_id in state.public_capital:
            return
        candidates = [
            team_id
            for team_id, capital in state.public_capital.items()
            if abs(float(capital) - float(state.capital)) <= 1e-7
            and state.public_inventories.get(team_id) == state.inventory
        ]
        if len(candidates) == 1:
            self._team_id = candidates[0]
            return
        # With the default visibility policy every visible transaction is our
        # own. A unique actor therefore resolves otherwise identical starts.
        actors = {
            transaction.buyer for transaction in state.auction_transactions
        } | {transaction.seller for transaction in state.sales_transactions}
        matching = actors.intersection(candidates)
        if len(matching) == 1 and len(actors) == 1:
            self._team_id = next(iter(matching))

    def _own_auction_transactions(self, state: Observation):
        if self._team_id is None:
            return None
        transactions = state.auction_transactions[self._auction_count :]
        self._auction_count = len(state.auction_transactions)
        return [item for item in transactions if item.buyer == self._team_id]

    def _own_sales_transactions(self, state: Observation):
        if self._team_id is None:
            return None
        transactions = state.sales_transactions[self._sales_count :]
        self._sales_count = len(state.sales_transactions)
        return [item for item in transactions if item.seller == self._team_id]

    def _consume_feedback(self, state: Observation) -> None:
        self._resolve_team_id(state)
        auction_transactions = self._own_auction_transactions(state)
        if auction_transactions is not None:
            won: dict[str, int] = {}
            paid: dict[str, float] = {}
            for transaction in auction_transactions:
                won[transaction.good] = won.get(transaction.good, 0) + transaction.quantity
                paid[transaction.good] = paid.get(transaction.good, 0.0) + (
                    transaction.price * transaction.quantity
                )
            for good in self.auctions.last_requested:
                quantity = won.get(good, 0)
                average = paid.get(good, 0.0) / quantity if quantity else 0.0
                self.auctions.observe(good, quantity, average)

        sales_transactions = self._own_sales_transactions(state)
        if sales_transactions is not None and self._last_listings:
            sold: dict[str, int] = {}
            revenue: dict[str, float] = {}
            for transaction in sales_transactions:
                sold[transaction.good] = sold.get(transaction.good, 0) + transaction.quantity
                revenue[transaction.good] = revenue.get(transaction.good, 0.0) + (
                    transaction.price * transaction.quantity
                )
            for good, (offered, listed_price) in self._last_listings.items():
                units = sold.get(good, 0)
                proceeds = revenue.get(good, 0.0)
                # We list all finished stock. At the next turn, remaining public
                # inventory plus sales is therefore the quantity that really
                # existed after production; planned listings may have been
                # capped when an auction filled only partially.
                offered = min(
                    offered,
                    int(state.inventory.get(good, 0)) + units,
                )
                units = min(offered, units)
                if offered <= 0:
                    continue
                observed_price = proceeds / units if units else listed_price
                censored = offered > 0 and units >= offered
                if self.use_demand_learning:
                    self.demand.model(good).update(
                        observed_price, offered, units, censored=censored
                    )
                self.pricing.feedback(
                    good, offered=offered, sold=units, revenue=proceeds
                )

    # ------------------------------------------------------------------
    # Market estimates
    # ------------------------------------------------------------------

    @staticmethod
    def _recipe_cost(recipe, raw_costs: dict[str, float]) -> float:
        return sum(
            raw_costs.get(material, 0.0) * units
            for material, units in recipe.inputs.items()
        )

    @staticmethod
    def _average_recipe_size(goods_config: GoodsConfig) -> float:
        sizes = [sum(recipe.inputs.values()) for recipe in goods_config.recipes.values()]
        return sum(sizes) / max(1, len(sizes))

    def _budget_share(self, state: Observation, team_count: int) -> float:
        if not self.adaptive_budget:
            return 0.45
        crowding = max(0, team_count - 4)
        share = 0.50 - min(0.16, crowding * 0.008)
        if self._capital_anchor is not None and state.capital < 0.55 * self._capital_anchor:
            share -= 0.08
        raw_stock = sum(
            state.inventory.get(lot.good, 0) for lot in state.auction_lots
        )
        if raw_stock > 2 * max(1, len(state.auction_lots)):
            share -= 0.06
        return max(0.24, min(0.56, share))

    def _competition(self, state: Observation, good: str, prior: float) -> float:
        if len(state.public_inventories) <= 1:
            return 0.0
        visible = sum(
            inventory.get(good, 0)
            for team_id, inventory in state.public_inventories.items()
            if team_id != self._team_id
        )
        average = visible / max(1, len(state.public_inventories) - 1)
        return average / max(1.0, prior)

    @staticmethod
    def _material_value_caps(
        plan,
        goods_config: GoodsConfig,
        raw_costs: dict[str, float],
        expected_unit_revenue: dict[str, float],
    ) -> dict[str, float]:
        caps: dict[str, float] = {}
        for recipe_name, quantity in plan.quantities.items():
            if quantity <= 0:
                continue
            recipe = goods_config.recipes[recipe_name]
            value = expected_unit_revenue.get(recipe.output_good, 0.0)
            for material, units in recipe.inputs.items():
                other_cost = sum(
                    raw_costs.get(other, 0.0) * amount
                    for other, amount in recipe.inputs.items()
                    if other != material
                )
                marginal = max(0.0, (value - other_cost) / units * 0.92)
                caps[material] = max(caps.get(material, 0.0), marginal)
        return caps

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def decide(self, state: Observation, goods_config: GoodsConfig) -> Decision:
        if self._capital_anchor is None:
            self._capital_anchor = max(1.0, state.capital)
        self._consume_feedback(state)

        team_count = max(1, len(state.public_capital))
        # The public tutorials expose why one fixed pricing policy is brittle:
        # a single product benefits from learning/segmentation, while multiple
        # simultaneous products need stable prices and less exploration under
        # competition. This switch uses only the public GoodsConfig shape.
        if self._adaptive_pricing:
            self.pricing.mode = (
                "hybrid" if len(goods_config.finished_goods) == 1 else "margin"
            )
        if self._adaptive_ladder:
            self.pricing.ladder_levels = (
                3 if len(goods_config.finished_goods) == 1 else 1
            )
        lot_by_good = {lot.good: lot for lot in state.auction_lots}
        raw_costs: dict[str, float] = {}
        for material in goods_config.raw_materials:
            lot = lot_by_good.get(material)
            reserve = lot.reserve_price if lot is not None else 0.01
            raw_costs[material] = self.auctions.expected_price(
                material, max(0.01, reserve), 0.8
            )

        recipe_costs: dict[str, float] = {}
        for recipe in goods_config.recipes.values():
            cost = self._recipe_cost(recipe, raw_costs)
            previous = recipe_costs.get(recipe.output_good)
            if previous is None or cost < previous:
                recipe_costs[recipe.output_good] = cost

        average_inputs = self._average_recipe_size(goods_config)
        per_team_raw = sum(lot.quantity for lot in state.auction_lots) / team_count
        total_target = max(2, min(24, int(per_team_raw / max(1.0, average_inputs) + 0.5)))
        good_count = max(1, len(goods_config.finished_goods))
        prior_per_good = max(0.75, total_target / good_count)
        per_good_limit = max(2, min(12, ceil(2.2 * prior_per_good)))

        prices: dict[str, float] = {}
        targets: dict[str, int] = {}
        expected_unit_revenue: dict[str, float] = {}
        for good in goods_config.finished_goods:
            current = state.inventory.get(good, 0)
            model = self.demand.model(good)
            competition = self._competition(state, good, prior_per_good)
            quote_stock = max(1, current + per_good_limit // 2)
            price = self.pricing.quote(
                good,
                recipe_costs.get(good, 1.0),
                quote_stock,
                model,
                prior_per_good,
                competition,
                state.turn,
            )
            prices[good] = price
            if self.use_demand_learning:
                target = model.target_stock(price, prior_per_good, per_good_limit)
                expected = model.expected_demand(price, prior_per_good)
            else:
                target = max(1, min(per_good_limit, ceil(prior_per_good)))
                expected = prior_per_good
            # Visible rival stock reduces new exposure, never below one unit.
            target = max(1, min(per_good_limit, int(target / (1.0 + 0.28 * competition) + 0.75)))
            targets[good] = target
            sale_probability = max(0.28, min(1.0, expected / max(1, target)))
            expected_unit_revenue[good] = price * sale_probability

        budget_share = self._budget_share(state, team_count)
        planning_budget = max(0.0, state.capital * budget_share)
        if self.use_joint_planning:
            plan = joint_plan(
                goods_config.recipes,
                state.inventory,
                raw_costs,
                state.inventory,
                targets,
                expected_unit_revenue,
                planning_budget,
                total_target,
            )
        else:
            plan = greedy_plan(
                goods_config.recipes,
                state.inventory,
                raw_costs,
                state.inventory,
                targets,
                expected_unit_revenue,
                planning_budget,
                total_target,
            )

        value_caps = self._material_value_caps(
            plan, goods_config, raw_costs, expected_unit_revenue
        )
        worst_case_budget = min(
            planning_budget,
            max(plan.new_cost * 1.18, plan.new_cost + 1.0),
        )
        bid_rows = self.auctions.build_tiers(
            plan.requirements,
            state.inventory,
            state.auction_lots,
            value_caps,
            worst_case_budget,
            team_count,
            self.tiered_bids,
        )
        bids = [Bid(good, quantity, price) for good, quantity, price in bid_rows]

        margins: dict[str, float] = {}
        for recipe_name, recipe in goods_config.recipes.items():
            margins[recipe_name] = expected_unit_revenue.get(recipe.output_good, 0.0) - self._recipe_cost(recipe, raw_costs)
        assembly = [
            AssemblyOrder(recipe_name, quantity)
            for recipe_name, quantity in sorted(
                plan.quantities.items(),
                key=lambda item: (-margins.get(item[0], 0.0), item[0]),
            )
            if quantity > 0
        ]

        planned_finished = {
            good: int(state.inventory.get(good, 0))
            for good in goods_config.finished_goods
        }
        for recipe_name, quantity in plan.quantities.items():
            output = goods_config.recipes[recipe_name].output_good
            planned_finished[output] = planned_finished.get(output, 0) + quantity

        listings: list[Listing] = []
        listing_memory: dict[str, tuple[int, float]] = {}
        for good in goods_config.finished_goods:
            quantity = planned_finished.get(good, 0)
            if quantity <= 0:
                continue
            ladder = self.pricing.ladder(prices[good], quantity)
            value = 0.0
            offered = 0
            for units, price in ladder:
                if units <= 0:
                    continue
                listings.append(Listing(good, units, price))
                value += units * price
                offered += units
            if offered:
                listing_memory[good] = (offered, value / offered)
        self._last_listings = listing_memory
        return Decision(bids=bids, assembly=assembly, listings=listings)
