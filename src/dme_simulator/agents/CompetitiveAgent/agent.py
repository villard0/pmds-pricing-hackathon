"""AIRIC competitive agent for the official Dynamic Market Exchange API."""

from __future__ import annotations

from collections import deque
from math import ceil, prod

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
    """Adaptive production specialist using only public observations.

    Optional constructor arguments exist solely for reproducible ablations in
    the local benchmark.  The official entry point remains constructible with
    no arguments.
    """

    def __init__(
        self,
        pricing_mode: str = "conservative",
        ladder_levels: int = 1,
        bid_aggressiveness: float = 1.0,
        price_factor: float = 1.0,
        use_joint_planning: bool = True,
        use_demand_learning: bool = True,
        tiered_bids: bool = True,
        adaptive_budget: bool = True,
    ) -> None:
        if pricing_mode not in {"robust", "conservative", "adaptive", "margin", "ucb", "gp", "demand", "hybrid"}:
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
        self._capital_peak: float = 0.0
        self._last_capital: float | None = None
        self._last_feedback_turn = 0
        self._last_market_test: dict[str, int] = {}
        self._recent_spend: deque[float] = deque(maxlen=8)
        self._recent_revenue: deque[float] = deque(maxlen=8)
        self._recent_sold: deque[int] = deque(maxlen=8)
        self._recent_offered: deque[int] = deque(maxlen=8)
        self._recent_capital_change: deque[float] = deque(maxlen=8)
        self._investment_share = 0.25
        self._specialist_good: str | None = None
        self._specialist_since_turn = 0

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
        transactions = state.auction_transactions[self._auction_count :]
        self._auction_count = len(state.auction_transactions)
        if self._team_id is None:
            return None
        return [item for item in transactions if item.buyer == self._team_id]

    def _own_sales_transactions(self, state: Observation):
        transactions = state.sales_transactions[self._sales_count :]
        self._sales_count = len(state.sales_transactions)
        if self._team_id is None:
            return None
        return [item for item in transactions if item.seller == self._team_id]

    def _consume_feedback(self, state: Observation) -> None:
        if state.turn <= self._last_feedback_turn:
            return
        self._resolve_team_id(state)
        auction_transactions = self._own_auction_transactions(state)
        turn_spend = 0.0
        if auction_transactions is not None:
            won: dict[str, int] = {}
            paid: dict[str, float] = {}
            for transaction in auction_transactions:
                turn_spend += transaction.price * transaction.quantity
                won[transaction.good] = won.get(transaction.good, 0) + transaction.quantity
                paid[transaction.good] = paid.get(transaction.good, 0.0) + (
                    transaction.price * transaction.quantity
                )
            for good in self.auctions.last_requested:
                quantity = won.get(good, 0)
                average = paid.get(good, 0.0) / quantity if quantity else 0.0
                self.auctions.observe(good, quantity, average)

        sales_transactions = self._own_sales_transactions(state)
        turn_revenue = 0.0
        turn_sold = 0
        turn_offered = 0
        if sales_transactions is not None and self._last_listings:
            sold: dict[str, int] = {}
            revenue: dict[str, float] = {}
            for transaction in sales_transactions:
                turn_revenue += transaction.price * transaction.quantity
                turn_sold += transaction.quantity
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
                turn_offered += offered
                self._last_market_test[good] = state.turn - 1
                observed_price = listed_price
                censored = offered > 0 and units >= offered
                if self.use_demand_learning:
                    self.demand.model(good).update(
                        observed_price, offered, units, censored=censored
                    )
                self.pricing.feedback(
                    good, offered=offered, sold=units, revenue=proceeds
                )
        if state.turn > 1:
            self._recent_spend.append(turn_spend)
            self._recent_revenue.append(turn_revenue)
            self._recent_sold.append(turn_sold)
            self._recent_offered.append(turn_offered)
            if self._last_capital is not None:
                self._recent_capital_change.append(state.capital - self._last_capital)
        self._capital_peak = max(self._capital_peak, state.capital)
        self._last_feedback_turn = state.turn

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

    def _investment_budget(
        self, state: Observation, goods_config: GoodsConfig
    ) -> float:
        """Smoothly scale exposure using realized operating evidence."""
        if not self.adaptive_budget:
            return state.capital * 0.30
        spend = sum(self._recent_spend)
        revenue = sum(self._recent_revenue)
        profit = revenue - spend
        roi = profit / spend if spend > 1e-9 else 0.0
        offered = sum(self._recent_offered)
        sell_through = sum(self._recent_sold) / offered if offered else 0.0
        peak = max(self._capital_peak, state.capital, 1.0)
        drawdown = max(0.0, (peak - state.capital) / peak)
        recent_trend = sum(self._recent_capital_change) / max(
            1.0, self._capital_anchor or state.capital
        )

        finished = sum(
            state.inventory.get(good, 0) for good in goods_config.finished_goods
        )
        raw = sum(
            state.inventory.get(good, 0) for good in goods_config.raw_materials
        )
        sales_rate = sum(self._recent_sold) / max(1, len(self._recent_sold))
        inventory_pressure = (finished + raw / max(1.0, self._average_recipe_size(goods_config))) / max(
            1.0, sales_rate * 3.0
        )

        target = 0.25
        if spend > 0:
            target += max(-0.12, min(0.14, roi * 0.16))
        if sell_through >= 0.65 and roi > 0:
            target += 0.08
        elif offered > 0 and sell_through < 0.20:
            target -= 0.09
        if recent_trend > 0.05:
            target += 0.05
        elif recent_trend < -0.04:
            target -= 0.07
        if inventory_pressure > 1.5:
            target -= min(0.12, 0.045 * inventory_pressure)
        if drawdown > 0.08:
            target -= min(0.15, drawdown * 0.55)
        if roi > 0.55 and sell_through > 0.70 and drawdown < 0.04:
            target = max(target, 0.46)
        target = max(0.10, min(0.52, target))
        self._investment_share = 0.72 * self._investment_share + 0.28 * target

        # Protect part of the current operation and accumulated peak, but never
        # freeze all profits. Drawdown raises the liquidity reserve smoothly.
        reserve_ratio = min(0.72, 0.48 + drawdown * 0.9 + max(0.0, inventory_pressure - 1.0) * 0.04)
        risk_capacity = state.capital * (1.0 - reserve_ratio)
        return max(0.0, min(state.capital * self._investment_share, risk_capacity))

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
            self._capital_peak = state.capital
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
        recipe_by_good: dict[str, str] = {}
        for recipe_name, recipe in goods_config.recipes.items():
            cost = self._recipe_cost(recipe, raw_costs)
            previous = recipe_costs.get(recipe.output_good)
            if previous is None or cost < previous:
                recipe_costs[recipe.output_good] = cost
                recipe_by_good[recipe.output_good] = recipe_name

        # Supply is not demand. Probe one unit; expand only after real sales.
        prior_per_good = 0.35

        prices: dict[str, float] = {}
        targets: dict[str, int] = {}
        expected_demand: dict[str, float] = {}
        expected_unit_revenue: dict[str, float] = {}
        limits: dict[str, int] = {}
        for good in goods_config.finished_goods:
            current = state.inventory.get(good, 0)
            model = self.demand.model(good)
            observed_scale = max(prior_per_good, model.recent_sales)
            per_good_limit = max(4, min(64, ceil(observed_scale * 1.8) + 2))
            limits[good] = per_good_limit
            competition = self._competition(state, good, prior_per_good)
            quote_stock = max(1, current)
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
            target = max(0, min(per_good_limit, target))
            if (
                target == 0
                and current == 0
                and model.should_reprobe(state.turn, sum(map(ord, good)))
            ):
                target = 1
                stale = state.turn - self._last_market_test.get(good, 0)
                recovery_weight = min(0.50, stale / 24.0)
                expected = (1.0 - recovery_weight) * expected + recovery_weight * prior_per_good
            targets[good] = target
            expected_demand[good] = expected
            sale_probability = min(1.0, expected / max(1, target))
            expected_unit_revenue[good] = price * sale_probability

        # Adaptive Specialist: compare one realistic three-unit production run
        # per product.  Completion risk, stale finished stock and recipe cost
        # matter alongside expected margin.  The incumbent changes only when a
        # challenger is clearly better, preventing turn-by-turn oscillation.
        specialist_scores: dict[str, float] = {}
        for good in goods_config.finished_goods:
            recipe_name = recipe_by_good.get(good)
            if recipe_name is None:
                continue
            recipe = goods_config.recipes[recipe_name]
            quantity = min(3, limits[good])
            fill_rates = []
            owned_units = 0
            required_units = 0
            for material, units in recipe.inputs.items():
                required = units * quantity
                owned = min(required, int(state.inventory.get(material, 0)))
                required_units += required
                owned_units += owned
                shortage = max(0, required - owned)
                lot = lot_by_good.get(material)
                fill_rates.append(
                    self.auctions.fill_probability(
                        material,
                        shortage,
                        lot.quantity if lot is not None else 0,
                        team_count,
                    )
                )
            completion = prod(fill_rates) if fill_rates else 1.0
            coverage = owned_units / max(1, required_units)
            unit_cost = recipe_costs.get(good, 0.0)
            margin = expected_unit_revenue.get(good, 0.0) - unit_cost
            model = self.demand.model(good)
            stale_stock = int(state.inventory.get(good, 0))
            inventory_risk = stale_stock * unit_cost * (
                0.20 + 0.55 * (1.0 - model.recent_sell_through)
            )
            completion_risk = (1.0 - completion) * unit_cost * quantity * 0.35
            specialist_scores[good] = (
                margin * quantity * (0.65 + 0.35 * completion)
                + coverage * unit_cost
                - inventory_risk
                - completion_risk
            )

        if specialist_scores:
            challenger = max(specialist_scores, key=specialist_scores.get)
            if self._specialist_good not in specialist_scores:
                self._specialist_good = challenger
                self._specialist_since_turn = state.turn
            elif challenger != self._specialist_good:
                incumbent_score = specialist_scores[self._specialist_good]
                challenger_score = specialist_scores[challenger]
                held_turns = state.turn - self._specialist_since_turn
                switch_margin = max(1.0, abs(incumbent_score) * 0.30)
                clearly_better = challenger_score - incumbent_score > switch_margin
                if held_turns >= 4 and clearly_better:
                    self._specialist_good = challenger
                    self._specialist_since_turn = state.turn

        specialist = self._specialist_good
        if specialist is not None:
            model = self.demand.model(specialist)
            current = int(state.inventory.get(specialist, 0))
            limit = limits[specialist]
            target = targets[specialist]
            if model.observations == 0:
                target = min(limit, 3)
            elif model.zero_streak:
                # A miss immediately stops expansion. Existing stock is sold
                # down; an empty line receives at most the scheduled probe.
                target = current if current > 0 else min(target, 1)
            elif model.recent_sales > 0:
                if model.recent_sales >= 5:
                    target = ceil(model.recent_sales * 1.65) + 1
                elif model.recent_sales >= 3:
                    target = 8
                elif model.recent_sales >= 1:
                    target = 5
                target = max(target, current)
            target = max(0, min(limit, target))
            targets = {
                good: int(state.inventory.get(good, 0))
                for good in goods_config.finished_goods
            }
            targets[specialist] = target

        primary_units = max(
            0,
            targets.get(specialist, 0) - int(state.inventory.get(specialist, 0)),
        ) if specialist is not None else 0
        total_target = max(1, min(64, primary_units))
        planning_budget = self._investment_budget(state, goods_config)
        availability = {lot.good: lot.quantity for lot in state.auction_lots}
        desired_materials = {good: 0 for good in goods_config.raw_materials}
        specialist_recipes = {
            name: recipe
            for name, recipe in goods_config.recipes.items()
            if recipe.output_good == specialist
        }
        for recipe in specialist_recipes.values():
            quantity = max(
                0,
                targets.get(recipe.output_good, 0)
                - int(state.inventory.get(recipe.output_good, 0)),
            )
            for material, amount in recipe.inputs.items():
                desired_materials[material] += amount * quantity
        fill_probabilities = {
            lot.good: self.auctions.fill_probability(
                lot.good,
                max(1, desired_materials.get(lot.good, 0) - state.inventory.get(lot.good, 0)),
                lot.quantity,
                team_count,
            )
            for lot in state.auction_lots
        }
        marginal_values = {}
        for good in goods_config.finished_goods:
            model = self.demand.model(good)
            stock = int(state.inventory.get(good, 0))
            marginal_values[good] = [
                prices[good] * model.sale_probability(stock + k, prices[good], prior_per_good)
                for k in range(1, limits[good] + 1)
            ]
        if self.use_joint_planning:
            plan = joint_plan(
                specialist_recipes,
                state.inventory,
                raw_costs,
                state.inventory,
                targets,
                expected_unit_revenue,
                planning_budget,
                total_target,
                available=availability,
                marginal_values=marginal_values,
                max_active_goods=1,
            )
        else:
            plan = greedy_plan(
                specialist_recipes,
                state.inventory,
                raw_costs,
                state.inventory,
                targets,
                expected_unit_revenue,
                planning_budget,
                total_target,
            )

        # Secondary production may consume only raw materials that were already
        # owned and are not reserved for the specialist plan.  It never creates
        # extra auction requirements, so new cash remains concentrated on the
        # primary product while stranded stock can still be monetized.
        combined_quantities = dict(plan.quantities)
        residual_inventory = dict(state.inventory)
        for material in goods_config.raw_materials:
            reserved = min(
                int(state.inventory.get(material, 0)),
                int(plan.requirements.get(material, 0)),
            )
            residual_inventory[material] = max(
                0, int(state.inventory.get(material, 0)) - reserved
            )
        secondary_recipes = {
            name: recipe
            for name, recipe in goods_config.recipes.items()
            if recipe.output_good != specialist
        }
        if secondary_recipes:
            secondary_targets = {
                good: int(state.inventory.get(good, 0)) + 2
                for good in goods_config.finished_goods
                if good != specialist
            }
            secondary = joint_plan(
                secondary_recipes,
                residual_inventory,
                raw_costs,
                state.inventory,
                secondary_targets,
                expected_unit_revenue,
                0.0,
                4,
                available={material: 0 for material in goods_config.raw_materials},
                marginal_values=marginal_values,
                max_active_goods=2,
            )
            for recipe_name, quantity in secondary.quantities.items():
                if quantity > 0:
                    combined_quantities[recipe_name] = quantity

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
                combined_quantities.items(),
                key=lambda item: (
                    goods_config.recipes[item[0]].output_good != specialist,
                    -margins.get(item[0], 0.0),
                    item[0],
                ),
            )
            if quantity > 0
        ]

        planned_finished = {
            good: int(state.inventory.get(good, 0))
            for good in goods_config.finished_goods
        }
        for recipe_name, quantity in combined_quantities.items():
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
        self._last_capital = state.capital
        return Decision(bids=bids, assembly=assembly, listings=listings)
