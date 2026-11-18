"""Adaptive scale-free pricing and optional price ladders."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import exp, log, sqrt

from .demand import DemandEstimator


# Multipliers are relative to the currently estimated replacement cost.  This
# makes the price grid portable across private scenarios with different scales.
MARKUPS = (1.50, 2.20, 3.20, 4.80, 7.20, 10.80, 16.20, 24.30, 36.50)


@dataclass(slots=True)
class PriceChoice:
    arm: int
    price: float
    unit_cost: float
    offered: int


class SlidingWindowUCB:
    def __init__(self, arms: int, window: int = 20) -> None:
        self._rewards = [deque(maxlen=window) for _ in range(arms)]
        self._clock = 0

    def update(self, arm: int, reward: float) -> None:
        if 0 <= arm < len(self._rewards):
            self._rewards[arm].append(max(0.0, min(1.0, float(reward))))
            self._clock += 1

    def values(self) -> list[float]:
        total = max(2, self._clock + 1)
        values: list[float] = []
        for rewards in self._rewards:
            if not rewards:
                values.append(1.5)
                continue
            mean = sum(rewards) / len(rewards)
            bonus = sqrt(1.35 * log(total) / len(rewards))
            values.append(mean + bonus)
        return values

    def select(self) -> int:
        values = self.values()
        return max(range(len(values)), key=lambda index: (values[index], -index))


class GaussianProcessUCB:
    """Tiny pure-Python GP-UCB over the fixed, scale-free price grid."""

    def __init__(self, window: int = 20) -> None:
        self._history: deque[tuple[int, float]] = deque(maxlen=window)
        logs = [log(value) for value in MARKUPS]
        span = logs[-1] - logs[0]
        self._coordinates = [(value - logs[0]) / span for value in logs]

    @property
    def observations(self) -> int:
        return len(self._history)

    @staticmethod
    def _kernel(left: float, right: float) -> float:
        distance = (left - right) / 0.24
        return exp(-0.5 * distance * distance)

    @staticmethod
    def _cholesky(matrix: list[list[float]]) -> list[list[float]]:
        size = len(matrix)
        lower = [[0.0] * size for _ in range(size)]
        for row in range(size):
            for column in range(row + 1):
                value = matrix[row][column] - sum(
                    lower[row][k] * lower[column][k] for k in range(column)
                )
                if row == column:
                    lower[row][column] = sqrt(max(value, 1e-10))
                else:
                    lower[row][column] = value / lower[column][column]
        return lower

    @staticmethod
    def _solve(lower: list[list[float]], values: list[float]) -> list[float]:
        size = len(values)
        forward = [0.0] * size
        for row in range(size):
            forward[row] = (
                values[row]
                - sum(lower[row][column] * forward[column] for column in range(row))
            ) / lower[row][row]
        result = [0.0] * size
        for row in range(size - 1, -1, -1):
            result[row] = (
                forward[row]
                - sum(lower[column][row] * result[column] for column in range(row + 1, size))
            ) / lower[row][row]
        return result

    def update(self, arm: int, reward: float) -> None:
        if 0 <= arm < len(MARKUPS):
            self._history.append((arm, max(0.0, min(1.0, float(reward)))))

    def values(self) -> list[float]:
        if not self._history:
            return [1.0] * len(MARKUPS)
        coordinates = [self._coordinates[arm] for arm, _ in self._history]
        rewards = [reward for _, reward in self._history]
        matrix = [
            [self._kernel(left, right) for right in coordinates]
            for left in coordinates
        ]
        for index in range(len(matrix)):
            matrix[index][index] += 0.075
        lower = self._cholesky(matrix)
        alpha = self._solve(lower, rewards)
        scores: list[float] = []
        for candidate in self._coordinates:
            vector = [self._kernel(candidate, observed) for observed in coordinates]
            mean = sum(weight * value for weight, value in zip(vector, alpha))
            projection = self._solve(lower, vector)
            variance = max(
                1e-8,
                1.0 - sum(weight * value for weight, value in zip(vector, projection)),
            )
            scores.append(mean + 1.35 * sqrt(variance))
        return scores

    def select(self) -> int:
        values = self.values()
        return max(range(len(values)), key=lambda index: (values[index], -index))


class PricingEngine:
    """Combines a demand curve with sliding-window exploration."""

    def __init__(
        self,
        mode: str = "hybrid",
        ladder_levels: int = 1,
        price_factor: float = 1.0,
    ) -> None:
        if mode not in {"margin", "ucb", "gp", "demand", "hybrid"}:
            raise ValueError(f"unsupported pricing mode: {mode}")
        if ladder_levels not in {1, 2, 3}:
            raise ValueError("ladder_levels must be 1, 2, or 3")
        self.mode = mode
        self.ladder_levels = ladder_levels
        self.price_factor = max(0.55, min(1.6, float(price_factor)))
        self._learners: dict[str, SlidingWindowUCB] = {}
        self._gp_learners: dict[str, GaussianProcessUCB] = {}
        self._last: dict[str, PriceChoice] = {}

    def _learner(self, good: str) -> SlidingWindowUCB:
        learner = self._learners.get(good)
        if learner is None:
            learner = self._learners[good] = SlidingWindowUCB(len(MARKUPS))
        return learner

    def _gp_learner(self, good: str) -> GaussianProcessUCB:
        learner = self._gp_learners.get(good)
        if learner is None:
            learner = self._gp_learners[good] = GaussianProcessUCB()
        return learner

    def quote(
        self,
        good: str,
        unit_cost: float,
        stock: int,
        demand: DemandEstimator,
        prior_demand: float,
        competition: float,
        turn: int,
    ) -> float:
        unit_cost = max(0.01, float(unit_cost))
        # High visible finished-goods stock suggests price competition.  The
        # bounded adjustment avoids a collapse toward zero in crowded markets.
        pressure_factor = max(0.72, min(1.05, 1.02 - 0.055 * competition))
        candidates = [
            unit_cost * markup * pressure_factor * self.price_factor
            for markup in MARKUPS
        ]
        learner = self._learner(good)

        if self.mode == "margin":
            arm = 3
        elif self.mode == "ucb":
            arm = learner.select()
        elif self.mode == "gp":
            gp = self._gp_learner(good)
            if gp.observations == 0:
                order = (3, 5, 7, 2, 6, 4, 8, 1, 0)
                arm = order[(turn - 1 + sum(map(ord, good))) % len(order)]
            else:
                arm = gp.select()
        else:
            demand_scores = [
                demand.expected_revenue(price, max(1, stock), prior_demand)
                for price in candidates
            ]
            demand_peak = max(demand_scores) or 1.0
            demand_values = [value / demand_peak for value in demand_scores]
            if self.mode == "demand":
                arm = max(
                    range(len(candidates)),
                    key=lambda index: (demand_values[index], -index),
                )
            else:
                ucb_values = learner.values()
                ucb_peak = max(ucb_values) or 1.0
                # Cold starts deliberately sample separated points instead of
                # marching monotonically through a price grid.
                if demand.observations < 2:
                    order = (3, 5, 7, 2, 6, 4, 8, 1, 0)
                    arm = order[(turn - 1 + sum(map(ord, good))) % len(order)]
                else:
                    arm = max(
                        range(len(candidates)),
                        key=lambda index: (
                            0.72 * demand_values[index]
                            + 0.28 * ucb_values[index] / ucb_peak,
                            -index,
                        ),
                    )

        price = max(unit_cost * 1.08, candidates[arm])
        self._last[good] = PriceChoice(arm, price, unit_cost, max(0, int(stock)))
        return price

    def feedback(
        self,
        good: str,
        *,
        offered: int,
        sold: int,
        revenue: float,
    ) -> None:
        choice = self._last.get(good)
        if choice is None or offered <= 0:
            return
        # Revenue per offered unit, normalized by the highest candidate price,
        # rewards both conversion and price without depending on currency scale.
        ceiling = max(choice.unit_cost * MARKUPS[-1], choice.price)
        reward = max(0.0, revenue) / (max(1, offered) * ceiling)
        self._learner(good).update(choice.arm, reward)
        self._gp_learner(good).update(choice.arm, reward)

    def ladder(self, price: float, quantity: int) -> list[tuple[int, float]]:
        """Split stock into one, two, or three price bands."""
        quantity = max(0, int(quantity))
        if quantity == 0:
            return []
        if self.ladder_levels == 1 or quantity < 3:
            return [(quantity, price)]
        if self.ladder_levels == 2:
            low = max(1, quantity * 2 // 5)
            return [(low, price * 0.90), (quantity - low, price * 1.13)]
        low = max(1, quantity // 3)
        middle = max(1, (quantity - low) // 2)
        high = quantity - low - middle
        result = [(low, price * 0.86), (middle, price)]
        if high:
            result.append((high, price * 1.18))
        return result
