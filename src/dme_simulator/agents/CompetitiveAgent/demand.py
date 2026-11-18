"""Small, dependency-free demand model using only the agent's own sales."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import ceil, exp, log


@dataclass(slots=True)
class DemandObservation:
    price: float
    offered: int
    sold: int
    censored: bool


class DemandEstimator:
    """Recency-weighted kernel estimate with explicit stockout censoring.

    A sell-out says demand was *at least* the offered quantity.  We therefore
    use a conservative lower-bound uplift instead of treating it as an exact
    count.  The model intentionally avoids assumptions about customer segments,
    which are not part of ``Observation``.
    """

    def __init__(self, window: int = 24) -> None:
        self._history: deque[DemandObservation] = deque(maxlen=window)

    @property
    def observations(self) -> int:
        return len(self._history)

    def update(
        self, price: float, offered: int, sold: int, *, censored: bool
    ) -> None:
        if offered <= 0 or price <= 0:
            return
        sold = max(0, min(int(sold), int(offered)))
        self._history.append(
            DemandObservation(float(price), int(offered), sold, bool(censored))
        )

    def expected_demand(self, price: float, prior: float) -> float:
        """Estimate demand at ``price`` using nearby historical prices."""
        prior = max(0.0, float(prior))
        if not self._history or price <= 0:
            return prior

        numerator = 0.75 * prior
        denominator = 0.75
        total = len(self._history)
        for index, observation in enumerate(self._history):
            # Similar prices carry more information. Recent observations retain
            # more weight so the estimate follows non-stationary competition.
            distance = abs(log(max(price, 1e-9) / observation.price))
            proximity = exp(-distance / 0.48)
            recency = 0.88 ** (total - index - 1)
            weight = proximity * recency
            if observation.censored:
                inferred = max(
                    observation.sold + 0.25,
                    observation.sold * 1.15,
                )
            else:
                inferred = float(observation.sold)
            numerator += weight * inferred
            denominator += weight
        estimate = max(0.0, numerator / denominator)

        # Kernel smoothing alone reverts to the prior far outside sampled
        # prices, which can make an unseen extreme price look attractive.
        # Impose only economically safe shape information: demand does not
        # increase when price rises. A successful observation becomes a soft
        # anchor; an uncensored zero sale is a strong upper-price boundary.
        successful = [item for item in self._history if item.sold > 0]
        if successful:
            anchor = max(successful, key=lambda item: item.price)
            anchor_demand = max(
                float(anchor.sold),
                anchor.sold * 1.20 if anchor.censored else float(anchor.sold),
            )
            if price > anchor.price:
                estimate = min(
                    estimate,
                    anchor_demand * (price / anchor.price) ** -1.65,
                )
        zeros = [item for item in self._history if item.sold == 0 and not item.censored]
        if zeros:
            # A miss is noisy (customers may simply not have arrived), but
            # repeated misses at a lower price must make still higher prices
            # progressively less plausible.  This is a soft power-law cap,
            # never the old hard 0.08 boundary.
            reference = zeros[-1].price
            if price > reference * 1.001:
                # Repeated misses are already reflected by the local kernel.
                # Keep the extrapolation cap deliberately weak so a sparse
                # market does not turn one unlucky turn into zero demand.
                soft_level = prior * 0.90
                estimate = min(
                    estimate,
                    soft_level * (price / reference) ** -1.05,
                )
        return estimate

    def demand_curve(self, prices: list[float], prior: float) -> list[float]:
        """Return a demand curve that is non-increasing in price.

        Kernel estimates are useful locally but can cross when observations are
        sparse.  Pricing always evaluates a finite candidate grid, so a simple
        monotone projection is both cheaper and safer than extrapolating a
        global parametric law.  Zero-sales samples remain noisy observations:
        they reduce the local estimate but never create a hard boundary.
        """
        if not prices:
            return []
        order = sorted(range(len(prices)), key=prices.__getitem__)
        projected = [0.0] * len(prices)
        ceiling = float("inf")
        for index in order:
            ceiling = min(ceiling, self.expected_demand(prices[index], prior))
            projected[index] = max(0.0, ceiling)
        return projected

    def sale_probability(self, unit: int, price: float, prior: float) -> float:
        """P(D >= unit), using a bounded Poisson approximation, not a floor."""
        mean = min(100.0, self.expected_demand(price, prior))
        if unit <= 0:
            return 1.0
        mass = exp(-mean)
        cumulative = mass
        for k in range(1, unit):
            mass *= mean / k
            cumulative += mass
        return max(0.0, min(1.0, 1.0 - cumulative))

    @staticmethod
    def expected_sales_from_mean(stock: int, mean: float) -> float:
        """E[min(D, stock)] for a Poisson approximation with bounded work."""
        stock = max(0, int(stock))
        mean = max(0.0, min(100.0, float(mean)))
        if stock == 0:
            return 0.0
        mass = exp(-mean)
        cumulative = mass
        expected = max(0.0, 1.0 - cumulative)
        for unit in range(2, stock + 1):
            mass *= mean / (unit - 1)
            cumulative += mass
            expected += max(0.0, 1.0 - cumulative)
        return min(float(stock), expected)

    @property
    def last(self):
        return self._history[-1] if self._history else None

    @property
    def highest_successful_price(self) -> float | None:
        prices = [item.price for item in self._history if item.sold > 0]
        return max(prices) if prices else None

    @property
    def zero_streak(self) -> int:
        count = 0
        for item in reversed(self._history):
            if item.sold:
                break
            count += 1
        return count

    @property
    def recent_sales(self) -> float:
        observations = list(self._history)[-6:]
        if not observations:
            return 0.0
        weights = list(range(1, len(observations) + 1))
        return sum(item.sold * weight for item, weight in zip(observations, weights)) / sum(weights)

    @property
    def recent_sell_through(self) -> float:
        observations = list(self._history)[-6:]
        offered = sum(item.offered for item in observations)
        return sum(item.sold for item in observations) / offered if offered else 0.0

    def should_reprobe(self, turn: int, phase: int = 0) -> bool:
        """Occasionally reopen a product after a run of unsuccessful offers."""
        if self.zero_streak < 3:
            return False
        interval = min(16, 5 + self.zero_streak)
        return (int(turn) + int(phase)) % interval == 0

    def expected_revenue(self, price: float, stock: int, prior: float) -> float:
        if stock <= 0:
            return 0.0
        return float(price) * self.expected_sales_from_mean(
            stock, self.expected_demand(price, prior)
        )

    def target_stock(self, price: float, prior: float, maximum: int) -> int:
        estimate = self.expected_demand(price, prior)
        if not self._history:
            return min(1, maximum)
        if self.zero_streak >= 3:
            return 0
        if estimate < 0.08 and self.recent_sales == 0:
            return 0
        target = max(estimate * 1.35, self.recent_sales * 1.25)
        if self.last is not None and self.last.censored:
            target = max(target, self.last.sold + 1.0)
        return max(0, min(int(maximum), ceil(target)))


class DemandBook:
    def __init__(self) -> None:
        self._models: dict[str, DemandEstimator] = {}

    def model(self, good: str) -> DemandEstimator:
        model = self._models.get(good)
        if model is None:
            model = self._models[good] = DemandEstimator()
        return model
