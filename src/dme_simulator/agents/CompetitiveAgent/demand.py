"""Small, dependency-free demand model using only the agent's own sales."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import exp, log, ceil


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
        return estimate

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

    @property
    def last(self):
        return self._history[-1] if self._history else None

    @property
    def zero_streak(self) -> int:
        count = 0
        for item in reversed(self._history):
            if item.sold:
                break
            count += 1
        return count

    def expected_revenue(self, price: float, stock: int, prior: float) -> float:
        if stock <= 0:
            return 0.0
        return float(price) * sum(self.sale_probability(k, price, prior) for k in range(1, stock + 1))

    def target_stock(self, price: float, prior: float, maximum: int) -> int:
        estimate = self.expected_demand(price, prior)
        # A small buffer is useful when the last sample sold out, but bounded
        # targets prevent an optimistic cold-start model from accumulating stock.
        if not self._history:
            return min(1, maximum)
        if self.zero_streak >= 3 or estimate < 0.15:
            return 0
        return max(0, min(int(maximum), ceil(estimate * 1.4)))


class DemandBook:
    def __init__(self) -> None:
        self._models: dict[str, DemandEstimator] = {}

    def model(self, good: str) -> DemandEstimator:
        model = self._models.get(good)
        if model is None:
            model = self._models[good] = DemandEstimator()
        return model
