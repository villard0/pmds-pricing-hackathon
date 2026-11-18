"""Small, dependency-free demand model using only the agent's own sales."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import exp, log


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
        prior = max(0.25, float(prior))
        if not self._history or price <= 0:
            return prior

        numerator = 1.75 * prior
        denominator = 1.75
        total = len(self._history)
        for index, observation in enumerate(self._history):
            # Similar prices carry more information. Recent observations retain
            # more weight so the estimate follows non-stationary competition.
            distance = abs(log(max(price, 1e-9) / observation.price))
            proximity = exp(-distance / 0.48)
            recency = 0.55 + 0.45 * (index + 1) / total
            weight = proximity * recency
            if observation.censored:
                inferred = max(
                    observation.sold + 0.75,
                    observation.sold * 1.20,
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
            elif price < anchor.price:
                estimate = max(estimate, min(anchor_demand, prior * 2.5))
        zero_prices = [
            item.price
            for item in self._history
            if item.sold == 0 and not item.censored
        ]
        if zero_prices:
            zero_boundary = min(zero_prices)
            if price >= zero_boundary:
                estimate = min(estimate, 0.08)
            elif not successful:
                estimate *= max(0.15, 1.0 - 0.8 * price / zero_boundary)
        return estimate

    def expected_revenue(self, price: float, stock: int, prior: float) -> float:
        if stock <= 0:
            return 0.0
        return float(price) * min(float(stock), self.expected_demand(price, prior))

    def target_stock(self, price: float, prior: float, maximum: int) -> int:
        estimate = self.expected_demand(price, prior)
        # A small buffer is useful when the last sample sold out, but bounded
        # targets prevent an optimistic cold-start model from accumulating stock.
        buffer = 1.25 if self._history and self._history[-1].censored else 0.5
        return max(1, min(int(maximum), int(estimate + buffer + 0.999)))


class DemandBook:
    def __init__(self) -> None:
        self._models: dict[str, DemandEstimator] = {}

    def model(self, good: str) -> DemandEstimator:
        model = self._models.get(good)
        if model is None:
            model = self._models[good] = DemandEstimator()
        return model
