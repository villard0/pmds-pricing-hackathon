"""First-price auction adaptation with coordinated, budget-safe bid tiers."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil


@dataclass(slots=True)
class MaterialState:
    multiplier: float = 1.35
    loss_streak: int = 0
    paid_ema: float = 0.0


class AuctionModel:
    def __init__(self, aggressiveness: float = 1.0) -> None:
        self.aggressiveness = max(0.65, min(1.55, float(aggressiveness)))
        self._states: dict[str, MaterialState] = {}
        self.last_requested: dict[str, int] = {}

    def _state(self, good: str) -> MaterialState:
        state = self._states.get(good)
        if state is None:
            state = self._states[good] = MaterialState()
        return state

    def observe(self, good: str, won: int, paid: float) -> None:
        requested = self.last_requested.get(good, 0)
        if requested <= 0:
            return
        state = self._state(good)
        fill = won / requested
        if fill < 0.30:
            state.loss_streak = min(5, state.loss_streak + 1)
            state.multiplier = min(2.35, state.multiplier + 0.10)
        elif fill > 0.80:
            state.loss_streak = 0
            state.multiplier = max(1.08, state.multiplier - 0.045)
        else:
            state.loss_streak = max(0, state.loss_streak - 1)
        if won > 0 and paid > 0:
            state.paid_ema = paid if state.paid_ema <= 0 else 0.72 * state.paid_ema + 0.28 * paid

    def expected_price(self, good: str, reserve: float, scarcity: float) -> float:
        state = self._state(good)
        scarcity_bonus = 0.20 * max(0.0, min(2.0, scarcity))
        multiplier = (state.multiplier + scarcity_bonus) * self.aggressiveness
        price = reserve * max(1.01, multiplier)
        if state.paid_ema > 0:
            price = max(price, state.paid_ema * (1.0 + 0.025 * state.loss_streak))
        return price

    def build_tiers(
        self,
        requirements: dict[str, int],
        inventory: dict[str, int],
        lots: list,
        value_caps: dict[str, float],
        budget: float,
        team_count: int,
        tiered: bool,
    ) -> list[tuple[str, int, float]]:
        """Return tuples consumed by ``agent.py`` to construct public ``Bid``s.

        The sum of worst-case payments never exceeds ``budget``. This matters
        because the simulator clears materials in random order.
        """
        lot_by_good = {lot.good: lot for lot in lots}
        planned: list[tuple[str, int, float, float]] = []
        self.last_requested = {}
        for good, required in requirements.items():
            lot = lot_by_good.get(good)
            shortage = max(0, int(required) - int(inventory.get(good, 0)))
            if lot is None or shortage <= 0 or lot.quantity <= 0:
                continue
            fair_share = max(1.0, lot.quantity / max(1, team_count))
            scarcity = shortage / fair_share
            base = self.expected_price(good, lot.reserve_price, scarcity)
            cap = max(lot.reserve_price, float(value_caps.get(good, base)))
            base = min(base, cap)
            if base + 1e-9 < lot.reserve_price:
                continue
            shortage = min(shortage, lot.quantity)
            priority = max(1, min(shortage, ceil(shortage * 0.38)))
            if tiered and shortage >= 3:
                high = min(cap, base * 1.075)
                low = max(lot.reserve_price, min(cap, base * 0.93))
                planned.append((good, priority, high, 2.0))
                planned.append((good, shortage - priority, low, 1.0))
            else:
                planned.append((good, shortage, base, 1.5))

        # Higher priority tiers survive budget pressure first, but scale each
        # tier proportionally enough to retain complementary components.
        total = sum(quantity * price for _, quantity, price, _ in planned)
        scale = min(1.0, max(0.0, budget) / total) if total > 0 else 0.0
        bids: list[tuple[str, int, float]] = []
        remaining_budget = max(0.0, budget)
        # First pass: scaled complementary quantities in stable good order.
        for good, quantity, price, _priority in sorted(planned, key=lambda item: item[0]):
            scaled = min(quantity, int(quantity * scale))
            if scale > 0 and scaled == 0 and price <= remaining_budget:
                scaled = 1
            scaled = min(scaled, int((remaining_budget + 1e-9) // price))
            if scaled <= 0:
                continue
            bids.append((good, scaled, price))
            remaining_budget -= scaled * price
            self.last_requested[good] = self.last_requested.get(good, 0) + scaled
        return bids
