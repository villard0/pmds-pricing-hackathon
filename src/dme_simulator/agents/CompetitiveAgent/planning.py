"""Bounded joint production planning over complementary raw materials."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod


@dataclass(slots=True)
class Plan:
    quantities: dict[str, int]
    requirements: dict[str, int]
    score: float
    new_cost: float


def _requirements(quantities: dict[str, int], recipes: dict) -> dict[str, int]:
    required: dict[str, int] = {}
    for recipe_name, quantity in quantities.items():
        if quantity <= 0:
            continue
        for material, units in recipes[recipe_name].inputs.items():
            required[material] = required.get(material, 0) + units * quantity
    return required


def _new_cost(
    requirements: dict[str, int], inventory: dict[str, int], costs: dict[str, float]
) -> float:
    return sum(
        max(0, quantity - int(inventory.get(material, 0)))
        * float(costs.get(material, 0.0))
        for material, quantity in requirements.items()
    )


def joint_plan(
    recipes: dict,
    inventory: dict[str, int],
    raw_costs: dict[str, float],
    finished_inventory: dict[str, int],
    target_by_good: dict[str, int],
    expected_unit_revenue: dict[str, float],
    budget: float,
    total_limit: int,
    beam_width: int = 96,
    available: dict[str, int] | None = None,
    marginal_values: dict[str, list[float]] | None = None,
    fill_probabilities: dict[str, float] | None = None,
    max_active_goods: int | None = None,
) -> Plan:
    """Choose a whole recipe portfolio, not independent material purchases.

    Owned materials have zero *new* cash cost. Missing complementary inputs are
    costed together, which prevents isolated purchases that cannot be assembled.
    """
    recipe_items = sorted(
        recipes.items(),
        key=lambda item: (
            -expected_unit_revenue.get(item[1].output_good, 0.0),
            item[0],
        ),
    )
    # (score, quantities, requirements, new_cost, produced_count)
    states: list[tuple[float, dict[str, int], dict[str, int], float, int]] = [
        (0.0, {}, {}, 0.0, 0)
    ]
    for recipe_name, recipe in recipe_items:
        good = recipe.output_good
        maximum = max(
            0,
            min(
                total_limit,
                int(target_by_good.get(good, 0))
                - int(finished_inventory.get(good, 0)),
            ),
        )
        next_states: list[
            tuple[float, dict[str, int], dict[str, int], float, int]
        ] = []
        for _score, quantities, _required, _cost, produced in states:
            room = max(0, total_limit - produced)
            for quantity in range(min(maximum, room) + 1):
                candidate = dict(quantities)
                candidate[recipe_name] = quantity
                required = _requirements(candidate, recipes)
                if available is not None and any(
                    amount > inventory.get(material, 0) + available.get(material, 0)
                    for material, amount in required.items()
                ):
                    continue
                new_cost = _new_cost(required, inventory, raw_costs)
                if new_cost > budget + 1e-9:
                    continue
                revenue = 0.0
                output_units = {}
                for name, units in candidate.items():
                    output = recipes[name].output_good
                    output_units[output] = output_units.get(output, 0) + units
                if any(units > max(0, target_by_good.get(good, 0) - finished_inventory.get(good, 0))
                       for good, units in output_units.items()):
                    continue
                if max_active_goods is not None and sum(units > 0 for units in output_units.values()) > max_active_goods:
                    continue
                for output, units in output_units.items():
                    revenue += (sum(marginal_values.get(output, [])[:units]) if marginal_values is not None
                                else units * expected_unit_revenue.get(output, 0.0))
                if fill_probabilities is None:
                    score = revenue - new_cost
                else:
                    missing = {
                        material: max(0, amount - inventory.get(material, 0))
                        for material, amount in required.items()
                    }
                    relevant = [
                        fill_probabilities.get(material, 0.5)
                        for material, amount in missing.items()
                        if amount > 0
                    ]
                    # A recipe is usable only when every missing complement is
                    # acquired.  Treat the learned fill rates as a lightweight
                    # independent approximation of that joint event.
                    completion = prod(relevant) if relevant else 1.0
                    expected_cost = sum(
                        amount * raw_costs.get(material, 0.0)
                        * fill_probabilities.get(material, 0.5)
                        for material, amount in missing.items()
                    )
                    stranded = max(0.0, expected_cost - completion * new_cost)
                    score = completion * revenue - expected_cost - 0.35 * stranded
                next_states.append(
                    (score, candidate, required, new_cost, produced + quantity)
                )
        # Deterministic pruning: prefer profit, then less new cash and fewer units.
        next_states.sort(
            key=lambda state: (state[0], -state[3], -state[4]), reverse=True
        )
        states = next_states[:beam_width] or states
    best = states[0]
    return Plan(best[1], best[2], best[0], best[3])


def greedy_plan(
    recipes: dict,
    inventory: dict[str, int],
    raw_costs: dict[str, float],
    finished_inventory: dict[str, int],
    target_by_good: dict[str, int],
    expected_unit_revenue: dict[str, float],
    budget: float,
    total_limit: int,
) -> Plan:
    quantities: dict[str, int] = {}
    ranked = sorted(
        recipes.items(),
        key=lambda item: expected_unit_revenue.get(item[1].output_good, 0.0)
        / max(
            1e-9,
            sum(raw_costs.get(material, 0.0) * units for material, units in item[1].inputs.items()),
        ),
        reverse=True,
    )
    for name, recipe in ranked:
        maximum = max(
            0,
            min(
                total_limit - sum(quantities.values()),
                target_by_good.get(recipe.output_good, 0)
                - finished_inventory.get(recipe.output_good, 0),
            ),
        )
        for _ in range(maximum):
            candidate = dict(quantities)
            candidate[name] = candidate.get(name, 0) + 1
            required = _requirements(candidate, recipes)
            cost = _new_cost(required, inventory, raw_costs)
            revenue = sum(
                units * expected_unit_revenue.get(recipes[key].output_good, 0.0)
                for key, units in candidate.items()
            )
            if cost <= budget + 1e-9 and revenue > cost:
                quantities = candidate
    required = _requirements(quantities, recipes)
    cost = _new_cost(required, inventory, raw_costs)
    revenue = sum(
        units * expected_unit_revenue.get(recipes[key].output_good, 0.0)
        for key, units in quantities.items()
    )
    return Plan(quantities, required, revenue - cost, cost)
