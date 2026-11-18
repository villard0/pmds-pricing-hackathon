"""Reproducible paired benchmark for CompetitiveAgent.

The focal strategy changes while scenario, seed, team IDs, and opponents stay
fixed.  Generated CSV/JSON files live outside the upload bundle.

Examples:
  uv run python benchmarks/competitive_benchmark.py --quick
  uv run python benchmarks/competitive_benchmark.py --counts 4 8 16 32 \
      --seeds 10 --stress --output reports/competitive/holdout
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from dme_simulator.agents import BaseAgent
from dme_simulator.agents.loader import load_agent_bundle_class
from dme_simulator.config import load_config
from dme_simulator.runner import _validate_decision, run_simulation


ROOT = Path(__file__).resolve().parents[1]
AGENTS = ROOT / "src" / "dme_simulator" / "agents"

PROFILES = {
    "competitive": {},
    "adaptive": {"pricing_mode": "adaptive", "ladder_levels": 0},
    "hybrid": {"pricing_mode": "hybrid", "ladder_levels": 1},
    "margin": {"pricing_mode": "margin"},
    "swucb": {"pricing_mode": "ucb"},
    "gpucb": {"pricing_mode": "gp"},
    "demand": {"pricing_mode": "demand"},
    "ladder2": {"pricing_mode": "hybrid", "ladder_levels": 2},
    "ladder3": {"pricing_mode": "hybrid", "ladder_levels": 3},
    "no_joint_plan": {"use_joint_planning": False},
    "no_demand": {"use_demand_learning": False},
    "no_bid_tiers": {"tiered_bids": False},
    "fixed_budget": {"adaptive_budget": False},
    "conservative_bid": {"bid_aggressiveness": 0.82},
    "aggressive_bid": {"bid_aggressiveness": 1.24},
    "aggressive_price": {"price_factor": 0.80},
}


class MeasuredAgent(BaseAgent):
    """Testing-only wrapper that counts invalid focal decisions."""

    def __init__(self, delegate: BaseAgent) -> None:
        self.delegate = delegate
        self.invalid = 0
        self.calls = 0
        self.durations: list[float] = []

    def decide(self, state, goods_config):
        started = time.perf_counter()
        self.calls += 1
        try:
            decision = self.delegate.decide(state, goods_config)
            _validate_decision(decision, state, goods_config)
            return decision
        except BaseException:
            self.invalid += 1
            raise
        finally:
            self.durations.append(time.perf_counter() - started)


@dataclass(slots=True)
class Result:
    profile: str
    scenario: str
    seed: int
    teams: int
    final_capital: float
    rank: int
    win: int
    units_sold: int
    unsold_inventory: int
    auction_spend: float
    sales_revenue: float
    decision_mean_ms: float
    decision_max_ms: float
    invalid_or_timed_out: int


def _classes():
    return {
        name: load_agent_bundle_class(AGENTS / "builtin_bundles" / name)
        for name in ("DummyAgent", "PatientAgent", "PriceWarAgent")
    } | {"CompetitiveAgent": load_agent_bundle_class(AGENTS / "CompetitiveAgent")}


def _opponents(classes, count: int) -> dict[str, BaseAgent]:
    opponents: dict[str, BaseAgent] = {
        "Dummy": classes["DummyAgent"](),
        "Patient": classes["PatientAgent"](),
        "PriceWar": classes["PriceWarAgent"](),
    }
    recipes = (
        {"bid_aggressiveness": 0.84},
        {"bid_aggressiveness": 1.20},
        {"pricing_mode": "margin"},
        {"pricing_mode": "ucb"},
        {"price_factor": 0.82},
        {},
    )
    index = 0
    while len(opponents) < count - 1:
        kwargs = recipes[index % len(recipes)]
        opponents[f"AdaptiveRival_{index + 1:02d}"] = classes["CompetitiveAgent"](**kwargs)
        index += 1
    return opponents


def _scenario(path: Path, stress: str | None):
    config = load_config(path)
    label = path.stem
    if stress == "scarce":
        suppliers = {
            good: replace(
                supplier,
                production_min=max(0, int(supplier.production_min * 0.55)),
                production_max=max(1, int(supplier.production_max * 0.65)),
            )
            for good, supplier in config.suppliers.items()
        }
        config = replace(config, suppliers=suppliers)
        label += "_stress_scarce"
    elif stress == "crowded":
        market = replace(
            config.consumer_market,
            base_customers=max(1, int(config.consumer_market.base_customers * 0.65)),
            amplitude=int(config.consumer_market.amplitude * 0.5),
        )
        config = replace(config, consumer_market=market)
        label += "_stress_crowded"
    config = replace(
        config,
        output=replace(
            config.output,
            render_plots=False,
            dashboard="none",
            progress_interval_turns=0,
        ),
    )
    return label, config


def run_case(classes, profile, kwargs, label, config, seed, teams) -> Result:
    focal = MeasuredAgent(classes["CompetitiveAgent"](**kwargs))
    agents = {"Focal": focal, **_opponents(classes, teams)}
    simulator = run_simulation(
        config,
        agents,
        seed_override=seed,
        profile=True,
        quiet=True,
    )
    final = {name: state.capital for name, state in simulator.teams_state.items()}
    capital = final["Focal"]
    rank = 1 + sum(value > capital + 1e-9 for value in final.values())
    history = simulator.history
    timings = history.diagnostics.decision_duration_seconds["Focal"]
    inventory = simulator.teams_state["Focal"].inventory
    return Result(
        profile=profile,
        scenario=label,
        seed=seed,
        teams=teams,
        final_capital=capital,
        rank=rank,
        win=int(rank == 1),
        units_sold=int(sum(history.units_sold["Focal"])),
        unsold_inventory=int(
            sum(inventory.get(good, 0) for good in config.goods.finished_goods)
        ),
        auction_spend=sum(history.auction_spend["Focal"]),
        sales_revenue=sum(history.revenue["Focal"]),
        decision_mean_ms=statistics.fmean(timings) * 1000 if timings else 0.0,
        decision_max_ms=max(timings, default=0.0) * 1000,
        invalid_or_timed_out=focal.invalid,
    )


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def summarize(results: list[Result]) -> list[dict[str, float | str | int]]:
    rows = []
    keys = sorted({(row.profile, row.scenario, row.teams) for row in results})
    for profile, scenario, teams in keys:
        group = [
            row
            for row in results
            if (row.profile, row.scenario, row.teams) == (profile, scenario, teams)
        ]
        rows.append(
            {
                "profile": profile,
                "scenario": scenario,
                "teams": teams,
                "games": len(group),
                "mean_rank": statistics.fmean(row.rank for row in group),
                "win_rate": statistics.fmean(row.win for row in group),
                "mean_capital": statistics.fmean(row.final_capital for row in group),
                "capital_p10": percentile([row.final_capital for row in group], 0.10),
                "mean_sold": statistics.fmean(row.units_sold for row in group),
                "mean_unsold": statistics.fmean(row.unsold_inventory for row in group),
                "mean_spend": statistics.fmean(row.auction_spend for row in group),
                "mean_revenue": statistics.fmean(row.sales_revenue for row in group),
                "decision_mean_ms": statistics.fmean(row.decision_mean_ms for row in group),
                "decision_max_ms": max(row.decision_max_ms for row in group),
                "invalid_or_timed_out": sum(row.invalid_or_timed_out for row in group),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--counts", nargs="+", type=int, default=[4, 8, 16, 32])
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--seed-start", type=int, default=1000)
    parser.add_argument("--stress", action="store_true")
    parser.add_argument("--profiles", nargs="+", choices=sorted(PROFILES))
    parser.add_argument(
        "--output", type=Path, default=ROOT / "reports" / "competitive" / "benchmark"
    )
    args = parser.parse_args()
    profiles = args.profiles or list(PROFILES)
    counts = [4, 8] if args.quick else args.counts
    seed_count = 2 if args.quick else args.seeds
    config_paths = [ROOT / "config" / "default.yaml", ROOT / "config" / "scaled_market.yaml"]
    if not args.quick:
        config_paths += [
            ROOT / "config" / "tutorial_pricing.yaml",
            ROOT / "config" / "tutorial_auction.yaml",
        ]
    stress_modes = [None, "scarce", "crowded"] if args.stress else [None]
    classes = _classes()
    results: list[Result] = []
    for config_path in config_paths:
        for stress in stress_modes:
            label, config = _scenario(config_path, stress)
            for teams in counts:
                for seed in range(args.seed_start, args.seed_start + seed_count):
                    for profile in profiles:
                        result = run_case(
                            classes, profile, PROFILES[profile], label, config, seed, teams
                        )
                        results.append(result)
                        print(
                            f"{label:<24} teams={teams:>2} seed={seed} "
                            f"{profile:<18} rank={result.rank:>2} "
                            f"capital={result.final_capital:>10.2f}"
                        )
    raw_rows = [asdict(row) for row in results]
    summary_rows = summarize(results)
    _write_csv(args.output.with_name(args.output.name + "-runs.csv"), raw_rows)
    _write_csv(args.output.with_name(args.output.name + "-summary.csv"), summary_rows)
    manifest = {
        "seed_start": args.seed_start,
        "seed_count": seed_count,
        "counts": counts,
        "profiles": profiles,
        "stress": args.stress,
        "configs": [str(path.relative_to(ROOT)) for path in config_paths],
        "runs": len(results),
    }
    manifest_path = args.output.with_name(args.output.name + "-manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
