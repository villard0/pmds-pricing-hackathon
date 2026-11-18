"""Reproducible, paired simulation matrix for the two preserved growth agents."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import statistics
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, fields, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dme_simulator.agents import BaseAgent  # noqa: E402
from dme_simulator.agents.loader import load_agent_bundle_class  # noqa: E402
from dme_simulator.config import EnvironmentConfig, load_config  # noqa: E402
from dme_simulator.runner import run_simulation  # noqa: E402


CURRENT = ROOT / "src" / "dme_simulator" / "agents" / "CompetitiveAgent"
IMPORTED = ROOT / "simulation_lab" / "agents" / "ImportedCompetitiveGrowth"
BUILTINS = ROOT / "src" / "dme_simulator" / "agents" / "builtin_bundles"
FOCAL_NAMES = ("CurrentGrowth", "ImportedCompetitiveGrowth")


@dataclass(frozen=True, slots=True)
class Scenario:
    base: str = "default.yaml"
    turns: int | None = None
    capital: float | None = None
    supply: float = 1.0
    customers: float = 1.0
    amplitude: float = 1.0
    period: float = 1.0
    price_sensitivity: float = 1.0
    willingness_to_pay: float = 1.0
    capacity: int | None = None


SCENARIOS: dict[str, Scenario] = {
    "baseline_40": Scenario(),
    "extended_240": Scenario(turns=240, capital=40_000.0),
    "scaled_growth": Scenario(base="scaled_market.yaml", turns=120),
    "scarce_supply": Scenario(turns=120, capital=40_000.0, supply=0.45, customers=0.85),
    "abundant_market": Scenario(turns=120, capital=40_000.0, supply=1.75, customers=1.50),
    "demand_recession": Scenario(
        turns=120,
        capital=40_000.0,
        customers=0.45,
        amplitude=0.35,
        willingness_to_pay=0.85,
    ),
    "demand_boom": Scenario(
        turns=120,
        capital=40_000.0,
        customers=2.0,
        amplitude=1.6,
        willingness_to_pay=1.15,
    ),
    "production_bottleneck": Scenario(turns=120, capital=40_000.0, capacity=4),
    "price_sensitive": Scenario(
        turns=120,
        capital=40_000.0,
        price_sensitivity=3.0,
        willingness_to_pay=0.92,
    ),
    "price_tolerant": Scenario(
        turns=120,
        capital=40_000.0,
        price_sensitivity=0.40,
        willingness_to_pay=1.20,
    ),
    "volatile_demand": Scenario(
        turns=120,
        capital=40_000.0,
        amplitude=2.5,
        period=0.45,
    ),
    "premium_market": Scenario(
        turns=120,
        capital=40_000.0,
        customers=0.75,
        willingness_to_pay=1.65,
    ),
    "low_capital": Scenario(turns=80, capital=5_000.0, supply=0.75),
}

ROSTERS = {
    "duel": "2 teams: the two focal agents",
    "standard": "5 teams: focal agents plus the three official baselines",
    "pressure": "12 teams: aggressive PriceWar-heavy competition",
    "crowded_12": "12 teams: balanced official-opponent mix",
    "selfplay_16": "16 teams: seven additional copies of each focal strategy",
    "crowded_43": "43 teams: leaderboard-scale official-opponent mix",
}

SUITES: dict[str, dict[str, Any]] = {
    "smoke": {
        "scenarios": ["baseline_40"],
        "rosters": ["duel"],
        "seeds": 1,
        "turns": 5,
    },
    "core": {
        "scenarios": [
            "baseline_40",
            "extended_240",
            "scarce_supply",
            "abundant_market",
            "demand_recession",
            "demand_boom",
            "production_bottleneck",
            "price_sensitive",
        ],
        "rosters": ["standard", "pressure"],
        "seeds": 3,
        "turns": None,
    },
    "extended": {
        "scenarios": [
            "extended_240",
            "scaled_growth",
            "scarce_supply",
            "demand_recession",
            "volatile_demand",
            "premium_market",
            "low_capital",
        ],
        "rosters": ["standard", "selfplay_16", "crowded_43"],
        "seeds": 3,
        "turns": None,
    },
    "all": {
        "scenarios": list(SCENARIOS),
        "rosters": list(ROSTERS),
        "seeds": 5,
        "turns": None,
    },
}


class MeasuredAgent(BaseAgent):
    """Count focal calls and failures while the official runner handles fallback."""

    def __init__(self, delegate: BaseAgent) -> None:
        self.delegate = delegate
        self.calls = 0
        self.failures = 0

    def decide(self, state, goods_config):
        self.calls += 1
        try:
            return self.delegate.decide(state, goods_config)
        except BaseException:
            self.failures += 1
            raise


@dataclass(slots=True)
class RunResult:
    run_id: str
    strategy: str
    scenario: str
    roster: str
    seed: int
    turns: int
    teams: int
    initial_capital: float
    final_capital: float
    profit: float
    return_pct: float
    rank: int
    best_capital: float
    gap_to_best: float
    max_drawdown_pct: float
    units_produced: int
    units_sold: int
    raw_inventory: int
    finished_inventory: int
    auction_spend: float
    sales_revenue: float
    operating_roi: float
    decision_mean_ms: float
    decision_max_ms: float
    deadline_exceeded: int
    agent_failures: int
    calls: int


def _scaled_count(value: int, factor: float, minimum: int = 0) -> int:
    return max(minimum, int(round(value * factor)))


def build_scenario(name: str, turns_override: int | None = None) -> EnvironmentConfig:
    profile = SCENARIOS[name]
    config = load_config(ROOT / "config" / profile.base)
    suppliers = {}
    for good, supplier in config.suppliers.items():
        minimum = _scaled_count(supplier.production_min, profile.supply)
        maximum = max(minimum, _scaled_count(supplier.production_max, profile.supply, 1))
        capacity = supplier.stock_capacity
        if capacity is not None:
            capacity = max(maximum, _scaled_count(capacity, profile.supply, 1))
        suppliers[good] = replace(
            supplier,
            production_min=minimum,
            production_max=maximum,
            stock_capacity=capacity,
        )

    segments = {}
    for segment_name, segment in config.consumer_market.segments.items():
        willingness = {
            good: (low * profile.willingness_to_pay, high * profile.willingness_to_pay)
            for good, (low, high) in segment.willingness_to_pay.items()
        }
        segments[segment_name] = replace(segment, willingness_to_pay=willingness)
    market = replace(
        config.consumer_market,
        base_customers=_scaled_count(config.consumer_market.base_customers, profile.customers),
        amplitude=_scaled_count(config.consumer_market.amplitude, profile.amplitude),
        period_divisor=max(0.1, config.consumer_market.period_divisor * profile.period),
        price_sensitivity_beta=(
            config.consumer_market.price_sensitivity_beta * profile.price_sensitivity
        ),
        segments=segments,
    )
    initial_state = replace(
        config.initial_state,
        starting_capital=(
            config.initial_state.starting_capital
            if profile.capital is None
            else profile.capital
        ),
    )
    production = replace(config.production, capacity_per_turn=profile.capacity)
    output = replace(
        config.output,
        render_plots=False,
        dashboard="none",
        progress_interval_turns=0,
    )
    configured_turns = turns_override or profile.turns or config.turns
    configured = replace(
        config,
        turns=configured_turns,
        initial_state=initial_state,
        suppliers=suppliers,
        production=production,
        consumer_market=market,
        output=output,
    )
    configured.validate()
    return configured


def _load_classes() -> dict[str, type[BaseAgent]]:
    return {
        "current": load_agent_bundle_class(CURRENT),
        "imported": load_agent_bundle_class(IMPORTED),
        "dummy": load_agent_bundle_class(BUILTINS / "DummyAgent"),
        "patient": load_agent_bundle_class(BUILTINS / "PatientAgent"),
        "pricewar": load_agent_bundle_class(BUILTINS / "PriceWarAgent"),
    }


def _append_copies(
    agents: dict[str, BaseAgent],
    classes: dict[str, type[BaseAgent]],
    key: str,
    prefix: str,
    count: int,
) -> None:
    for index in range(1, count + 1):
        agents[f"{prefix}_{index:02d}"] = classes[key]()


def build_roster(name: str, seed: int) -> tuple[dict[str, BaseAgent], dict[str, MeasuredAgent]]:
    classes = _load_classes()
    measured = {
        "CurrentGrowth": MeasuredAgent(classes["current"]()),
        "ImportedCompetitiveGrowth": MeasuredAgent(classes["imported"]()),
    }
    agents: dict[str, BaseAgent] = dict(measured)
    if name == "standard":
        _append_copies(agents, classes, "dummy", "Dummy", 1)
        _append_copies(agents, classes, "patient", "Patient", 1)
        _append_copies(agents, classes, "pricewar", "PriceWar", 1)
    elif name == "pressure":
        _append_copies(agents, classes, "pricewar", "PriceWar", 7)
        _append_copies(agents, classes, "patient", "Patient", 2)
        _append_copies(agents, classes, "dummy", "Dummy", 1)
    elif name == "crowded_12":
        _append_copies(agents, classes, "pricewar", "PriceWar", 4)
        _append_copies(agents, classes, "patient", "Patient", 3)
        _append_copies(agents, classes, "dummy", "Dummy", 3)
    elif name == "selfplay_16":
        _append_copies(agents, classes, "current", "CurrentCopy", 7)
        _append_copies(agents, classes, "imported", "ImportedCopy", 7)
    elif name == "crowded_43":
        _append_copies(agents, classes, "pricewar", "PriceWar", 14)
        _append_copies(agents, classes, "patient", "Patient", 14)
        _append_copies(agents, classes, "dummy", "Dummy", 13)
    elif name != "duel":
        raise ValueError(f"unknown roster: {name}")

    order = list(agents)
    random.Random(seed).shuffle(order)
    return {team: agents[team] for team in order}, measured


def _drawdown(values: list[float]) -> float:
    peak = values[0]
    worst = 0.0
    for value in values:
        peak = max(peak, value)
        if peak > 0:
            worst = max(worst, (peak - value) / peak)
    return worst * 100.0


def run_job(job: dict[str, Any]) -> list[dict[str, Any]]:
    scenario_name = job["scenario"]
    roster_name = job["roster"]
    seed = job["seed"]
    config = build_scenario(scenario_name, job.get("turns_override"))
    order_seed = int.from_bytes(
        hashlib.sha256(f"{scenario_name}|{roster_name}|{seed}".encode()).digest()[:8],
        "big",
    )
    agents, measured = build_roster(roster_name, order_seed)
    simulator = run_simulation(
        config,
        agents,
        seed_override=seed,
        profile=True,
        quiet=True,
    )
    history = simulator.history
    final_capitals = {
        team: state.capital for team, state in simulator.teams_state.items()
    }
    best_capital = max(final_capitals.values())
    rows: list[dict[str, Any]] = []
    for strategy in FOCAL_NAMES:
        state = simulator.teams_state[strategy]
        final_capital = state.capital
        initial_capital = config.initial_state.starting_capital
        spend = float(sum(history.auction_spend[strategy]))
        revenue = float(sum(history.revenue[strategy]))
        timings = history.diagnostics.decision_duration_seconds[strategy]
        raw_inventory = sum(
            state.inventory.get(good, 0) for good in config.goods.raw_materials
        )
        finished_inventory = sum(
            state.inventory.get(good, 0) for good in config.goods.finished_goods
        )
        result = RunResult(
            run_id=f"{scenario_name}|{roster_name}|{seed}",
            strategy=strategy,
            scenario=scenario_name,
            roster=roster_name,
            seed=seed,
            turns=len(history.turns),
            teams=len(agents),
            initial_capital=initial_capital,
            final_capital=final_capital,
            profit=final_capital - initial_capital,
            return_pct=(final_capital / initial_capital - 1.0) * 100.0,
            rank=1 + sum(value > final_capital + 1e-9 for value in final_capitals.values()),
            best_capital=best_capital,
            gap_to_best=best_capital - final_capital,
            max_drawdown_pct=_drawdown([initial_capital, *history.capital[strategy]]),
            units_produced=int(sum(history.units_produced[strategy])),
            units_sold=int(sum(history.units_sold[strategy])),
            raw_inventory=int(raw_inventory),
            finished_inventory=int(finished_inventory),
            auction_spend=spend,
            sales_revenue=revenue,
            operating_roi=((revenue - spend) / spend if spend else 0.0),
            decision_mean_ms=(statistics.fmean(timings) * 1000.0 if timings else 0.0),
            decision_max_ms=(max(timings, default=0.0) * 1000.0),
            deadline_exceeded=sum(
                duration * 1000.0 >= config.agents.decide_timeout_ms
                for duration in timings
            ),
            agent_failures=measured[strategy].failures,
            calls=measured[strategy].calls,
        )
        rows.append(asdict(result))
    return rows


def safe_run_job(job: dict[str, Any]) -> dict[str, Any]:
    try:
        return {"job": job, "rows": run_job(job), "error": None}
    except BaseException as error:
        return {
            "job": job,
            "rows": [],
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        }


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries = []
    keys = sorted({(row["strategy"], row["scenario"], row["roster"]) for row in rows})
    for strategy, scenario, roster in keys:
        group = [
            row
            for row in rows
            if (row["strategy"], row["scenario"], row["roster"])
            == (strategy, scenario, roster)
        ]
        summaries.append(
            {
                "strategy": strategy,
                "scenario": scenario,
                "roster": roster,
                "games": len(group),
                "mean_final_capital": statistics.fmean(row["final_capital"] for row in group),
                "median_final_capital": statistics.median(row["final_capital"] for row in group),
                "capital_p10": percentile([row["final_capital"] for row in group], 0.10),
                "mean_return_pct": statistics.fmean(row["return_pct"] for row in group),
                "mean_rank": statistics.fmean(row["rank"] for row in group),
                "win_rate": statistics.fmean(row["rank"] == 1 for row in group),
                "mean_drawdown_pct": statistics.fmean(row["max_drawdown_pct"] for row in group),
                "mean_units_sold": statistics.fmean(row["units_sold"] for row in group),
                "mean_raw_inventory": statistics.fmean(row["raw_inventory"] for row in group),
                "mean_finished_inventory": statistics.fmean(row["finished_inventory"] for row in group),
                "mean_operating_roi": statistics.fmean(row["operating_roi"] for row in group),
                "decision_mean_ms": statistics.fmean(row["decision_mean_ms"] for row in group),
                "decision_max_ms": max(row["decision_max_ms"] for row in group),
                "deadline_exceeded": sum(row["deadline_exceeded"] for row in group),
                "agent_failures": sum(row["agent_failures"] for row in group),
            }
        )
    return summaries


def pairwise(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_run: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_run.setdefault(row["run_id"], {})[row["strategy"]] = row
    comparisons = []
    for run_id, strategies in sorted(by_run.items()):
        if set(strategies) != set(FOCAL_NAMES):
            continue
        current = strategies["CurrentGrowth"]
        imported = strategies["ImportedCompetitiveGrowth"]
        delta = current["final_capital"] - imported["final_capital"]
        winner = "tie" if abs(delta) < 1e-9 else ("CurrentGrowth" if delta > 0 else "ImportedCompetitiveGrowth")
        comparisons.append(
            {
                "run_id": run_id,
                "scenario": current["scenario"],
                "roster": current["roster"],
                "seed": current["seed"],
                "current_capital": current["final_capital"],
                "imported_capital": imported["final_capital"],
                "current_minus_imported": delta,
                "winner": winner,
            }
        )
    return comparisons


def fingerprint() -> str:
    digest = hashlib.sha256()
    paths = [Path(__file__), *CURRENT.glob("*.py"), *IMPORTED.glob("*.py")]
    for path in sorted(paths):
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def list_catalog() -> None:
    print("Scenarios:")
    for name, scenario in SCENARIOS.items():
        print(f"  {name:<24} base={scenario.base:<20} turns={scenario.turns or 'base'}")
    print("\nRosters:")
    for name, description in ROSTERS.items():
        print(f"  {name:<24} {description}")
    print("\nSuites:")
    for name, suite in SUITES.items():
        jobs = len(suite["scenarios"]) * len(suite["rosters"]) * suite["seeds"]
        print(f"  {name:<24} {jobs} jobs by default")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=sorted(SUITES), default="core")
    parser.add_argument("--scenarios", nargs="+", choices=sorted(SCENARIOS))
    parser.add_argument("--rosters", nargs="+", choices=sorted(ROSTERS))
    parser.add_argument("--seeds", type=int)
    parser.add_argument("--seed-start", type=int, default=30_000)
    parser.add_argument("--workers", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    parser.add_argument("--turns", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--list", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.list:
        list_catalog()
        return
    suite = SUITES[args.suite]
    scenarios = args.scenarios or suite["scenarios"]
    rosters = args.rosters or suite["rosters"]
    seed_count = args.seeds if args.seeds is not None else suite["seeds"]
    turns_override = args.turns if args.turns is not None else suite["turns"]
    if seed_count < 1 or args.workers < 1:
        raise SystemExit("--seeds and --workers must be positive")
    if turns_override is not None and turns_override < 1:
        raise SystemExit("--turns must be positive")
    jobs = [
        {
            "scenario": scenario,
            "roster": roster,
            "seed": seed,
            "turns_override": turns_override,
        }
        for scenario in scenarios
        for roster in rosters
        for seed in range(args.seed_start, args.seed_start + seed_count)
    ]
    if args.dry_run:
        print(json.dumps({"jobs": jobs, "count": len(jobs)}, indent=2))
        return

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output = (args.output or ROOT / "simulation_lab" / "results" / timestamp).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = datetime.now(UTC).isoformat()
    source_fingerprint = fingerprint()
    manifest = {
        "status": "running",
        "started_at": started,
        "suite": args.suite,
        "fingerprint": source_fingerprint,
        "jobs": jobs,
        "python": sys.version,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    partial_path = output / "partial.jsonl"
    with partial_path.open("w", encoding="utf-8") as partial:
        if args.workers == 1:
            outcomes = map(safe_run_job, jobs)
            executor = None
        else:
            executor = ProcessPoolExecutor(max_workers=args.workers)
            outcomes = executor.map(safe_run_job, jobs)
        try:
            for index, outcome in enumerate(outcomes, 1):
                partial.write(json.dumps(outcome) + "\n")
                partial.flush()
                rows.extend(outcome["rows"])
                if outcome["error"]:
                    errors.append(outcome)
                label = outcome["job"]
                status = "ERROR" if outcome["error"] else "ok"
                print(
                    f"[{index:>4}/{len(jobs)}] {status:<5} "
                    f"{label['scenario']} / {label['roster']} / seed {label['seed']}",
                    flush=True,
                )
        finally:
            if executor is not None:
                executor.shutdown()

    write_csv(output / "runs.csv", rows, [item.name for item in fields(RunResult)])
    summary_rows = summarize(rows)
    if summary_rows:
        write_csv(output / "summary.csv", summary_rows, list(summary_rows[0]))
    comparison_rows = pairwise(rows)
    if comparison_rows:
        write_csv(output / "pairwise.csv", comparison_rows, list(comparison_rows[0]))
    if errors:
        error_rows = [
            {
                **outcome["job"],
                "error": outcome["error"],
                "traceback": outcome.get("traceback", ""),
            }
            for outcome in errors
        ]
        write_csv(output / "errors.csv", error_rows, list(error_rows[0]))

    if fingerprint() != source_fingerprint:
        errors.append({"error": "agent or runner source changed during the suite"})
    manifest.update(
        {
            "status": "complete" if not errors else "partial",
            "completed_at": datetime.now(UTC).isoformat(),
            "successful_jobs": len(jobs) - len(errors),
            "failed_jobs": len(errors),
            "result_rows": len(rows),
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Results: {output}")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
