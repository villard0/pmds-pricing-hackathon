# Simulation lab

This is the single comparison environment for the two strategies that must be
preserved:

- **CurrentGrowth** loads the current source bundle from
  `src/dme_simulator/agents/CompetitiveAgent/`.
- **ImportedCompetitiveGrowth** is the frozen bundle supplied separately and
  stored under `simulation_lab/agents/ImportedCompetitiveGrowth/`.

Both focal agents face exactly the same scenario, seed and opponent roster in
every job. Team insertion order is shuffled deterministically to reduce ordering
bias. The official simulator and official YAML files are not modified.

## Quick start

From the repository root:

```powershell
uv sync
uv run python simulation_lab/run_suite.py --list
uv run python simulation_lab/run_suite.py --suite smoke
```

The smoke suite is deliberately tiny: one five-turn duel. It checks loading,
valid decisions, result serialization and the 100 ms timing telemetry.

Useful suites:

```powershell
# Broad development matrix: 8 markets x 2 rosters x 3 seeds
uv run python simulation_lab/run_suite.py --suite core --workers 3

# Long and crowded games, including 240 turns and 43 teams
uv run python simulation_lab/run_suite.py --suite extended --workers 3

# Every catalog entry and roster (large)
uv run python simulation_lab/run_suite.py --suite all --seeds 5 --workers 4

# A targeted paired experiment
uv run python simulation_lab/run_suite.py --scenarios extended_240 scarce_supply --rosters standard crowded_43 --seeds 5 --seed-start 31000 --workers 3
```

Use `--dry-run` to inspect the matrix without executing it. `--turns` overrides
the duration for every selected scenario, which is useful for quick diagnostics.

## Scenario catalog

The catalog covers different failure modes instead of repeating one market:

- `baseline_40`: unchanged default market.
- `extended_240`: 240-turn, EUR 40,000 compounding environment.
- `scaled_growth`: larger volumes and EUR 60,000 capital.
- `scarce_supply`: strong component scarcity.
- `abundant_market`: plentiful supply and deeper demand.
- `demand_recession`: fewer customers and lower willingness to pay.
- `demand_boom`: many customers and stronger willingness to pay.
- `production_bottleneck`: strict per-turn factory capacity.
- `price_sensitive`: demand reacts strongly to price.
- `price_tolerant`: less price-sensitive, higher-value customers.
- `volatile_demand`: large, fast customer-cycle swings.
- `premium_market`: smaller but much richer customer pool.
- `low_capital`: tests survival and capital efficiency from EUR 5,000.

## Roster catalog

- `duel`: only the two focal agents.
- `standard`: focal agents plus one Dummy, Patient and PriceWar opponent.
- `pressure`: 12 teams, dominated by PriceWar agents.
- `crowded_12`: 12 teams with a balanced synthetic mix.
- `selfplay_16`: focal agents plus seven copies of each growth strategy.
- `crowded_43`: 43 teams, matching the observed leaderboard scale.

## Outputs

Each run creates a timestamped directory in `simulation_lab/results/`:

- `manifest.json`: exact matrix, status, environment and source fingerprint.
- `runs.csv`: one row per focal agent per simulation.
- `summary.csv`: aggregates by strategy, scenario and roster.
- `pairwise.csv`: capital delta and winner between the two focal agents.
- `partial.jsonl`: append-only recovery trail while work is running.
- `errors.csv`: failed jobs, if any.

The main metrics are final capital, return, rank, drawdown, sales, leftover raw
and finished inventory, auction spend, revenue, operating ROI, decision latency,
deadline breaches and agent errors. Results are evidence for local comparison,
not a claim about the official leaderboard.
