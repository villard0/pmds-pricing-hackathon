# Workspace map

This repository now has one purpose per top-level directory:

- `src/dme_simulator/`: official simulator and the current `CompetitiveAgent`.
- `config/`: official example scenarios. These are kept unchanged.
- `simulation_lab/`: reproducible comparison environment for the two growth agents.
- `submission/`: the two ZIP files that must be preserved.
- `tests/`: fast correctness and timing checks.
- `docs/agent_interface.md`: official agent API reference.

Generated environments, plots, reports and virtual environments are not source
artifacts. Recreate the Python environment with `uv sync`; simulation results are
written below `simulation_lab/results/` and intentionally ignored by Git.

Start with:

```powershell
uv sync
uv run python simulation_lab/run_suite.py --list
uv run python simulation_lab/run_suite.py --suite smoke
```

See `simulation_lab/README.md` for the full scenario and roster catalog.
