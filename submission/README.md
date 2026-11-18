# Preserved submissions

This directory intentionally contains two agent bundles:

- `CompetitiveAgent.zip`: **CurrentGrowth**, the strategy in
  `src/dme_simulator/agents/CompetitiveAgent/`.
- `uVeBeenPattinsoned-CompetitiveGrowth.zip`: **ImportedCompetitiveGrowth**,
  the externally supplied bundle preserved byte-for-byte.

Both archives contain exactly `agent.py`, `auctions.py`, `demand.py`,
`planning.py` and `pricing.py` at the ZIP root. The extracted imported agent used
by the simulation lab is in
`simulation_lab/agents/ImportedCompetitiveGrowth/`.
