"""Paired official-simulator comparisons; reference bundle is frozen outside discovery."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import random
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from dme_simulator.agents import BaseAgent, Bid, AssemblyOrder, Listing, Decision
from dme_simulator.agents.loader import load_agent_bundle_class
from dme_simulator.config import load_config
from dme_simulator.runner import run_simulation, _validate_decision


class SimpleRival(BaseAgent):
    """Synthetic behaviors, not claims about any LLM or actual opponent."""
    def __init__(self, kind):
        self.kind = kind

    def decide(self, state, goods_config):
        goods = goods_config
        if self.kind == 'idle':
            return Decision()
        lots = {lot.good: lot for lot in state.auction_lots}
        variants = {'cheap': (1.15, 1.8, 3), 'hoarder': (2.1, 6.0, 8),
                    'margin': (1.4, 4.8, 3), 'random': (1.3, random.uniform(2, 8), 2)}
        bid_factor, markup, units = variants[self.kind]
        bids = [Bid(g, min(units * 2, lot.quantity), lot.reserve_price * bid_factor)
                for g, lot in lots.items()]
        assembly = [AssemblyOrder(name, units) for name in goods.recipes]
        prices = {}
        for recipe in goods.recipes.values():
            cost = sum(lots[m].reserve_price * n * bid_factor for m, n in recipe.inputs.items() if m in lots)
            prices[recipe.output_good] = max(.01, cost * markup)
        return Decision(bids=bids, assembly=assembly,
                        listings=[Listing(g, 10000, prices.get(g, 1)) for g in goods.finished_goods])


class Checked(BaseAgent):
    def __init__(self, agent):
        self.agent = agent
        self.errors = 0
        self.calls = 0

    def decide(self, state, goods_config):
        goods = goods_config
        self.calls += 1
        try:
            action = self.agent.decide(state, goods)
            _validate_decision(action, state, goods)
            return action
        except BaseException:
            self.errors += 1
            raise


def case(job):
    profile, scenario, roster, seed, teams = job
    old = load_agent_bundle_class(ROOT / 'benchmarks/reference_bundle')
    new = load_agent_bundle_class(ROOT / 'src/dme_simulator/agents/CompetitiveAgent')
    filename = {'scaled': 'scaled_market.yaml', 'tutorial_pricing': 'tutorial_pricing.yaml',
                'tutorial_auction': 'tutorial_auction.yaml'}.get(scenario, 'default.yaml')
    cfg = load_config(ROOT / 'config' / filename)
    if scenario == 'crowded':
        cfg = replace(cfg, consumer_market=replace(cfg.consumer_market, base_customers=12, amplitude=3))
    if scenario == 'short':
        cfg = replace(cfg, turns=12)
    cfg = replace(cfg, output=replace(cfg.output, render_plots=False, dashboard='none', progress_interval_turns=0))
    focal = Checked(old() if profile == 'original' else new())
    agents = {'Focal': focal}
    for index in range(teams - 1):
        if roster == 'copies':
            opponent = old(price_factor=(.75, 1, 1.2)[index % 3],
                           bid_aggressiveness=(.85, 1, 1.3)[index % 3])
        elif roster == 'selfplay':
            opponent = new()
        elif index % 3 == 0:
            opponent = old(pricing_mode=('margin', 'hybrid', 'ucb')[index % 9 // 3])
        else:
            opponent = SimpleRival(('idle', 'cheap', 'hoarder', 'margin', 'random')[index % 5])
        agents[f'Rival{index:02d}'] = opponent
    keys = list(agents)
    random.Random(seed).shuffle(keys)
    agents = {key: agents[key] for key in keys}
    simulator = run_simulation(cfg, agents, seed_override=seed, profile=True, quiet=True)
    assert focal.calls == cfg.turns, 'Focal agent was not called every turn'
    own = simulator.teams_state['Focal']
    h = simulator.history
    times = h.diagnostics.decision_duration_seconds['Focal']
    return dict(profile=profile, scenario=scenario, roster=roster, seed=seed, teams=teams,
                final_cash=own.capital, gain=own.capital-cfg.initial_state.starting_capital,
                rank=1+sum(s.capital > own.capital+1e-8 for s in simulator.teams_state.values()),
                min_cash=min(h.capital['Focal']),
                raw_left=sum(own.inventory[g] for g in cfg.goods.raw_materials),
                finished_left=sum(own.inventory[g] for g in cfg.goods.finished_goods),
                spend=sum(h.auction_spend['Focal']), revenue=sum(h.revenue['Focal']),
                max_ms=max(times)*1000, errors=focal.errors,
                deadline_exceeded=sum(t*1000 >= cfg.agents.decide_timeout_ms for t in times))


def fingerprint():
    h=hashlib.sha256()
    for folder in ['src/dme_simulator/agents/CompetitiveAgent', 'benchmarks/reference_bundle']:
        for path in sorted((ROOT/folder).glob('*.py')):
            h.update(path.relative_to(ROOT).as_posix().encode()); h.update(path.read_bytes())
    h.update(Path(__file__).read_bytes())
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--seed-start',type=int,default=21000)
    p.add_argument('--seeds',type=int,default=2)
    p.add_argument('--workers',type=int,default=3)
    p.add_argument('--scenarios',nargs='+',default=['default','crowded'])
    p.add_argument('--rosters',nargs='+',default=['mixed','copies'])
    p.add_argument('--teams',type=int,default=36)
    a=p.parse_args(); a.out.mkdir(parents=True,exist_ok=False)
    jobs=[(profile,scene,roster,seed,a.teams) for seed in range(a.seed_start,a.seed_start+a.seeds)
          for scene in a.scenarios for roster in a.rosters for profile in ['original','conservative']]
    manifest={'hash':fingerprint(),'jobs':jobs,'status':'running'}
    (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    rows=[]
    with (a.out/'partial.jsonl').open('w') as partial, ProcessPoolExecutor(max_workers=a.workers) as pool:
        for row in pool.map(case,jobs):
            rows.append(row); partial.write(json.dumps(row)+'\n'); partial.flush()
            print(f'{len(rows)}/{len(jobs)} {row["profile"]} {row["scenario"]} {row["roster"]} cash={row["final_cash"]:.1f} rank={row["rank"]}',flush=True)
    with (a.out/'runs.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    summary=[]
    for profile,scene,roster in sorted({(r['profile'],r['scenario'],r['roster']) for r in rows}):
        group=[r for r in rows if (r['profile'],r['scenario'],r['roster'])==(profile,scene,roster)]
        summary.append(dict(profile=profile,scenario=scene,roster=roster,runs=len(group),
            mean_rank=statistics.mean(r['rank'] for r in group),mean_gain=statistics.mean(r['gain'] for r in group),
            losses=sum(r['gain']<0 for r in group),max_ms=max(r['max_ms'] for r in group),
            errors=sum(r['errors'] for r in group)))
    (a.out/'summary.json').write_text(json.dumps(summary,indent=2))
    assert fingerprint()==manifest['hash'], 'Code changed during benchmark'
    manifest['status']='complete'; (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2))


if __name__=='__main__':
    main()
