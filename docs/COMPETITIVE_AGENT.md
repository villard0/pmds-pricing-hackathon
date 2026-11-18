# CompetitiveAgent

## Scopo e compatibilità

`CompetitiveAgent` è un bundle autonomo per l'interfaccia ufficiale
`decide(state: Observation, goods_config: GoodsConfig) -> Decision`. Non legge
configurazioni del simulatore, non effettua I/O e non usa dati nascosti. I cinque
file Python del bundle usano soltanto la libreria standard e gli oggetti pubblici
del simulatore.

Differenze rispetto al progetto sperimentale 0.2.1:

- l'asta ufficiale è first-price/pay-your-bid, non second-price;
- acquisti, produzione e listing sono decisi insieme prima del clearing;
- non esistono costi di produzione o magazzino;
- capacità produttiva, distribuzione dei clienti e durata totale non sono esposte
  all'agente;
- le transazioni sono cumulative e, per default, limitate alle proprie;
- prezzi, ricette, quantità e numero di squadre non possono essere hardcoded;
- il capitale finale, non l'inventario, determina il rango.

## Architettura

- `agent.py`: unico `BaseAgent` concreto; coordina feedback, piano, bid, assembly
  e listing. Gli argomenti opzionali del costruttore servono alle ablation, ma
  l'istanza ufficiale non richiede argomenti.
- `planning.py`: beam search limitata a 96 stati e fallback greedy. Valuta insieme
  i componenti complementari e addebita come investimento soltanto i materiali
  mancanti, non quelli già posseduti.
- `demand.py`: stima kernel recency-weighted della domanda propria; i sell-out
  sono censurati (`domanda >= venduto`) e gli zeri non censurati impongono un
  limite prudente all'estrapolazione dei prezzi.
- `pricing.py`: griglia moltiplicativa rispetto al costo di sostituzione, modello
  esplicito di domanda, Sliding-Window UCB, GP-UCB puro Python e ladder di 1/2/3
  fasce. Il default selezionato è ibrido con tre fasce.
- `auctions.py`: apprendimento del fill-rate, bid first-price a due livelli,
  willingness-to-pay derivata dal margine della ricetta e vincolo sul pagamento
  massimo totale. Il vincolo resta valido qualunque materiale venga liquidato
  per primo.

## Algoritmi recuperati e adattati

Dal progetto sperimentale sono stati recuperati i principi di:

- pianificazione congiunta con beam e pruning;
- costo marginale dei componenti mancanti;
- domanda censurata e forgetting per mercati non stazionari;
- pricing bandit a finestra scorrevole;
- GP-UCB su una griglia di prezzi;
- pacing del budget e stime persistenti fra turni.

Sono stati modificati o sostituiti:

- bidding second-price -> bid first-price adattivi con price tiers;
- prezzi assoluti 5–30 -> moltiplicatori del costo osservato;
- Poisson parametrico legato al vecchio mercato -> stima kernel monotona basata
  sulle sole vendite accessibili;
- capacità/configurazione nota -> limiti derivati da lotti, ricette e numero di
  squadre pubblici;
- acquisti indipendenti -> portafoglio di ricette con complementi e budget comune;
- prezzo singolo obbligatorio -> ladder sperimentali supportate dal simulatore.

Il GP-UCB è rimasto disponibile come modalità sperimentale, ma non è il default:
nel confronto misurato è stato più lento e ha ottenuto un rango peggiore della
strategia selezionata.

## Riproduzione

Dal root del repository:

```bash
uv run python -m compileall -q src/dme_simulator/agents/CompetitiveAgent
uv run dme-sim --list-agents
uv run python -m unittest discover -s tests -v
uv run dme-sim --turns 5 --ui none --profile

# sviluppo rapido
uv run python benchmarks/competitive_benchmark.py --quick \
  --output reports/competitive/development

# confronto completo 4/8/16/32 agenti
uv run python benchmarks/competitive_benchmark.py \
  --counts 4 8 16 32 --seeds 2 --seed-start 9000 \
  --profiles competitive adaptive margin hybrid \
  --output reports/competitive/confirmation

# stress test separati dalle configurazioni ufficiali
uv run python benchmarks/competitive_benchmark.py \
  --counts 16 32 --seeds 1 --seed-start 12000 --stress \
  --profiles competitive adaptive margin \
  --output reports/competitive/stress
```

Il benchmark usa lo stesso seed, roster e ID delle squadre per ogni profilo
focale. CSV raw, summary e manifest sono scritti fuori dal bundle di upload.
