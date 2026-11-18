# Validazione di CompetitiveAgent

Data: 18 novembre 2026.

## Esito

- Bundle scoperto correttamente dal loader ufficiale.
- 5 file Python, nessuna dipendenza esterna, una sola classe concreta in
  `agent.py`, costruttore senza argomenti valido.
- 4/4 test automatici passati.
- 668 partite di benchmark completate fra sviluppo, holdout, conferma, stress,
  confronto dei modelli di pricing e baseline pubbliche.
- 0 decisioni invalide o scadute nei benchmark strumentati.
- Conferma finale Linux: 1,42 ms medi e 15,70 ms massimi per decisione sul
  campione aggregato; GP-UCB ha raggiunto 24,22 ms massimi. Entrambi restano
  sotto 100 ms, ma il default mantiene più margine.
- Smoke test Windows con tracing: massimo osservato 34,40 ms.

Questi risultati sono locali e non costituiscono una garanzia sugli scenari
privati.

## Baseline pubbliche

Capitale finale, roster di quattro agenti, seed 42. Il rango di
`CompetitiveAgent` è 1 in tutte e quattro le partite.

| Scenario | Competitive | Dummy | Patient | PriceWar |
|---|---:|---:|---:|---:|
| default | 98.286,01 | 20.155,01 | 12.000,00 | 8.712,67 |
| scaled_market | 376.475,61 | 65.936,89 | 60.000,00 | 61.459,59 |
| tutorial_pricing | 23.577,59 | 5.150,80 | 5.000,00 | 5.096,63 |
| tutorial_auction | 7.429,12 | 5.468,00 | 5.000,00 | 5.148,48 |

## Conferma su semi non usati per lo sviluppo

Semi 9000–9001, roster da 4/8/16/32 squadre, avversari fissi comprendenti
Dummy, Patient, PriceWar e copie adattive forti. Le grandezze sono medie su 8
partite per scenario; il criterio principale è il rango medio.

| Scenario | Rango medio | Vittorie | Capitale | P10 capitale | Venduti | Invenduto | Spesa aste | Ricavi | Media ms | Max ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| default | 5,50 | 25,0% | 42.904,99 | 11.964,24 | 164,38 | 6,38 | 12.159,08 | 43.064,06 | 1,84 | 7,70 |
| scaled_market | 3,75 | 25,0% | 182.962,80 | 80.236,12 | 335,12 | 8,88 | 24.560,69 | 147.523,48 | 3,03 | 15,70 |
| tutorial_pricing | 5,25 | 25,0% | 10.133,39 | 5.260,86 | 97,88 | 3,25 | 214,03 | 5.347,42 | 0,51 | 1,83 |
| tutorial_auction | 7,38 | 50,0% | 6.597,42 | 4.929,82 | 29,75 | 1,25 | 1.027,39 | 2.624,82 | 0,31 | 1,47 |

Il P10 è calcolato separatamente per scenario, perché i capitali iniziali hanno
scale diverse.

## Confronto pricing

Semi 15000–15001, `default` e `scaled_market`, 4/8 squadre, 8 partite per
profilo. Tutti i profili hanno 0 errori/timeout.

| Profilo | Rango medio | Vittorie | Capitale medio | Media ms | Max ms |
|---|---:|---:|---:|---:|---:|
| Margine | 1,375 | 75,0% | 157.125,7 | 3,24 | 15,33 |
| Domanda esplicita | 1,750 | 62,5% | 168.863,8 | 3,68 | 14,70 |
| Ibrido + ladder-3 (selezionato) | 2,000 | 50,0% | 170.022,3 | 3,75 | 15,52 |
| GP-UCB | 2,250 | 50,0% | 112.353,7 | 4,97 | 24,22 |
| Sliding-Window UCB | 2,625 | 50,0% | 88.984,0 | 2,51 | 13,55 |

Il margine vince questo campione ristretto, ma la selezione finale usa tutti i
campioni: su 40 confronti multi-prodotto combinati ladder-3 ottiene rango 3,625
e 52,5% vittorie, contro rango 3,800 e 45,0% del margine. Ladder-3 è inoltre
nettamente più robusta nei tutorial mono-prodotto. Per questo resta il default;
`margin`, `demand`, `ucb` e `gp` rimangono riproducibili come ablation.

## Ablation e stress

Sono stati confrontati: pianificazione congiunta vs greedy, domanda on/off,
SW-UCB, GP-UCB, margine, domanda esplicita, ladder 1/2/3, bid tiers on/off,
budget adattivo/fisso e aggressività d'asta conservativa/aggressiva.

Lo stress test genera configurazioni in memoria, separate dagli YAML ufficiali:
offerta ridotta (`stress_scarce`) e domanda ridotta/competizione affollata
(`stress_crowded`). Su 24 partite focali ladder-3 con 16/32 squadre il rango
medio è 10,46, inventario invenduto medio 5,71, massimo 8,08 ms e 0 errori.
Nello stesso campione adattiva e margine ottengono ranghi 8,71 e 9,25: è una
criticità reale della selezione ladder-3 in alcuni mercati default molto
affollati. Sullo scaled market scarso ladder-3 torna invece competitiva.

## Criticità residue

- La durata totale non è esposta: l'agente non può liquidare esplicitamente
  nell'ultimo turno; l'inventario finale medio non è zero.
- Con 32 squadre e materie prime molto scarse la varianza per seed è elevata e
  nessuna strategia domina tutti gli scenari.
- Le transazioni proprie non identificano la domanda persa verso i concorrenti;
  gli inventari pubblici forniscono soltanto una proxy della pressione.
- La capacità produttiva non è in `GoodsConfig`; gli ordini sono quindi
  prioritizzati, ma l'agente non può pianificare sul limite esatto.
- GP-UCB è compatibile col limite ma non ha migliorato rango o capitale nei test.

## Artefatti

- `development-*`: semi 1000–1001, selezione iniziale.
- `holdout-*`: semi 5000–5001, 4/8/16/32 squadre.
- `final-*`: semi 8000–8001, confronto intermedio.
- `confirmation-*`: semi 9000–9001, conferma dopo la selezione.
- `stress-*`: seed 12000, scenari generati separatamente.
- `pricing-models-*`: semi 15000–15001, incluso GP-UCB.
- `baseline42-*`: baseline pubbliche seed 42.

Ogni famiglia contiene CSV raw, CSV summary e manifest JSON. Il bundle di upload
non contiene report, test, cache o grafici.
