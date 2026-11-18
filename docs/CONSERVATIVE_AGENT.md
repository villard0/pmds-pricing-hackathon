# Versione conservativa — 18 novembre 2026

Questa versione modifica il bundle ricevuto in `uVeBeenPattinsoned.zip`.
Il simulatore ufficiale resta invariato. Il bundle originale è conservato in
`benchmarks/reference_bundle/`, fuori dalla ricerca automatica degli agenti.
Le prestazioni dell'originale sono documentate in `reports/competitive/`;
non descrivono questa versione. Vedere il nuovo report `reports/CONSERVATIVE.md`.

## Comportamento scelto

- Una sola fascia di prezzo per prodotto: il feedback corrisponde al prezzo
  effettivamente offerto e non alla media delle sole unità vendute.
- Una unità obiettivo iniziale per prodotto; il piano può scegliere di non
  comprarla. Il prior non è ricavato dalla quantità di materiali disponibili.
- Espansione delle scorte in base alle vendite osservate, con limite massimo
  per prodotto. Dopo tre osservazioni consecutive senza vendite, il target
  di riassortimento diventa zero; le scorte esistenti continuano a essere offerte.
- La probabilità di vendita non ha un minimo artificiale del 28%. Il piano usa
  ricavi marginali decrescenti per nuove unità, tenendo conto delle scorte già
  possedute. Le code Poisson sono un'approssimazione, non la domanda ufficiale.
- Un solo zero non crea una barriera assoluta sul prezzo. Osservazioni vecchie
  perdono peso; gli stockout restano osservazioni censurate con uplift limitato.
- Prezzo iniziale proporzionale al costo di sostituzione; piccoli aumenti dopo
  stockout e riduzioni progressive dopo mancate vendite. Per smaltire scorte
  già pagate è consentito scendere sotto il costo di sostituzione. Il costo dei
  nuovi materiali continua invece a essere conteggiato nel piano d'acquisto.
- Riserva di liquidità pari al 97% del capitale osservato alla prima chiamata.
  Il totale dei pagamenti possibili non supera il saldo eccedente tale riserva.
  Questo limita la perdita finanziabile con nuovi acquisti al 3% del capitale
  iniziale, sotto le regole ufficiali senza ulteriori uscite obbligatorie.
  Non garantisce profitto, rango o la preservazione di tutti i guadagni accumulati.
- Il piano esclude acquisti che richiedono componenti non disponibili nei lotti;
  le offerte non vengono ridimensionate separatamente rompendo le ricette.
  Se il valore massimo è sotto la riserva, il portafoglio di offerte è rifiutato.
- Le ricette alternative dello stesso prodotto condividono lo stesso target.
  Le transazioni cumulative sono consumate per differenza; anche durante
  l'identificazione incerta del team i cursori avanzano, evitando di attribuire
  transazioni vecchie al piano dell'ultimo turno.

## Limiti deliberati

L'agente non conosce durata totale, capacità produttiva o distribuzione della
domanda: non legge gli YAML e non assume un ultimo turno fisso. Il limite di
produzione effettivo viene applicato dal simulatore; un riempimento parziale
delle aste può comunque lasciare componenti inutilizzati. Le aste non sono atomiche.

Il budget di rischio del 3% è una scelta conservativa, non un parametro
ottimizzato sui risultati della leaderboard. Può limitare l'esplorazione quando
i prodotti costano molto rispetto al capitale. I prezzi adattivi sono euristici,
non un algoritmo con garanzia di regret. Dopo uno stop di riassortimento senza
scorte residue non è garantita la riscoperta di una domanda tornata favorevole.

Le opzioni di pricing precedenti restano per confronti sperimentali. Il default
di consegna è `conservative`, con `ladder_levels=1` e `tiered_bids=False`.
Le ladder multiple non sono la variante validata per questa consegna.

Non è possibile conoscere gli agenti generati da Claude/GPT per altri team.
I test includono comportamenti sintetici economici, aggressivi, casuali,
inattivi, varianti dell'originale e copie della nuova strategia; non pretendono
di riprodurre i partecipanti reali o gli scenari privati della gara.

## Verifica e consegna

Con l'ambiente ufficiale installato, dalla radice del progetto:

```bash
uv run python -m unittest discover -s tests -v
uv run dme-sim --turns 5 --ui none --profile
uv run python benchmarks/conservative_benchmark.py --out reports/my-check --seed-start 41000 --seeds 3 --scenarios default crowded scaled short --rosters mixed copies selfplay --teams 36
```

Ogni output di benchmark deve essere nuovo. La classe focale deve essere stata
chiamata in tutti i turni, altrimenti il benchmark fallisce; vengono registrati
errori, tempi, liquidità minima, inventari grezzi/finiti e metriche per partita.
La selezione non va fatta dalla migliore singola partita. I report includono
gli hash del codice confrontato e dello script di benchmark.

Caricare sul portale soltanto lo ZIP del bundle conservativo con i cinque file
Python. L'archivio dell'intero progetto serve ai colleghi, non alla submission.
Nessuna dipendenza aggiuntiva è richiesta dall'agente: usa la libreria standard
e le classi pubbliche del simulatore. Il bundle deve essere caricato separatamente sul portale della gara.
