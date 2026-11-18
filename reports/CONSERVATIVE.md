# Versione conservativa — verifica del 18 novembre 2026

Modifiche e limiti: [CONSERVATIVE_AGENT.md](../docs/CONSERVATIVE_AGENT.md).

17 test passati. Valutazione completata: 72 partite, 36 squadre, semi 31000–31002,
scenari default, domanda ridotta, mercato scalato e partita breve (12 turni).
36 partite per versione, confrontate con stessi semi, ordine delle squadre e
avversari. Nessun errore focale o decisione oltre il limite di 100 ms.
Il codice del simulatore ufficiale non è stato modificato.

| Avversari | Rango medio originale | Rango medio conservativo | Partite in perdita originale / nuovo |
|---|---:|---:|---:|
| Misti sintetici | 13.92 | 1.00 | 4/12 / 0/12 |
| Varianti dell’originale | 11.42 | 1.25 | 4/12 / 0/12 |
| Copie della nuova versione | 12.42 | 17.92 | 1/12 / 1/12 |

**Limite osservato:** contro copie della strategia conservativa il rango medio
è peggiore dell'originale. Non si rivendica una strategia dominante o una vittoria
assicurata contro agenti reali o scenari privati. I gruppi sintetici non sono
output verificati di Claude/GPT. Tre semi per condizione sono un campione limitato.
Il massimo tempo osservato del nuovo agente è 46,49 ms (Windows con tracing).

La riserva monetaria protegge il 97% del capitale iniziale sotto le regole
ufficiali senza altre uscite obbligatorie; non garantisce un profitto positivo.
Il nuovo agente ha chiuso sotto il capitale iniziale in 1 delle 36 partite,
contro 9 delle 36 dell'originale. Sviluppo precedente: 16 partite su semi
21000–21001, conservate separatamente in conservative-dev-valid.

Dati completi, configurazioni nominali, semi e hash: conservative-evaluation/.
Il benchmark documenta nel sorgente le trasformazioni di scenario e i roster.
I report in competitive/ riguardano la versione originale ricevuta.

## File da usare

- `submission/uVeBeenPattinsoned-conservative.zip`: solo i cinque file Python,
  pronto per il portale. Il vecchio `submission/CompetitiveAgent.zip` è stato
  aggiornato allo stesso contenuto per evitare upload accidentali della versione vecchia.
- `src/dme_simulator/agents/CompetitiveAgent/`: sorgenti da copiare nel repository
  ufficiale; sostituire tutti e cinque i file insieme.
- `benchmarks/reference_bundle/`: originale congelato, non scoperto come agente
  dal simulatore; serve per ripetere il confronto.

Nessun upload al portale della gara è stato eseguito.
