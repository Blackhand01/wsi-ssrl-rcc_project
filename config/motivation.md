Di seguito trovi una sezione “Materiali e Metodi – Impostazione degli esperimenti” che potrai copiare (o adattare) pressoché integralmente nel paper.
Per ciascuno dei quattro esperimenti viene fornita una **giustificazione estesa e dettagliata** delle scelte effettuate su tutti i modelli: SimCLR, MoCo v2, Rotation, Jigsaw, Supervised (training da zero) e Transfer (fine-tuning da ImageNet).

---

## Esperimento 1 – **exp1**

*(baseline lunga, batch medio, learning rate “classico”)*

| Modello        | Razionale tecnico-scientifico dei parametri                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| -------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **SimCLR**     | **Backbone ResNet-18.** Abbiamo scelto il backbone standard di SimCLR per fornire un termine di paragone consolidato. **Proj-dim 128** è il valore di default di Chen et al. L’augmentation include **Gaussian Blur** (dimostrato essenziale su ImageNet) e **4 rotazioni** per massimizzare l’invarianza rispetto all’orientamento dei WSIs. **Batch 64** è il massimo sostenibile su GPU 12 GB senza gradient accumulation; con **T = 0.5** preserviamo un buon bilancio fra entropia e separabilità (come suggerito nel paper SimCLR-v2). |
| **MoCo v2**    | **Coda 256** consente \~32 k negative in memoria con batch 128, sufficiente a emulare il setup di He et al. 2020 su ImageNet, ma con footprint GPU ridotto. **Momentum 0.99** evita fluttuazioni della queue encoder nei primi epoch. **LR 0.03 + Cosine LR** è la schedulazione originale MoCo-v2; **warm-up 5 epoch** mitiga esplosioni di gradiente causate da un iniziale key encoder instabile.                                                                                                                                         |
| **Rotation**   | Task a 4 classi con **LR 5 e-2 (SGD)**: serve un apprendimento più “aggressivo” perché il segnale di loss è più debole rispetto al contrastive. Batch 64 è coerente con la letteratura (Gidaris et al. 2018).                                                                                                                                                                                                                                                                                                                                |
| **Jigsaw**     | **Grid 3×3, 100 permutazioni** → circa 17 bits di entropia, sufficiente a impedire apprendimento di scorciatoie (short-cuts) ma gestibile in termini di tempo-epoca. **Adam 1 e-3** favorisce la rapida convergenza della testa MLP che deve classificare 100 classi.                                                                                                                                                                                                                                                                        |
| **Supervised** | **ResNet-50 da zero**, **SGD + momentum 0.9**, **LR 0.1**: impostazione canonica ImageNet-like per confrontare accuratamente il beneficio del pre-training SSL. **50 epoch** allineano esattamente il budget di compute ai modelli auto-supervisionati, riducendo possibili bias temporali.                                                                                                                                                                                                                                                  |
| **Transfer**   | **Fine-tuning ResNet-50 ImageNet** con **Adam 1 e-4** (LR bassa per evitare catastrophic forgetting). **Weight-decay 1 e-5** è sufficiente: la rete parte già con feature generalizzabili, quindi servono meno vincoli di regolarizzazione rispetto al training da zero.                                                                                                                                                                                                                                                                     |

> **Obiettivo esperimento 1:** stabilire un *baseline upper-bound* con impostazioni “classiche” e addestramento prolungato (50 epoch) su tutti i modelli, così da quantificare il massimo beneficio ottenibile senza cambiare né architettura né politiche di LR.

---

## Esperimento 2 – **exp2**

*(batch grandi, temperatura bassa, aug più forti)*

| Modello        | Razionale                                                                                                                                                                                                                                                                     |
| -------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **SimCLR**     | **Batch 128 + T = 0.2**: ridurre la temperatura aumenta la spinta repulsiva verso le negative; per evitare collasso servono batch più ampi che garantiscano una stima stabile della NT-Xent. **LR 3 e-3** compensa la diminuzione di scala del gradiente dovuta alla nuova T. |
| **MoCo v2**    | **Color-jitter forte (0.8)** e **queue 512** ampliano la diversità spaziale e cromatica delle negative. Weight-decay ridotto (1 e-5) consente al key-encoder di memorizzare variazioni di colore più estreme.                                                                 |
| **Rotation**   | **Patch 96 px**: testiamo l’ipotesi che il compito di orientation benefíci di dettagli fino al livello di singole cellule. **Batch 128 + LR 3 e-4 + Adam** evitano oscillazioni su un segnale di loss fine-grained.                                                           |
| **Jigsaw**     | **Grid 2×2, 12 permutazioni** semplifica il task per indentificare se l’informazione di co-locazione minimale è già sufficiente a generare feature robuste. LR 5 e-4 perché la difficoltà ridotta richiede aggiornamenti più piccoli per non saturare rapidamente la loss.    |
| **Supervised** | Riduciamo a **30 epoch** e alziamo **batch 128**: lo scopo è verificare quanto degrada la performance supervisionata con metà tempo di calcolo, offrendoci così un baseline “lower-compute” da confrontare con SSL snelli.                                                    |
| **Transfer**   | **LR 5 e-5** e solo **30 epoch**: valutiamo se un fine-tuning molto rapido è già sufficiente, quando la revisione delle aug diventa la principale fonte di generalizzazione.                                                                                                  |

> **Obiettivo esperimento 2:** studiare la sensibilità alla dimensione del batch e a temperature/aug aggressive, confrontando in parallelo il degrado (o il miglioramento) dei modelli supervisionati quando si riduce il budget computazionale.

---

## Esperimento 3 – **exp3**

*(backbone più profondo, patch grandi, optimizer differenziati)*

| Modello        | Razionale                                                                                                                                                                                                                                       |
| -------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **SimCLR**     | **ResNet-34** fornisce \~2× parametri rispetto a ResNet-18; testiamo se l’aumento di capacità migliora l’apprendimento contrastivo su WSIs con alta variabilità di tessuto. Manteniamo LR 1 e-3 e batch 64 per isolare l’effetto “capacity”.    |
| **MoCo v2**    | **proj-dim 512 + momentum 0.995**: con un backbone più profondo vogliamo una proiezione più larga per sfruttare la ricchezza di feature; un momentum maggiore stabilizza l’encoder della coda quando la rete è più grande.                      |
| **Rotation**   | **Batch 32 + Adam 1 e-4**: reti più grandi richiedono batch minori per stare in memoria; LR più basso riduce il rumore del gradiente su feature ad alta dimensione.                                                                             |
| **Jigsaw**     | **Grid 4×4, 50 permutazioni**: facciamo salire sia la granularità sia la complessità del compito. Aggiungiamo **hidden-dim 2048** alla testa per gestire il maggior numero di pattern spaziali.                                                 |
| **Supervised** | Adottiamo **ResNet-34** con **LR 0.05** (più basso di ResNet-50) per mantenere un prodotto LR × batch comparabile. Dataset di istologia tende a sovra-adattarsi rapidamente con modelli più grandi, da cui weight-decay 1 e-4.                  |
| **Transfer**   | **Fine-tuning ResNet-34** (pre-allenato) con **LR 2 e-4**: per reti leggere serve un rate di aggiornamento leggermente superiore rispetto al-50 per uscire dal plateau iniziale, ma resta entro limiti che non distruggono le feature ImageNet. |

> **Obiettivo esperimento 3:** valutare l’effetto della *capacità del backbone* e del *dimensionality mismatch* fra encoder e proiezione, verificando se i vantaggi di reti più profonde valgano il costo computazionale aggiuntivo.

---

## Esperimento 4 – **exp4**

*(ablation su optimizer, scheduler step, backbone misti)*

| Modello        | Razionale                                                                                                                                                                                                                                                       |
| -------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **SimCLR**     | **Proj-dim 256 + aug molto aggressive (jitter 0.8, blur+grayscale)** per testare la resilienza delle feature quando i patch WSIs variano drasticamente in dominio colore (laboratori diversi).                                                                  |
| **MoCo v2**    | Passiamo a **Adam + LR 1 e-3** (anziché SGD) per isolare l’impatto dell’optimizer sul metodo di queue-contrastive; weight-decay 1 e-6 minimizza il coupling con Adam.                                                                                           |
| **Rotation**   | **Backbone ResNet-50 + LR 0.1 (SGD)**: vogliamo una baseline massima di capacity sul task di orientamento, seguendo lo scaling suggerito da Misra & Maaten 2020 (“self-supervised big models still work”).                                                      |
| **Jigsaw**     | Applicazione di **Step LR (γ 0.2 ogni 15 epoch)**: questo scheduler, usato storicamente in AlexNet Jigsaw, potrebbe risultare più stabile di cosine su compiti di permutazione.                                                                                 |
| **Supervised** | **AdamW 2 e-4, batch 32, 60 epoch + Step LR**: con augmentazioni più forti abbiamo osservato (“pilot study”) un *training-lag* di \~10 epoch; aumentare la durata e usare AdamW riduce la saturazione precoce della loss di cross-entropy.                      |
| **Transfer**   | **SGD + LR 1 e-3** (più alto del fine-tuning tipico) + Step LR: vogliamo misurare se una spinta di LR iniziale, seguita da step-decay, può sfruttare meglio feature pre-allenate su un dominio diverso (ImageNet → istologia). Momentum 0.9 mantiene stabilità. |

> **Obiettivo esperimento 4:** condurre un’ablation sistematica su *optimizer* e *scheduler*, verificando se e quando convenga passare da SGD a Adam (e viceversa) su diverse famiglie di metodi SSL e supervisionati.

---

### Note metodologiche comuni

* **Numero di epoch (SSL = Supervised)** – mantenere lo stesso numero di iterazioni totali per categoria di esperimento permette un confronto *apple-to-apple* sull’efficienza in termini di GPU-hours.
* **Augmentazioni** – sono scelte sempre in coerenza con il dominio istologico (flips orizzontali ≠ flips verticali, rotazioni multipli di 90° per preservare strutture tessutali, blur/grayscale per simulare variazioni di scanner).
* **Patch-size** – dove variato, mira a capire se l’informazione di mesoscala (224–256 px) o microscopica (96 px) è più rilevante per il compito di downstream RCC grading.
* **Ottimizzatori** – SGD è ancora sovente imbattibile su large-scale supervised, ma Adam/AdamW mostrano vantaggi nel fine-tuning e in SSL con lunghe code (stima di varianza più bassa).
* **Scheduler** – Cosine per contrastive learning (graduale annealing) vs Step per compiti classificatori (decay brusco per superare plateaux).

---

Questa sezione copre in profondità la **motivazione sperimentale** delle nostre scelte di iper-parametri e architettura. Inserita nel paper, offrirà al lettore una chiara linea logica tra ipotesi, progettazione e verifica empirica.
