Qui sotto trovi **quattro configurazioni Jigsaw** pensate per esplorare altrettante ipotesi sperimentali – sempre con **50 epoche**, così da rendere diretto il confronto con il run di riferimento che hai già eseguito.

Ogni configurazione è presentata in YAML (stesso schema del tuo `config.yaml`) seguita da una breve **motivazione** che collega l’ipotesi alle intuizioni del paper di Noroozi & Favaro e alle caratteristiche del tuo dataset
(≈ 2 500 / 5 000 patch totali, split 60 / 20 / 20).

---

### 1. Più permutazioni per aumentare la “varianza spaziale”

```yaml
jigsaw:
  type:  ssl
  backbone:  "resnet18"
  grid_size: 3          # 3 × 3 come nel paper
  n_permutations: 100   # ↑ da 30 → 100 (top-3% Hamming distance)
  patch_size: 224
  training:
    epochs:        50
    batch_size:    64
    optimizer:     "adam"
    learning_rate: 1e-3
    weight_decay:  1e-5
```

**Perché**
Noroozi & Favaro mostrano che passare da ≈20 a 1000 permutazioni rende il compito più “informat- icamente ricco” perché costringe la rete a discriminare layout molto simili. Sul tuo dataset – più piccolo di ImageNet – 100 permutazioni sono un buon punto di equilibrio: quadruplichi la difficoltà senza esplodere il numero di classi né il rumore di stima dei gradienti. L’esperimento servirà a capire se, a parità di epoche, “più varianza spaziale” dà feature più trasferibili sulle WSI renali.

---

### 2. Puzzle più semplice (2 × 2) → verifica del **granularity gap**

```yaml
jigsaw:
  type:  ssl
  backbone:  "resnet18"
  grid_size: 2          # 2 × 2 ⇒ 4 tessere
  n_permutations: 12    # dalle 24 totali, max Hamming distance ≥3
  patch_size: 224
  training:
    epochs:        50
    batch_size:    64
    optimizer:     "adam"
    learning_rate: 5e-4 # LR più bassa (classe più “piccola”)
    weight_decay:  1e-5
```

**Perché**
Con pochi esempi i modelli self-sup trovano più segnali nei **pattern locali** che nella coerenza globale. Ridurre il grigliato a 2 × 2 isola macro-parti delle biopsie (es. stroma vs. area tumorale) e può facilitare l’ottimizzazione. Questo test ti dirà se, per un dataset medico limitato, è meglio imparare “relazioni grossolane” piuttosto che layout fini.

---

### 3. Puzzle più complesso (4 × 4) + backbone più capiente

```yaml
jigsaw:
  type:  ssl
  backbone:  "resnet34"   # capacità ↑
  grid_size: 4            # 4 × 4 ⇒ 16 tessere
  n_permutations: 50
  patch_size: 256         # divisibile per 4, FOV leggermente ↑
  hidden_dim: 2048        # testa più larga
  training:
    epochs:        50
    batch_size:    32     # memoria ↑ → batch ↓
    optimizer:     "adamw"
    learning_rate: 2e-4
    weight_decay:  5e-4
```

**Perché**
Aumentare la granularità costringe la rete a ragionare su **micro-indizi morfologici** (nuclei, micro-vascolarizzazione). Per reggere il carico di 16 tasselli concatenati (16 × 512 feat), servono una **ResNet 34** e una testa più larga. Il batch dimezzato evita OOM, mentre AdamW mitiga l’over-fitting tipico dei modelli grossi sul tuo set limitato.

---

### 4. Cambio di ottimizzatore: **SGD + momentum** (senza Adam)

```yaml
jigsaw:
  type:  ssl
  backbone:  "resnet18"
  grid_size: 3
  n_permutations: 30      # identico al baseline
  patch_size: 224
  training:
    epochs:        50
    batch_size:    64
    optimizer:     "sgd"
    learning_rate: 0.1    # LR iniziale alta, step decay
    weight_decay:  1e-4
    lr_scheduler:
      type: "step"
      step_size: 15
      gamma: 0.2
```

**Perché**
Nel paper originale la CFN è addestrata da zero con **SGD**: gradienti meno “rumorosi” ma maggiore stabilità di generalizzazione una volta trovato il minimo. Portare questa ricetta nel tuo setting permette di misurare se il boost di Adam alle prime epoche vale più o meno di una convergenza SGD “classica” con step-decay.

---

## Come mettere a fuoco i risultati

1. **Metriche interne al pretest**

   * Accuracies di classificazione della permutazione (proxy della difficoltà).
   * Curva di convergenza confrontabile perché le epoche sono fisse.

2. **Transfer sulle WSI**

   * Ripeti l’estrazione feat + Linear/k-NN + Majority-Voting per ognuna.
   * Confronto diretto del gap tra patch-level e patient-level.

3. **Analisi ablativa**

   * Effetto di `grid_size` (Conf 2 vs 3).
   * Effetto di `n_permutations` (Conf 1 vs baseline).
   * Effetto dell’ottimizzatore (Conf 4 vs baseline).

Queste quattro varianti coprono **ampiezza del compito**, **capacità del modello** e **dinamica di apprendimento**: tre assi riconosciuti dalla letteratura come cruciali per l’efficacia dei metodi Jigsaw e, allo stesso tempo, pienamente rilevanti per la tua scala di dati istologici.
