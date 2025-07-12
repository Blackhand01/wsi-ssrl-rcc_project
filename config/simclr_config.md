Ecco quattro configurazioni **SimCLR** - tutte con lo stesso numero di epoche - pensate per esplorare i fattori che, in letteratura, influenzano maggiormente la qualità delle rappresentazioni auto-supervisionate. Ogni blocco YAML può essere copiato sotto `models:` nel tuo `training.yaml` (o salvato in file separati da dare al launcher).

---

### 1. **Baseline-plus** – controllo con lievi miglioramenti

Punto di partenza identico al run già eseguito, ma aggiungiamo **Gaussian Blur** (cruciale in SimCLR v2) e fissiamo `epochs` a 50 per allineare i confronti.

```yaml
simclr_exp1:
  type: ssl
  backbone: "resnet18"
  proj_dim: 128
  augmentation:
    enabled: true
    horizontal_flip: true
    rotation: [0, 90, 180, 270]
    color_jitter: {brightness: 0.4, contrast: 0.4, saturation: 0.4, hue: 0.1}
    gaussian_blur: true      # NOVITÀ
  training:
    epochs:        50        # invariato
    batch_size:    64
    optimizer:     "adam"
    learning_rate: 1e-3
    weight_decay:  1e-5
    temperature:   0.5
```

*Motivazione* Il blur simula la perdita di fuoco nei WSIs e rende la task contraria più difficile, spingendo il modello a focalizzarsi su strutture tessutali più robuste. Manteniamo lr, batch e temperatura originali per avere un “quasi-controllo” del tuo run precedente.&#x20;

---

### 2. **Large-batch & low-T** – scala secondo SimCLR

```yaml
simclr_exp2:
  type: ssl
  backbone: "resnet18"
  proj_dim: 128
  augmentation:
    enabled: true
    horizontal_flip: true
    rotation: [0, 90, 180, 270]
    color_jitter: {brightness: 0.4, contrast: 0.4, saturation: 0.4, hue: 0.1}
    gaussian_blur: true
  training:
    epochs:        50
    batch_size:    128        # +2×
    optimizer:     "adam"
    learning_rate: 3e-3       # lr ↗ proporzionale al batch
    weight_decay:  1e-5
    temperature:   0.2        # più bassa: penalizza di più i negativi
```

*Motivazione* Il paper SimCLR evidenzia che **batch più grande + temperatura più bassa** migliorano la stima delle distribuzioni di similarità tra viste; il lr viene scalato linearmente (rule-of-thumb di Goyal et al.) per mantenere la stabilità. Utile a testare se, con 5000 patch, il segnale di contrasto diventa più netto.

---

### 3. **Deeper encoder** – capacità di rappresentazione

```yaml
simclr_exp3:
  type: ssl
  backbone: "resnet34"        # encoder più profondo
  proj_dim: 128
  augmentation:
    enabled: true
    horizontal_flip: true
    rotation: [0, 90, 180, 270]
    color_jitter: {brightness: 0.4, contrast: 0.4, saturation: 0.4, hue: 0.1}
    gaussian_blur: true
  training:
    epochs:        50
    batch_size:    64          # come baseline per memoria
    optimizer:     "adam"
    learning_rate: 1e-3
    weight_decay:  1e-5
    temperature:   0.5
```

*Motivazione* WSIs contengono micro-pattern complessi; un **ResNet-34** offre \~2× parametri rispetto a ResNet-18, potendo apprendere feature più granulari senza variare batch/T. Confronto diretto della profondità a parità di epoche.

---

### 4. **Strong-aug + wider head** – regularizzazione estrema

```yaml
simclr_exp4:
  type: ssl
  backbone: "resnet18"
  proj_dim: 256              # proiezione più ampia
  augmentation:
    enabled: true
    horizontal_flip: true
    rotation: [0, 90, 180, 270]
    color_jitter: {brightness: 0.8, contrast: 0.8, saturation: 0.8, hue: 0.2}
    grayscale:      true     # +RandomGrayscale p=0.2
    gaussian_blur:  true
  training:
    epochs:        50
    batch_size:    64
    optimizer:     "adam"
    learning_rate: 1e-3
    weight_decay:  1e-5
    temperature:   0.5
```

*Motivazione* Spingiamo il modello con **augmentazioni più pesanti** (jitter + grayscale) per forzare invarianza cromatica, molto utile nei vetrini digitalizzati con variazioni di staining. Aumentiamo `proj_dim` a 256 per evitare collo di bottiglia quando il contrastive task diventa più difficile.

---

## Come documentare gli esperimenti

* **Tabella di confronto:** inserisci i quattro run con *loss finale*, *accuracy del linear probe* e *macro-F1* paziente-level.
* **Grafico di convergenza:** loss vs epoch; metterà in evidenza l’effetto di batch/T.
* **Discussion:** collega i risultati alle ipotesi (es. “low-T accelera l’abbassamento della loss ma richiede batch grande, confermando Chen et al.”).

Con queste varianti ottieni un ventaglio di ablation study che evidenziano: capacità del backbone, influenza dei parametri di temperatura e batch size, e ruolo della composizione di augmentazioni.
