Di seguito trovi quattro configurazioni - tutte con **50 epoche** - pensate per isolare fattori diversi che, secondo la letteratura MoCo v2 e le peculiarità del tuo dataset (2 500 / 5 000 patch, batch-per-epoch molto ridotti), meritano di essere analizzati nel paper.
Ogni esperimento modifica **solo pochi iper-parametri chiave** per facilitarne l’interpretazione statistica.

---

## 1️⃣  *Queue size & Temperature*: «moco\_v2\_smallQ»

```yaml
moco_v2_smallQ:
  type: ssl
  backbone: "resnet18"
  proj_dim: 256
  patch_size: 224
  augmentation:
    enabled: true
    horizontal_flip: true
    rotation: [90, 180, 270]     # piccola variazione per invarianze di orientamento
    gaussian_blur: true
    grayscale: true
    color_jitter: {brightness: 0.4, contrast: 0.4, saturation: 0.4, hue: 0.1}
  training:
    epochs:        50            # ✔ invariato
    batch_size:    128           # metà del baseline → code più frequente
    optimizer:     "sgd"
    learning_rate: 0.03
    weight_decay:  1e-4
    queue_size:    256           # 🔻
    momentum:      0.99
    temperature:   0.07          # 🔻 default MoCo-v2 per set piccoli
    lr_schedule:   "cosine"
    warmup_epochs: 5
```

**Perché?**
Con \~1 500 sample/batch = 128 ⇒ solo 12 batch/epoch, la coda di 1 024 chiavi del baseline resta “vecchia” per molte epoche, introducendo rumore e possibile collasso (come si vede nel log con loss ≈ 8). Ridurre la **queue** a 256 e la **temperatura** a 0.07 (valore ottimale per MoCo v2 quando è presente l’MLP) dovrebbe:

* aumentare la freschezza dei negativi;
* rendere la distribuzione dei logit più scalata;
* stabilizzare la discesa della loss.

---

## 2️⃣  *Augmentazioni spinte*: «moco\_v2\_augStrong»

```yaml
moco_v2_augStrong:
  type: ssl
  backbone: "resnet18"
  proj_dim: 256
  patch_size: 224
  augmentation:
    enabled: true
    horizontal_flip: true
    rotation: [0, 90, 180, 270]
    gaussian_blur: true
    grayscale: true
    color_jitter: {brightness: 0.8, contrast: 0.8, saturation: 0.8, hue: 0.2}  # 🔺
  training:
    epochs:        50
    batch_size:    128
    optimizer:     "sgd"
    learning_rate: 0.03
    weight_decay:  1e-5
    queue_size:    512
    momentum:      0.99
    temperature:   0.20
    lr_schedule:   "cosine"
    warmup_epochs: 5
```

**Perché?**
Il paper MoCo v2 dimostra che **strong augmentation** (+ blur, + color-jitter) è ortogonale al batch-size e porta a boost lineari e di transfer. Su un dataset di istopatologia con patch estremamente simili, aumentare drasticamente la diversità sintattica → riduce “shortcut learning” su texture locali e forza il modello a catturare pattern più robusti.

*Hypothesis:* F1 macro del classifier lineare ↑, possibile leggera penalità su accuracy patch-level ma migliore generalizzazione paziente-level.

---

## 3️⃣  *Capacità del backbone & Momentum*: «moco\_v2\_deeper»

```yaml
moco_v2_deeper:
  type: ssl
  backbone: "resnet34"   # 🔺 più capiente
  proj_dim: 512          # 🔺
  patch_size: 224
  augmentation:
    enabled: true
    horizontal_flip: true
    gaussian_blur: true
    grayscale: true
    color_jitter: {brightness: 0.4, contrast: 0.4, saturation: 0.4, hue: 0.1}
  training:
    epochs:        50
    batch_size:    64            # ↓ per limiti memoria GPU
    optimizer:     "sgd"
    learning_rate: 0.05          # LR ∝ batch (linear scaling rule)
    weight_decay:  1e-4
    queue_size:    512
    momentum:      0.995         # 🔺 encoder_k più “inerziale”
    temperature:   0.20
    lr_schedule:   "cosine"
    warmup_epochs: 5
```

**Perché?**
Verifichiamo se un backbone **più profondo** + **proiezione 512-d** è utile nonostante il dataset compatto.
Aumentare la **momentum** a 0.995 compensa la maggior instabilità dovuta al minor numero di batch (queue update più dolce).

*Goal:* misurare trade-off costi/benefici tra capacità e overfitting; utile per discutere se reti “leggere” siano sufficienti in ambito medicale.

---

## 4️⃣  *Ottimizzatore alternativo*: «moco\_v2\_adam»

```yaml
moco_v2_adam:
  type: ssl
  backbone: "resnet18"
  proj_dim: 256
  patch_size: 224
  augmentation:
    enabled: true
    horizontal_flip: true
    gaussian_blur: true
    grayscale: true
    color_jitter: {brightness: 0.4, contrast: 0.4, saturation: 0.4, hue: 0.1}
  training:
    epochs:        50
    batch_size:    128
    optimizer:     "adam"        # 🔄
    learning_rate: 0.001         # tipico per Adam
    weight_decay:  1e-6
    queue_size:    512
    momentum:      0.99
    temperature:   0.20
    lr_schedule:   "cosine"
    warmup_epochs: 5
```

**Perché?**
La letteratura MoCo usa quasi sempre **SGD+momentum**; testare **Adam** quantifica il contributo del solo schema d’ottimizzazione su:

* velocità di convergenza (loss più bassa nelle prime epoche?);
* stabilità su dataset piccoli con feature già normalizzate;
* eventuale peggioramento nei transfer-task (Adam tende a sovradattare).

---

### Come presentare i risultati nel paper

| Exp | Principale variabile | Attesa vs baseline          | Se migliora → …                             |
| --- | -------------------- | --------------------------- | ------------------------------------------- |
| 1   | queue 256 + τ 0.07   | loss ↓, feature quality ↑   | argomentare importanza di negative freschi  |
| 2   | strong aug           | macro-F1 patch & paziente ↑ | mostrare robustezza a trasformazioni        |
| 3   | ResNet34 + proj 512  | se ↑ macro-F1 ma ↑ tempo    | discutere trade-off capacità vs overfitting |
| 4   | Adam                 | curva loss più regolare?    | discutere ruolo dell’ottimizzatore          |

Queste quattro varianti coprono **capacità del modello**, **dinamica di coda**, **forza delle viste** e **schema di ottimizzazione**: un ablation study completo e immediatamente confrontabile grazie alle stesse 50 epoche.
