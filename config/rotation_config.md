Di seguito trovi quattro varianti **plug-and-play** per il blocco `rotation:` del tuo file YAML.
Tutte tengono fissi i 50 epoche così potrai confrontare in modo pulito i risultati; ciò che cambia sono
backbone, dimensione della patch, batch-size, ottimizzatore, LR e weight-decay.

---

### 1. `rotation_sgd_light` – stessa capacità ma ottimizzatore *SGD*

```yaml
rotation:
  type: ssl
  backbone: "resnet18"          # stessa profondità del baseline
  patch_size: 224               # invariato
  training:
    epochs:        50           # fisso
    batch_size:    64
    optimizer:     "sgd"        # RotNet originale usa SGD con mom. 0 .9
    learning_rate: 5e-2         # LR classico di ResNet/SGD, scala con batch=64
    weight_decay:  1e-4
```

**Perché provarla?**
Nel paper RotNet (ICLR 2018) l’uso di *SGD+momentum* ha mostrato feature più generalizzabili rispetto ad Adam; su dataset piccoli potrebbe ridurre l’over-fitting che hai osservato (plateau a \~1.39 loss).

---

### 2. `rotation_patch96` – patch più piccola, batch grande

```yaml
rotation:
  type: ssl
  backbone: "resnet18"
  patch_size: 96                # ↑ dettaglio locale / invariance a contesto
  training:
    epochs:        50
    batch_size:    128          # entra in memoria perché la patch è 4× più piccola
    optimizer:     "adam"
    learning_rate: 3e-4         # LR ridotta per batch grande
    weight_decay:  1e-6         # WD ridotto: meno rischio di under-fit su patch minuscole
```

**Perché provarla?**
Con 2 500 ⇢ 5 000 patch totali, ridurre il FOV da 224→96 costringe la rete ad imparare micro-pattern (nuclei, stroma) utili nelle WSI. Inoltre un batch più grande stabilizza *Adam* (meno varianza nella stima del gradiente).

---

### 3. `rotation_resnet34` – backbone più profondo

```yaml
rotation:
  type: ssl
  backbone: "resnet34"
  patch_size: 224
  training:
    epochs:        50
    batch_size:    32           # RAM ↑ per la rete più grande
    optimizer:     "adam"
    learning_rate: 1e-4         # LR più bassa per rete più profonda
    weight_decay:  1e-5
```

**Perché provarla?**
Aumentare i livelli residui da 18→34 fa crescere la capacità (≈ 2× parametri) e potrebbe
catturare meglio la diversità intra-classe tipica di ccRCC vs pRCC. Manteniamo *Adam* per la stabilità su pochi dati, ma con LR e batch-size ridimensionati.

---

### 4. `rotation_resnet50_sgd` – “heavy-capacity + SGD”

```yaml
rotation:
  type: ssl
  backbone: "resnet50"
  patch_size: 224
  training:
    epochs:        50
    batch_size:    32           # compromesso memoria/varianza
    optimizer:     "sgd"
    learning_rate: 1e-1         # regola heuristic LR = 0.1 × (batch/256)
    weight_decay:  5e-4         # WD più alto per contrastare over-fitting
```

**Perché provarla?**
Vuoi testare il *ceiling* della capacità: ResNet-50 (≈ 25 M param.), *SGD* a LR alta + WD maggiore come in training ImageNet. Se i tuoi patch sono sufficienti, dovresti vedere se il gain di capacità compensa l’over-fit. In caso contrario, questo esperimento fornisce evidenza empirica per discutere i limiti del pretest RotNet su set di piccole dimensioni.

---

## Come documentare i risultati

1. **Confronto loss RotNet**: curva di training (50 ep) per vedere velocità di convergenza.
2. **Linear/1-NN probing**: estrai features e ri-addestra il tuo classificatore - usa le stesse 60/20/20 split.
3. **Patient-level Majority Voting**: identico protocollo del baseline, così puoi compilare una tabella 4 × (Accuracy/F1/Recall/Precision) per il paper.
4. **Analisi qualitativa**: t-SNE delle feature dei quattro modelli → mostra se patch96 separa meglio i pattern micro-istologici.

Con questi quattro set avrai una griglia di esperimenti (ottimizzatore, capacità, scala spaziale) che copre le variabili più significative per la self-supervised rotation prediction nel tuo contesto WSI.
