fold,split,n_patients,patient_ids
fold0,train,12,HP17.7980;HP18.11474;HP18.5818;HP18005453;HP18014084;HP19.4372;HP19.5524;HP19.754;HP19012316;HP20.2506;HP20.5602;HP20001530
fold0,val,4,HP19.10064;HP19.1773;HP20002300;HP20002450
fold1,train,12,HP17.7980;HP18.11474;HP18005453;HP18014084;HP19.10064;HP19.1773;HP19.5524;HP19.754;HP20.2506;HP20.5602;HP20002300;HP20002450
fold1,val,4,HP18.5818;HP19.4372;HP19012316;HP20001530
fold2,train,12,HP18.11474;HP18.5818;HP18014084;HP19.1773;HP19.4372;HP19.5524;HP19.754;HP19012316;HP20.5602;HP20001530;HP20002300;HP20002450
fold2,val,4,HP17.7980;HP18005453;HP19.10064;HP20.2506
fold3,train,12,HP17.7980;HP18.11474;HP18.5818;HP18005453;HP18014084;HP19.10064;HP19.4372;HP19.5524;HP19012316;HP20.2506;HP20.5602;HP20001530
fold3,val,4,HP19.1773;HP19.754;HP20002300;HP20002450
fold4,train,12,HP17.7980;HP18.5818;HP18005453;HP19.10064;HP19.1773;HP19.4372;HP19.754;HP19012316;HP20.2506;HP20001530;HP20002300;HP20002450
fold4,val,4,HP18.11474;HP18014084;HP19.5524;HP20.5602


✔️ Fold0 salvato: train=1500 patch, val=500, test_holdout=500

📋 Fold0 train
  pazienti per classe: {'CHROMO': 3, 'ONCO': 3, 'ccRCC': 3, 'not_tumor': 6, 'pRCC': 3}
  patch per classe:    {'CHROMO': 300, 'ONCO': 300, 'ccRCC': 300, 'not_tumor': 300, 'pRCC': 300}

📋 Fold0 val
  pazienti per classe: {'CHROMO': 1, 'ONCO': 1, 'ccRCC': 1, 'not_tumor': 2, 'pRCC': 1}
  patch per classe:    {'CHROMO': 100, 'ONCO': 100, 'ccRCC': 100, 'not_tumor': 100, 'pRCC': 100}
✔️ Fold1 salvato: train=1525 patch, val=475 patch

📋 Fold1 train
  pazienti per classe: {'CHROMO': 3, 'ONCO': 3, 'ccRCC': 3, 'not_tumor': 6, 'pRCC': 3}
  patch per classe:    {'CHROMO': 303, 'ONCO': 308, 'ccRCC': 295, 'not_tumor': 305, 'pRCC': 314}

📋 Fold1 val
  pazienti per classe: {'CHROMO': 1, 'ONCO': 1, 'ccRCC': 1, 'not_tumor': 2, 'pRCC': 1}
  patch per classe:    {'CHROMO': 97, 'ONCO': 92, 'ccRCC': 105, 'not_tumor': 95, 'pRCC': 86}
✔️ Fold2 salvato: train=1476 patch, val=524 patch

📋 Fold2 train
  pazienti per classe: {'CHROMO': 3, 'ONCO': 3, 'ccRCC': 3, 'not_tumor': 6, 'pRCC': 3}
  patch per classe:    {'CHROMO': 298, 'ONCO': 293, 'ccRCC': 300, 'not_tumor': 297, 'pRCC': 288}

📋 Fold2 val
  pazienti per classe: {'CHROMO': 1, 'ONCO': 1, 'ccRCC': 1, 'not_tumor': 2, 'pRCC': 1}
  patch per classe:    {'CHROMO': 102, 'ONCO': 107, 'ccRCC': 100, 'not_tumor': 103, 'pRCC': 112}
✔️ Fold3 salvato: train=1495 patch, val=505 patch

📋 Fold3 train
  pazienti per classe: {'CHROMO': 3, 'ONCO': 3, 'ccRCC': 3, 'not_tumor': 6, 'pRCC': 3}
  patch per classe:    {'CHROMO': 300, 'ONCO': 300, 'ccRCC': 294, 'not_tumor': 301, 'pRCC': 300}

📋 Fold3 val
  pazienti per classe: {'CHROMO': 1, 'ONCO': 1, 'ccRCC': 1, 'not_tumor': 2, 'pRCC': 1}
  patch per classe:    {'CHROMO': 100, 'ONCO': 100, 'ccRCC': 106, 'not_tumor': 99, 'pRCC': 100}
✔️ Fold4 salvato: train=1504 patch, val=496 patch

📋 Fold4 train
  pazienti per classe: {'CHROMO': 3, 'ONCO': 3, 'ccRCC': 3, 'not_tumor': 6, 'pRCC': 3}
  patch per classe:    {'CHROMO': 299, 'ONCO': 299, 'ccRCC': 311, 'not_tumor': 297, 'pRCC': 298}

📋 Fold4 val
  pazienti per classe: {'CHROMO': 1, 'ONCO': 1, 'ccRCC': 1, 'not_tumor': 2, 'pRCC': 1}
  patch per classe:    {'CHROMO': 101, 'ONCO': 101, 'ccRCC': 89, 'not_tumor': 103, 'pRCC': 102}

🔎 Verifica che i pazienti nei fold siano differenti:
  • Fold1 vs Fold2 → comuni (train): 8 | comuni (val): 0
  • Fold1 vs Fold3 → comuni (train): 8 | comuni (val): 0
  • Fold1 vs Fold4 → comuni (train): 8 | comuni (val): 0
  • Fold2 vs Fold3 → comuni (train): 8 | comuni (val): 0
  • Fold2 vs Fold4 → comuni (train): 8 | comuni (val): 0
  • Fold3 vs Fold4 → comuni (train): 8 | comuni (val): 0

