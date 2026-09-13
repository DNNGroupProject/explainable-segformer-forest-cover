# Overfitting-mitigation grid (weight decay / LR schedule)

Seed 42. Gap = best-validation Dice minus final-epoch (epoch 20) validation Dice — how much validation performance decays once training runs past its early optimum. Baseline is the library-default AdamW (`weight_decay=0.01`, no schedule); the other three cells each change one knob.

| weight decay | LR schedule | variant | best-val Dice | final (ep 20) val Dice | gap | final val IoU |
|---|---|---|---|---|---|---|
| 0.01 (default) | none | vanilla | 0.7924 | 0.7614 | 0.0310 | 0.6849 |
| **0.05** | none | vanilla | 0.7922 | 0.7805 | **0.0117** | 0.7065 |
| 0.01 | cosine | vanilla | 0.7993 | 0.7805 | 0.0188 | 0.7046 |
| 0.05 | cosine | vanilla | 0.7955 | 0.7804 | 0.0151 | 0.7043 |
| 0.01 (default) | none | attention | 0.7941 | 0.7867 | 0.0074 | 0.7101 |
| 0.05 | none | attention | 0.7936 | 0.7654 | 0.0282 | 0.6866 |
| **0.01** | **cosine** | attention | 0.7916 | 0.7855 | **0.0061** | 0.7083 |
| 0.05 | cosine | attention | 0.7923 | 0.7722 | 0.0201 | 0.6960 |

`weight_decay=0.05` alone roughly halves the vanilla model's generalization gap (0.0310 → 0.0117) at no cost to best-validation Dice. The same setting nearly quadruples the attention-loss variant's gap (0.0074 → 0.0282) — extra weight decay helps `vanilla` and hurts `att`. A cosine LR schedule alone is the standout for the attention variant instead: it produces the smallest gap of the whole grid (0.0061) at almost no best-val cost. Net recommendation: `weight_decay=0.05` alone for the vanilla baseline; `cosine` alone (without added weight decay) for the attention-consistency variant. This is a single seed per cell, so treat it as a directional finding rather than a confirmed effect.
