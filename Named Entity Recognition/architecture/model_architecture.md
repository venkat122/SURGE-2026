
```
Input Text (Legal Judgment)
        │
        ▼
┌─────────────────────────┐
│   3-Sentence Context    │  ← FLERT-style: [Prev Sentence] + [Target Sentence] + [Next Sentence]
│      Window             │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│    InLegalBERT          │  ← Pre-trained Legal BERT encoder (768-dim)
│  (Frozen Epoch 1)       │     evolawyer/inlegalbert-sc-ner-silver
└───────────┬─────────────┘
            │
            ▼
    Word-level Representations (768-dim)
            │
     ┌──────┴──────┐
     │             │
     ▼             ▼
┌─────────┐  ┌──────────────────┐
│ BiLSTM  │  │  Syntactic GAT   │  ← spaCy dependency parse tree
│ (512)   │  │  (512)           │     edges feed the graph structure
└────┬────┘  └────────┬─────────┘
     │                │
     ▼                ▼
  Sequential       Structural
  Features         Features
  (512-dim)        (512-dim)
     │                │
     └───────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │  Gated Fusion   │  ← g = σ(W·[h_seq; h_syn])
    │  g·h_seq +      │     output = g·h_seq + (1-g)·h_syn
    │  (1-g)·h_syn    │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ Linear Classifier│  ← Projects to NUM_LABELS dimensions
    │    (512 → 29)   │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │       CRF       │  ← Viterbi decoding ensures valid BIO sequences
    │ (Conditional    │
    │  Random Field)  │
    └────────┬────────┘
             │
             ▼
    BIO Tag Sequence
    [O, B-JUDGE, I-JUDGE, O, B-COURT, I-COURT, ...]
```


 v2 Concatenation (No Gated Fusion)
Same as above, but the Gated Fusion layer is replaced with simple concatenation:
```
BiLSTM output (512) ──┐
                      ├── torch.cat ──► Linear (1024 → 29) ──► CRF
GAT output (512) ─────┘
```

v2 GAT-Only (No BiLSTM)
The BiLSTM is completely removed:
```
InLegalBERT (768) ──► GAT (512) ──► Linear (512 → 29) ──► CRF
```

v1 Baseline (No GAT)
The original architecture without any syntactic features:
```
InLegalBERT (768) ──► BiLSTM (512) ──► Linear (512 → 29) ──► CRF
```
 Training Strategy
- **Epoch 1**: Encoder (InLegalBERT) is **frozen** — only BiLSTM, GAT, Fusion, and CRF learn
- **Epochs 2–5**: Encoder is **unfrozen** — entire model fine-tunes end-to-end
- **Optimizer**: AdamW with discriminative learning rates (Encoder: 2e-5, Head: 1e-3)
- **Early Stopping**: Patience of 2 epochs based on Dev Macro-F1
