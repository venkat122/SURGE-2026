# Named Entity Recognition (NER) Architecture

This document outlines the detailed architectural progression of our Named Entity Recognition models, highlighting the winning **Concatenation Model** that achieved the highest Macro-F1 score.

---

## 🏗️ Detailed Architectural Progression

### v1 Baseline Model (BiLSTM + CRF)
The standard sequence tagging architecture utilizing contextual embeddings from InLegalBERT, passed through a BiLSTM for temporal sequence modeling, and decoded via CRF.

```text
Input Legal Sentence
       │
       ▼
 [InLegalBERT Encoder]  ──> Extracts 768-dim contextual word embeddings
       │
       ▼
   [BiLSTM Layer]       ──> Captures bidirectional temporal sequence features
       │
       ▼
 [Linear Classifier]    ──> Projects embeddings to 29 dimensions (14 entities * 2 + O)
       │
       ▼
    [CRF Layer]         ──> Decodes token pathways globally via Viterbi transitions
       │
       ▼
 Predicted Named Entities (JUDGE, PETITIONER, STATUTE, etc.)
```

---

### v2 Syntactic Dependency Framework (GAT + CRF)
Replaces the BiLSTM with a Graph Attention Network (GAT) to model the structural dependencies of the sentence via spaCy dependency trees, completely ignoring temporal order in favor of syntactic relationships.

```text
Input Legal Sentence
       │
       ▼
 [InLegalBERT Encoder]  ──> Extracts 768-dim contextual word embeddings
       │
       │                        [spaCy Dependency Parser]
       │                                │
       ▼                                ▼ (Extracts syntactic tree)
  [Graph Attention Network (GAT)] <─────┘
       │
       │ (Updates node embeddings based on graph neighbors)
       ▼
 [Linear Classifier]    ──> Projects to 29 dimensions
       │
       ▼
    [CRF Decoder]       ──> Computes contextual transitions
```

---

### v3 Unified Concatenation Framework (BiLSTM + GAT + Concat + CRF) 
Our complete final system running a parallel architecture. It simultaneously captures the linear temporal dynamics of the sentence (via BiLSTM) and the complex structural dependencies of the legal text (via GAT). These distinct feature spaces are then concatenated to form a highly robust token representation before final CRF decoding.

```text
Input Legal Sentence (with 3-Sentence Context Window)
       │
       ▼
 [InLegalBERT Encoder]  ──> Extracts 768-dim contextual word embeddings
       │
       ├───┬────────────────────────────────────────┐
       │   │                                        │
       │   ▼                                        ▼
       │ [BiLSTM Layer]                       [spaCy Dependency Parser]
       │   │                                        │
       │   │ (Captures temporal dynamics)           ▼ (Extracts syntactic tree)
       │   │                                  [Graph Attention Network (GAT)]
       │   │                                        │
       │   ▼                                        ▼
       └───┴──────────> [Concatenation Layer] <─────┘
                               │
                               │ (Merges temporal and syntactic features)
                               ▼
                       [Linear Classifier]  ──> Projects to 29 dimensions (14 entities * 2 + O)
                               │
                               ▼
                          [CRF Decoder]     ──> Optimizes global sequence likelihoods
                               │
                               ▼
               Predicted Named Entities (JUDGE, PETITIONER, STATUTE, etc.)
```
