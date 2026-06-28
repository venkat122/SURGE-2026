v1 Baseline Model (No Hierarchical Transformer Context)
The original sequence architecture analyzing text streams without global document context blocks or adaptive dynamic boundary penalties.

Input Text (Legal Judgment)
       │
       ▼
 [LegalBERT Encoder] ──> Extracts 768-dim word/sentence level distributions
       │
       ▼
  [BiLSTM Layer]     ──> Captures bidirectional sequence features (512-dim)
       │
       ▼
[Linear Classifier]  ──> Projects embeddings to NUM_LABELS dimensions (512 ──> 14)
       │
       ▼
   [CRF Layer]       ──> Decodes token pathways globally via Viterbi transition weights
       │
       ▼
Predicted Rhetorical Roles (ANALYSIS, FAC, ISSUE, PREAMBLE, etc.)

v2 Hierarchical Context Framework (LegalBERT + Transformer)
Introduces hierarchical macro-context mapping by stacking a global sequence aggregator layer above the raw BERT embeddings to track long-range document layout dependencies.
Input Sentence Chunks (Size = 4)
       │
       ▼
 [LegalBERT Encoder]          ──> Unfrozen backbone generating dense feature vectors
       │
       ▼
[3-Layer Context Transformer] ──> Multi-Head Self-Attention updates global positioning
       │
       ▼
 [Feature Dense Head]         ──> Linear activation block + LayerNorm + GELU + Dropout
       │
       ▼
 [Linear Classifier]          ──> Projects structural constraints to 14 classes
       │
       ▼
   [CRF Decoder]              ──> Computes contextual transitions to enforce continuity
   Performance Baseline Update: Effectively captured topology across extended document layers, pushing overall sequence token mapping stability to a saturated Weighted F1-score of 0.82.

   v3 Boundary-Aware Framework (LegalBERT + Shift + Transformer)
Our complete final system running an adaptive auxiliary multi-task penalty framework. It computes distribution variances across contiguous target boundaries to counteract majority sequence bias and handle extreme class imbalances.
[Global Context Output Logits]
                                         │
                    ┌────────────────────┴────────────────────┐
                    ▼                                         ▼
       [Native CRF Sequence Loss]                 [_compute_label_shift_penalty]
                    │                                         │
                    ▼                                         ▼
       Standard sequence optimization             Adjacent Cosine Probability Delta
                    │                                         │
                    └────────────────────┬────────────────────┘
                                         ▼
                            [Weighted Mean Combination]
                     Total Loss = (Alpha * CRF) + (Beta * Shift)