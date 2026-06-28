# SAIL-NER: Syntax and Structure-Aware InLegalBERT for Indian Legal Named Entity Recognition

> **SURGE 2026** — Summer Undergraduate Research Grant for Excellence  
> Indian Institute of Technology, Kanpur

## Overview

This repository contains the codebase and experiments for extracting 14 Legal Named Entities from Indian Supreme Court judgments using the EkStep dataset. We fine-tuned the `evolawyer/inlegalbert-sc-ner-silver` model and conducted extensive ablation studies comparing different architectures (BiLSTM, GAT, Concatenation, and Gated Fusion).

All experiments were executed using cloud accelerators: the Kaggle T4 GPU and the Lightning.ai L4 GPU environments.

## Repository Structure

```text
NER/
├── README.md
├── .gitignore
│
├── src/                                # Jupyter Notebooks for each model architecture
│   ├── baseline/
│   │   └── baseline.ipynb              # BiLSTM + CRF
│   ├── gat_only/
│   │   └── gat_only.ipynb              # GAT + CRF
│   ├── concat/
│   │   └── concat.ipynb                # BiLSTM + GAT + Concat + CRF
│   └── gated_fusion/
│       └── gated_fusion.ipynb          # BiLSTM + GAT + Gated Fusion + CRF
│
├── results/                            # Test results and training curves
│   ├── ablation_comparison.md
│   ├── v2_gated_fusion_results.txt
│   ├── v2_concat_results.txt
│   ├── v2_gat_only_results.txt
│   ├── v2_gated_fusion_training_curves.png
│   └── v2_concat_training_curves.png
│
├── data/                               # Dataset info and download links
│   └── README.md
│
├── references/                         # Research papers
│   ├── InLegalBERT_2023.pdf
│   ├── Legal_NER_EkStep_2210.pdf
│   ├── Legal_NER_Survey_2024.pdf
│   ├── Legal_Entity_Extraction.pdf
│   ├── TEXT2KG_Paper_6.pdf
│   ├── Legal_NER_Frontiers_AI.pdf
│   ├── Effective_Search_Legal_NER.pdf
│   └── NER_Legal_Book_Chapter.pdf
│
└── architecture/                       # Architecture documentation
    └── model_architecture.md
```

## Architectures & Results

We evaluated four primary model architectures against the unseen EkStep test dataset. The following hyperparameters were held constant across all models: Batch Size = 8, Max Encoder Tokens = 512, Gradient Accumulation = 2, and 5 Epochs.

| Model Architecture | GPU Used | Macro-F1 |
|---|---|---|
| **BiLSTM + CRF** | Kaggle T4 | 81.31% |
| **GAT + CRF** | Lightning.ai L4 | 83.97% |
| **BiLSTM + GAT + Concat + CRF** | Lightning.ai L4 | 85.77% |
| **BiLSTM + GAT + Gated Fusion + CRF** | Kaggle T4 | **85.83%** |

### Per-Entity F1 Score Comparison

| Entity | BiLSTM + CRF | GAT + CRF | BiLSTM + GAT + Concat + CRF | BiLSTM + GAT + Gated Fusion + CRF |
|---|---|---|---|---|
| CASE_NUMBER | 0.66 | 0.73 | 0.77 | **0.77** |
| COURT | 0.90 | 0.92 | 0.92 | **0.92** |
| DATE | 0.87 | 0.88 | 0.89 | **0.89** |
| GPE | 0.76 | 0.75 | 0.78 | **0.78** |
| JUDGE | 0.90 | **0.94** | **0.94** | 0.92 |
| LAWYER | 0.87 | 0.95 | 0.95 | **0.96** |
| ORG | 0.72 | **0.76** | 0.75 | **0.76** |
| OTHER_PERSON | 0.82 | 0.84 | **0.87** | **0.87** |
| PETITIONER | 0.69 | 0.77 | 0.78 | **0.80** |
| PRECEDENT | 0.73 | 0.75 | **0.78** | 0.77 |
| PROVISION | 0.92 | 0.92 | **0.94** | **0.94** |
| RESPONDENT | 0.72 | 0.73 | 0.77 | **0.77** |
| STATUTE | 0.90 | 0.91 | 0.93 | **0.93** |
| WITNESS | 0.86 | 0.90 | **0.93** | 0.92 |

## How to Run

To train and evaluate any of the models:
1. Navigate to the `src/` directory.
2. Select the Jupyter Notebook corresponding to the architecture you wish to run (e.g., `src/gated_fusion/gated_fusion.ipynb`).
3. Upload the notebook to Kaggle, Google Colab, or Lightning.ai.
4. Run all cells. The notebook will automatically download the EkStep dataset, initialize the tokenizer and model, train for 5 epochs, and output the exact classification report shown above.
