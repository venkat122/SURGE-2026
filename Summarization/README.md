# Abstractive Legal Summarization (SURGE)

This subdirectory contains the codebase for fine-tuning the `allenai/led-base-16384` (Longformer Encoder-Decoder) model to perform abstractive summarization on extremely long Indian Legal Judgments (IN-Abs dataset). 

The pipeline is heavily optimized to run on constrained hardware (8GB VRAM) via LoRA, Gradient Checkpointing, and FP16 Mixed Precision.

## Dataset
We utilize the **IN-Abs** dataset, which consists of approximately 7,130 Indian Legal Judgments. These documents are highly unstructured and frequently exceed 8,192 tokens in length, requiring specialized sparse-attention transformers to process.

## Repository Structure

```text
Summarization/
├── data/
│   ├── raw/                 # The original unstructured legal documents (ignored by git)
│   └── processed/           # Processed datasets (e.g., tagged with rhetorical roles) (ignored by git)
├── src/
│   ├── data_processing/     # Scripts for loading, verifying, and modifying the dataset
│   └── training/            # PyTorch / HuggingFace training loops
├── logs/                    # Training output logs
├── weights/                 # Checkpointed LoRA adapter weights (ignored by git)
└── analysis/                # Statistical insights on dataset token lengths
```

## Running the Pipeline

### Phase 1: Baseline LoRA Training
To run the baseline training (r=8, targeting Query and Value matrices):
```bash
python src/training/train_v1_baseline.py
```

### Phase 2: Tweaked LoRA (Expanded Matrices & Schedulers)
To run the more advanced version (r=32, cosine decay, all attention matrices, 1024-token output generation):
```bash
python src/training/train_v2_tweaked.py
```

### Phase 3: Natural Language Rhetorical Roles
To implement zero-shot semantic parsing of legal logic without expanding token embeddings, first tag the dataset:
```bash
python src/data_processing/v3_tag_rhetorical_roles.py
```
This generates a new dataset in `data/processed/IN-Abs_V3/`. Then, train the model to understand the natural language tags:
```bash
python src/training/train_v3_rhetorical.py
```
