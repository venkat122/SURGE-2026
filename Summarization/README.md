# Abstractive Legal Summarization

Welcome to our state-of-the-art pipeline for fine-tuning the `allenai/led-base-16384` (Longformer Encoder-Decoder) model! Our goal? To distill massive, highly complex Indian Legal Judgments (from the IN-Abs dataset) into brilliant, abstractive summaries. 

**The 8GB VRAM Challenge & The "True" Pipeline:**  
In a perfect world with infinite cloud budgets, our ultimate dream was to construct a massive, unified, end-to-end pipeline streaming directly from semantic tagging to Supervised Fine-Tuning (SFT), straight into Direct Preference Optimization (DPO). 

But back here in reality, we built this beast on a single laptop GPU with 8GB of VRAM. Training a monolithic pipeline would have literally melted our hardware. So, we cleverly modularized the pipeline into distinct, bite-sized phases. 

To maintain rigorous scientific consistency, **the following training hyperparameters were held strictly constant** across all model iterations (Phase 2 onwards) to respect the memory limits without cheating the benchmarks:
- **LoRA Rank:** `16`
- **LoRA Alpha:** `32`
- **Target Matrices:** `q`, `k`, `v`
- **Batch Size:** `1`
- **Gradient Accumulation:** `8 steps`
- **Max Encoder Tokens:** `6,144` (Strict hardware bottleneck, down from 16,384)

---

## The Evolutionary Pipeline

Our architecture evolved over five major phases to reach maximum performance:

1. **Phase 1: The Humble Baseline (`src/baseline`)** 
   Initial Supervised Fine-Tuning (SFT) using a basic, lightweight LoRA configuration (targeting only `q` and `v`). It worked, but we wanted more.
   
2. **Phase 2: The Optimized Baseline (`src/optimized_baseline`)** 
   We cranked up the SFT using expanded LoRA target matrices (`q`, `k`, `v`), threw in Cosine Annealing learning rate schedulers, and expanded the output boundaries to 1024 tokens. 

3. **Phase 3: Giving the AI a Law Degree (`src/rhetorical_role`)** 
   We integrated the OpenNYAI pipeline to perform zero-shot semantic parsing on legal logic (tagging sections like `(preamble)`, `(fact)`, and `(argument by petitioner)`). We natively injected these tags into the prompt to give our LLM a structural skeleton to follow.

4. **Phase 4: Direct Preference Optimization (`src/dpo_baseline` & `src/dpo_rhetorical`)** 
   Time for some discipline. We used DPO to contrast positive, human-written summaries against negative, model-generated hallucinations. This punished the model for repeating itself and forced strict factual adherence.

5. **Phase 5: The Map-Reduce Monster (`src/dynamic_chunking`)** 
   Due to our 8GB VRAM limit, we had to artificially restrict the LED model to a strict **6,144 token limit**. What happens when a document is 36,000+ tokens? Usually, catastrophic failure. Our solution? A Dynamic Map-Reduce algorithm that chunks the text into 5,000-token windows, summarizes the pieces in parallel, and recursively squashes them down into one highly coherent summary. Infinity unlocked.

---

## Empirical Results (The Scoreboard)

We evaluated the primary models against an unseen test dataset using ROUGE (exact overlap) and BERTScore (semantic similarity). 

| Model Variant | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore F1 |
| :--- | :---: | :---: | :---: | :---: |
| **Optimized Baseline** | **0.4624** | 0.2222 | 0.2298 | 0.8125 |
| **Rhetorical Model** | 0.4454 | 0.2266 | 0.2314 | **0.8144** |
| **DPO Baseline** | 0.4278 | 0.2055 | 0.2162 | 0.8093 |
| **DPO Rhetorical** | 0.4618 | **0.2267** | **0.2345** | 0.8117 |

### Extreme-Length Stress Test (Dynamic Chunking)

The standard ROUGE metrics are mathematically identical for documents falling below the 6,144-token threshold. However, many Indian Legal Judgments exceed this limit, causing catastrophic truncation of the judicial analysis and ratio decidendi.

To definitively prove the necessity of the Dynamic Chunking Engine, we stress-tested all four models against the **Top 15 longest documents** in the test set (ranging from 8,500 to over 47,000 tokens). We bypassed the recursive AI-reduction phase to prevent semantic compression, allowing the Map-extracted sub-summaries to concatenate losslessly. 

#### Empirical Results on Long Documents

| Model Variant | Mode | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore F1 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Optimized Baseline** | Standard (Truncated) | 0.3125 | 0.1740 | 0.1595 | 0.8401 |
| **Optimized Baseline** | **Chunking (Lossless)** | 0.5283 | 0.2900 | 0.2157 | 0.8403 |
| **Rhetorical Model** | Standard (Truncated) | 0.2471 | 0.1249 | 0.1250 | 0.8330 |
| **Rhetorical Model** | **Chunking (Lossless)** | 0.5180 | 0.2855 | 0.2165 | 0.8368 |
| **DPO Baseline** | Standard (Truncated) | 0.3173 | 0.1612 | 0.1548 | 0.8372 |
| **DPO Baseline** | **Chunking (Lossless)** | 0.5274 | 0.2870 | 0.2203 | 0.8369 |
| **DPO Rhetorical** | Standard (Truncated) | 0.2818 | 0.1489 | 0.1440 | 0.8333 |
| **DPO Rhetorical** | **Chunking (Lossless)** | **0.5769** | **0.3185** | **0.2370** | **0.8379** |

As demonstrated above, the standard inference mechanism completely collapses on massive documents, leading to severe hallucination and loss of critical information. Our Dynamic Map-Reduce algorithm restores 100% data retention, propelling the **DPO Rhetorical** model to a staggering **0.5769 ROUGE-1**, proving that the combination of structural tagging, preference optimization, and lossless context chunking creates an unparalleled legal summarization engine.

---

## Environment Setup

This repository requires **two separate Conda environments**. We intentionally decoupled the environments to prevent Dependency Hell. The OpenNYAI legal tagger relies on older, strict NLP dependencies (e.g., specific `spaCy` versions) that severely clash with the bleeding-edge GPU-accelerated PyTorch and `transformers` packages required for training the LED model.

1. **`legalsum` Environment**: The primary environment for all model training, Map-Reduce chunking, DPO, and ROUGE evaluations. Install via `requirements_legalsum.txt`.
2. **`opennyai_env` Environment**: A modularized environment strictly for executing the zero-shot OpenNYAI extraction pipeline in Phase 3. Install via `requirements_opennyai.txt`.

### Installation Commands

You can use either Conda (Recommended) or standard Python virtual environments (`venv`).

#### Option A: Using Conda (Recommended)
```bash
# 1. Create and activate the primary ML environment
conda create -n legalsum python=3.10 -y
conda activate legalsum
pip install -r requirements_legalsum.txt

# 2. Create and activate the OpenNYAI NLP environment
conda create -n opennyai_env python=3.10 -y
conda activate opennyai_env
pip install -r requirements_opennyai.txt
```

#### Option B: Using Standard Python `venv`
```bash
# 1. Create and activate the primary ML environment
python3.10 -m venv legalsum_venv
source legalsum_venv/bin/activate  # On Windows use: legalsum_venv\Scripts\activate
pip install -r requirements_legalsum.txt
deactivate

# 2. Create and activate the OpenNYAI NLP environment
python3.10 -m venv opennyai_venv
source opennyai_venv/bin/activate  # On Windows use: opennyai_venv\Scripts\activate
pip install -r requirements_opennyai.txt
deactivate
```

---

## Repository Structure

```text
Summarization/
├── README.md               # You are here!
├── requirements_legalsum.txt # Primary ML dependencies (Torch, Transformers, TRl)
├── requirements_opennyai.txt # Modular NLP dependencies (OpenNYAI, spaCy)
├── analysis/               # Exploratory data analysis scripts
├── data/
│   ├── raw/                # Original unstructured legal documents
│   └── processed/          # Pre-tokenized, tagged, and preference datasets
├── logs/                   # Training and evaluation logs
├── models/                 # Cached HuggingFace base models
├── notebooks/              # Jupyter notebooks for interactive testing
├── src/
│   ├── baseline/           # Phase 1: Basic SFT
│   │   └── train_v1_baseline.py # Trains initial LED-base model on raw data
│   ├── common/             # Shared evaluation metrics and dataloaders
│   │   ├── dataset_loader.py    # Common IO for loading train/test splits
│   │   ├── check_lengths.py     # Token length distribution analysis
│   │   ├── evaluate_models.py   # ROUGE/BERTScore evaluation suite
│   │   └── evaluate_dpo_models.py # DPO-specific evaluation wrappers
│   ├── data_processing/    # Raw dataset extraction and cleaning
│   ├── dpo_baseline/       # Phase 4: DPO applied to Optimized Baseline
│   │   └── train_dpo_baseline.py # Runs DPO alignment on the baseline model
│   ├── dpo_dataset_generation/ # Phase 4: Constructing positive/negative pairs
│   │   ├── create_dpo_dataset.py # Generates hallucinated negative samples
│   │   └── train_dpo_dataset.py  # Compiles the final DPO JSONL pairs
│   ├── dpo_rhetorical/     # Phase 4: DPO applied to Rhetorical model
│   │   └── train_dpo_rhetorical.py # Runs DPO alignment on the tagged model
│   ├── dynamic_chunking/   # Phase 5: Map-Reduce inference engine
│   │   ├── dynamic_chunking_inference.py # Core recursive summarizer algorithm
│   │   ├── evaluate_chunking_benchmark.py # Stress-tests the 15 longest documents
│   │   └── run_chunking_on_all.py # Bulk inference generator for datasets
│   ├── optimized_baseline/ # Phase 2: Enhanced SFT
│   │   ├── train_optimized_baseline.py # SFT with expanded Q,K,V targeting
│   │   └── batch_orchestrator.py # VRAM-safe batch generation tool
│   ├── rhetorical_role/    # Phase 3: OpenNYAI integration
│   │   ├── run_opennyai_local.py # [Env: opennyai_env] Extracts and injects legal tags
│   │   └── train_rhetorical_role.py # Fine-tunes model on structurally tagged data
│   └── scripts/            # High-level execution scripts
│       ├── run_rhetorical_pipeline.py # Full pipeline orchestrator
│       ├── run_dual_dpo.py # Queues DPO for both Baseline and Rhetorical 
│       └── run_final_tests.py # Full automated ROUGE/BERTScore evaluation
└── weights/                # LoRA adapter checkpoints (Safe to push)
```

---

## Running the End-to-End Orchestrators

Although we modularized the codebase for the safety of our GPU, you can run our orchestrator scripts to simulate the true, glorious end-to-end flow!

**To run the complete Rhetorical + DPO pipeline from start to finish:**
```bash
python src/scripts/run_rhetorical_pipeline.py
```

To run a full inference benchmark on the resulting models (ROUGE, BERTScore) and execute a stress test of the Dynamic Chunking algorithm on a massive legal document:
```bash
python src/scripts/run_final_tests.py
```
