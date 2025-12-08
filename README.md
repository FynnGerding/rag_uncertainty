# Uncertainty Estimation in Long-Form RAG

## Installation

To get started, clone the repository, set up a virtual environment, and install the project dependencies:

```bash
git clone git@github.com:FynnGerding/rag_uncertainty.git
cd rag_uncertainty
pip install .
```

The pipeline can be run as follows:

```bash
python src/rag_pipeline/pipeline.py
```

## Novelty: Context-Delta Claim Uncertainty (CDCU)

Our approach draws inspiration from [**Claim-Level Uncertainty (CLUE)**](https://arxiv.org/pdf/2409.03021) and [**Graph-based Uncertainty Metrics**](https://neurips.cc/virtual/2024/poster/94679) but introduces a RAG-aware twist: **Context-Delta Claim Uncertainty (CDCU)**.

Instead of measuring static agreement or centrality, we quantify sensitivity to the retrieved context. We perform leave-one-evidence-out ablations on the top-k snippets (including a no-RAG pass), generating responses for each variation and mapping them to atomic claims. CDCU measures the stability of a claim by tracking the drop in support when specific evidence is removed. We aggregate these deltas, weighted by claim frequency across generations, to determine final certainty and per-snippet influence scores.


## Data Source

The questions in `src/rag_pipeline/questions.json` are taken from **Appendix E.6** of the paper [Long-form factuality in large language models](https://arxiv.org/pdf/2403.18802).

## Code Adaptation: Uncertainty estimation measures

Due to compatibility constraints, we adapted specific logic directly into our codebase rather than importing the packages.
* `atomic_facts.py` is adapted from **FActScore** ([GitHub](https://github.com/shmsw25/FActScore)).
* `uncertainty_estimation.py` integrates methods from **TruthTorchLM** ([GitHub](https://github.com/lexin-zhou/TruthTorchLM)) and **Google DeepMind's Long-form Factuality** ([GitHub](https://github.com/google-deepmind/long-form-factuality)).

## Repo Layout

- `src/rag_pipeline/` – runnable package (entry point `pipeline.py`)
  - `pipeline.py` – orchestrates retrieval-augmented generation, uncertainty scoring, and writes `results.json/csv` (incrementally during a run).
  - `pipeline_utils.py` – wraps HF causal models for sampling and builds retrieval-aware prompts.
  - `retrievers.py` – BM25 and Contriever retrievers implementing a shared interface.
  - `atomic_facts.py` – FActScore-style atomic fact extraction; loads prompts from `demos.json`.
  - `uncertainty_estimation.py` – adapters around TruthTorchLM semantic entropy, sum-eigen, and SAFE factuality scoring.
  - `data.py` – loads Hugging Face datasets, chunks articles, and caches them under `data/*.jsonl`.
  - `questions.json` – grouped evaluation prompts consumed by the pipeline.
- `data/` – cache directory created on demand by `data.py`.
- `third_party/` – vendored dependencies (e.g., TruthTorchLM assets, factscore helpers) imported by the pipeline.
- `results.json` / `results.csv` – outputs generated in the repo root; refreshed after each processed question.