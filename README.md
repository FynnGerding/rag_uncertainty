# Uncertainty Estimation in Long-Form RAG

## Installation

To get started, clone the repository, set up a virtual environment, and install the project dependencies:

```bash
git clone git@github.com:FynnGerding/rag_uncertainty.git
cd rag_uncertainty
make setup
```

The pipeline can be run as follows:

```bash
make run
```

## Novelty: Claim-Evidence Matrix (CEM)
Building on [**Claim-Level Uncertainty (CLUE)**](https://arxiv.org/pdf/2409.03021) and [**Graph-based Uncertainty Metrics**](https://neurips.cc/virtual/2024/poster/94679), CEM introduces a RAG-specific framework to quantify epistemic fragility. It replaces expensive stochastic consistency checks with a single-pass assessment of evidence grounding.

#### Key Advantages
- Grounding over Consistency: Verifies if claims are actively entailed by retrieved documents, independent of model confidence.
- Efficiency: Reduces compute by requiring only one generation pass ($1$ vs $N$) via parallelizable support checks.
- Retrieval-Awareness: Distinguishes between robustly supported claims and those hinging on sparse evidence.
#### Formulation
We decompose response $R$ into atomic claims $C$ and construct a binary matrix $M$ where $M_{ij}=1$ if document $d_j$ supports claim $c_i$. Uncertainty is defined as the inverse of evidence redundancy:

$$U(c_i) = 1 - \frac{1}{k} \sum_{j=1}^k M_{ij}$$

Here, $U \approx 0$ implies robust support across the context window, while high $U$ indicates scarce support for a claim.

## Uncertainty Estimation Methods

Our implementatoins are based on **TruthTorchLM** ([GitHub](https://github.com/lexin-zhou/TruthTorchLM)).

### Semantic Entropy (White-box)

**Paper:** Semantic Uncertainty [(arXiv)](https://arxiv.org/abs/2302.09664)

This method addresses the issue where standard entropy metrics overestimate uncertainty by treating semantically equivalent sentences (e.g., "Paris is the capital" vs. "The capital is Paris") as distinct outcomes.
* **Mechanism:** It generates multiple answers and clusters them by meaning using a bidirectional entailment check (NLI).
* **Metric:** It sums the probabilities of sequences within each cluster to compute the entropy over **meanings** rather than tokens, requiring access to model logits.

### SumEigen (Black-box)

**Paper:** Generating with Confidence [(arXiv)](https://arxiv.org/abs/2305.19187)

Designed for closed-source models where token logits are unavailable, this approach quantifies the "semantic dispersion" of generated responses.
* **Mechanism:** It generates multiple responses and computes a pairwise similarity matrix (using NLI entailment scores) to construct a graph Laplacian.
* **Metric:** The uncertainty score ($U_{EigV}$) is the sum of the Laplacian's eigenvalues, which serves as a proxy for the number of distinct semantic clusters in the output.

## Long-Form Factuality in RAG: RAFE (vs SAFE)

We additionally propose RAFE (Retrieval-Augmented Factuality Estimation). Adapting Google DeepMind's [SAFE (Search-Augmented Factuality Estimation)](https://arxiv.org/pdf/2403.18802), we replace external Google Search steps with local context verification. Our code is addapted from **Google DeepMind's Long-form Factuality** ([GitHub](https://github.com/google-deepmind/long-form-factuality)) to measure factuality strictly against retrieved evidence (RAG) rather than open web search.

## Data Source

The questions in `src/rag_uncertainty/questions.json` are taken from **Appendix E.6** of the paper [Long-form factuality in large language models](https://arxiv.org/pdf/2403.18802).

## Repo Layout

Add later...