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

### Eemantic Entropy

Add method summary here...

### SumEigen

Add method summary here...

## Long-Form Factuality: SAFE

Our code is addapted from **Google DeepMind's Long-form Factuality** ([GitHub](https://github.com/google-deepmind/long-form-factuality)), making SAFE usable within a RAG context.

## Data Source

The questions in `src/rag_uncertainty/questions.json` are taken from **Appendix E.6** of the paper [Long-form factuality in large language models](https://arxiv.org/pdf/2403.18802).

## Repo Layout

Add later...