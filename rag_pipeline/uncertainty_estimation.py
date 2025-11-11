from TruthTorchLM.truth_methods.semantic_entropy import SemanticEntropy
from TruthTorchLM.truth_methods.sum_eigen_uncertainty import SumEigenUncertainty


def semantic_entropy(generations, question, **kwargs):
    """
    Compute Semantic Entropy from externally provided generations.

    Parameters
    ----------
    generations : dict
        Must include:
          - "generated_texts": List[str]
          - "logprobs": List[List[float]]  # token-level logprobs per generation
        (Any extra fields are ignored.)
    question : str
        The original question/prompt (used for entailment clustering).
    **kwargs :
        Passed directly to SemanticEntropy(...), e.g.:
          scoring_function, number_of_generations,
          model_for_entailment, tokenizer_for_entailment,
          entailment_model_device, batch_generation.

    Returns
    -------
    dict
        Matches SemanticEntropy.forward_api(...) output:
          {
            "truth_value": float,
            "semantic_entropy": float,
            "score_for_each_generation": List[float],
            "generated_texts": List[str],
            "clusters": List[Set[str]],
          }
    """
    if "generated_texts" not in generations:
        raise ValueError('`generations` must contain key "generated_texts".')
    if "logprobs" not in generations:
        raise ValueError('`generations` must contain key "logprobs" for semantic entropy.')

    kwargs.setdefault("entailment_model_device", "cpu")
    se = SemanticEntropy(**kwargs)
    # Because sampling is external, we only pass the precomputed dict.
    return se.forward_api(
        model="",
        messages=[],
        generated_text="",
        question=question,
        sampled_generations_dict=generations,
    )


def sum_eigen(generations, question, **kwargs):
    """
    Compute Sum Eigen Uncertainty (U_eigv) from externally provided generations.

    Parameters
    ----------
    generations : dict
        Must include:
          - "generated_texts": List[str]
        (Any extra fields are ignored.)
    question : str
        The original question/prompt (used for similarity).
    **kwargs :
        Passed directly to SumEigenUncertainty(...), e.g.:
          method_for_similarity ("semantic" | "jaccard"),
          number_of_generations, temperature,
          model_for_entailment, tokenizer_for_entailment,
          entailment_model_device, batch_generation.

    Returns
    -------
    dict
        Matches SumEigenUncertainty.forward_api(...) output:
          {
            "U_eigv": float,
            "generated_texts": List[str],
            "truth_value": float,
          }
    """
    if "generated_texts" not in generations:
        raise ValueError('`generations` must contain key "generated_texts".')

    kwargs.setdefault("entailment_model_device", "cpu")
    seu = SumEigenUncertainty(**kwargs)
    # Because sampling is external, we only pass the precomputed dict.
    return seu.forward_api(
        model="",
        messages=[],
        generated_text="",
        question=question,
        sampled_generations_dict=generations,
    )
