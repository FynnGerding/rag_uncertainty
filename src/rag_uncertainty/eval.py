import io
import contextlib
import threading
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, logging as hf_logging
from TruthTorchLM.truth_methods.semantic_entropy import SemanticEntropy
from TruthTorchLM.truth_methods.sum_eigen_uncertainty import SumEigenUncertainty

from rag_uncertainty.atomic_facts import AtomicFactGenerator
from rag_uncertainty.retrievers import BM25Retriever
from rag_uncertainty.rafe import rafe_factuality
from rag_uncertainty.pipeline_utils import LLM

hf_logging.set_verbosity_error()

class RafeWrapper:
    """
    Wrapper to make rafe_factuality compatible with the EvalEngine interface.
    """
    def __init__(self, llm: LLM, retriever: BM25Retriever, fact_gen: AtomicFactGenerator):
        self.llm = llm
        self.retriever = retriever
        self.fact_gen = fact_gen

    def forward_api(self, question, sampled_generations_dict, **kwargs):
        return rafe_factuality(
            generations=sampled_generations_dict,
            question=question,
            llm=self.llm,
            retriever=self.retriever,
            fact_gen=self.fact_gen,
            top_k=5,
            per_generation=True
        )

class EvalEngine:
    """
    Singleton engine for evaluating text generation uncertainty and truthfulness.
    Loads shared models once and provides a thread-safe interface for multiple metrics.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, llm=None, retriever=None, fact_gen=None):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(EvalEngine, cls).__new__(cls)
                cls._instance._initialize(llm, retriever, fact_gen)
        return cls._instance

    def _initialize(self, llm, retriever, fact_gen):
        """
        Loads shared resources and initializes metric calculators.
        """
        self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        
        model_name = "microsoft/deberta-large-mnli"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()

        self.metrics = {}
        
        with contextlib.redirect_stdout(io.StringIO()):
            self._register_metric(
                "semantic_entropy", 
                SemanticEntropy(
                    model_for_entailment=self.model,
                    tokenizer_for_entailment=self.tokenizer,
                    entailment_model_device=self.device
                )
            )
            
            self._register_metric(
                "sum_eigen", 
                SumEigenUncertainty(
                    model_for_entailment=self.model,
                    tokenizer_for_entailment=self.tokenizer,
                    entailment_model_device=self.device,
                    method_for_similarity="semantic"
                )
            )

            # Register RAFE only if dependencies are provided
            if llm and retriever and fact_gen:
                self._register_metric(
                    "rafe",
                    RafeWrapper(llm, retriever, fact_gen)
                )

    def _register_metric(self, name, calculator_instance):
        self.metrics[name] = calculator_instance

    def evaluate(self, generations, question, metric_names=None):
        """
        Runs specified evaluation metrics on the provided generations.

        Args:
            generations (dict): Dictionary containing "generated_texts" and "logprobs".
            question (str): The prompt or question used to generate texts.
            metric_names (list[str], optional): List of metrics to run. Defaults to all.

        Returns:
            dict: A dictionary mapping metric names to their result objects.
        """
        if metric_names is None:
            metric_names = list(self.metrics.keys())

        results = {}
        
        with self._lock:
            for name in metric_names:
                if name not in self.metrics:
                    continue
                
                calculator = self.metrics[name]
                results[name] = calculator.forward_api(
                    model="", 
                    messages=[], 
                    generated_text="",
                    question=question, 
                    sampled_generations_dict=generations
                )
                
        return results