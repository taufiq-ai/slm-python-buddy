# metrics.py
"""DO NOT USE: NOT FUNCTIONAL YET"""

import torch
import math
import structlog
from transformers import TrainerCallback, Trainer
from typing import Optional, Callable, List

logger = structlog.get_logger(__name__)


def compute_perplexity(eval_loss: float) -> float:
    """
    Compute perplexity from cross-entropy loss.
    
    Args:
        eval_loss (float): Cross-entropy loss from evaluation.
    
    Returns:
        float: Perplexity (exp(loss))
    """
    return math.exp(eval_loss) if eval_loss < 100 else float("inf")



def get_compute_metrics(custom_metric_fn: Optional[Callable] = None):
    """
    Returns a compute_metrics function compatible with HuggingFace Trainer.
    
    Args:
        custom_metric_fn (Callable, optional): Function that takes (predictions, labels)
            and returns a dict of additional metrics, e.g., pass@k, BLEU, BERTScore, etc.
    
    Returns:
        function: compute_metrics(pred) -> dict
    """
    def compute_metrics(eval_pred):
        # eval_pred: EvalPrediction object from HF Trainer
        logits, labels = eval_pred.predictions, eval_pred.label_ids
        # Convert labels to float for loss-based perplexity calculation
        if labels is not None:
            # HuggingFace usually already computes loss, but just in case:
            mask = labels != -100  # ignore padding
            # Optional: can calculate CE loss here if needed
            # But Trainer automatically computes eval_loss
        metrics = {}

        # Include custom metrics if provided
        if custom_metric_fn:
            metrics.update(custom_metric_fn(eval_pred))

        # Include perplexity if eval_loss exists
        if hasattr(eval_pred, "metrics") and "eval_loss" in eval_pred.metrics:
            metrics["perplexity"] = compute_perplexity(eval_pred.metrics["eval_loss"])

        return metrics

    return compute_metrics



class LogPerplexityCallback(TrainerCallback):
    """
    Trainer callback to log perplexity at each evaluation.
    Useful for code generation tasks.
    """
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics and "eval_loss" in metrics:
            ppl = compute_perplexity(metrics["eval_loss"])
            metrics["perplexity"] = ppl
            logger.info("Evaluation metrics", **metrics)
        return control



def pass_at_k_metric(predictions: List[str], references: List[str], k: int = 5):
    """
    Example placeholder for pass@k evaluation for code generation tasks.
    In practice, implement test-case execution or correctness checker.
    
    Args:
        predictions (List[str]): List of generated code strings.
        references (List[str]): List of reference code strings.
        k (int): Number of top generations to check.
    
    Returns:
        dict: {"pass@k": float} metric
    """
    # TODO: implement real test-case execution
    correct = 0
    total = len(predictions)
    for pred, ref in zip(predictions, references):
        # Placeholder: exact match (replace with unit test)
        if pred.strip() == ref.strip():
            correct += 1
    return {"pass@k": correct / total if total > 0 else 0.0}
