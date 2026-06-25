"""Evaluation metrics for the EHR Copilot test suite.

Functions compute various quality metrics that compare pipeline outputs
against ground truth QA pairs.
"""

from __future__ import annotations


def compute_faithfulness(answer: str, evidence: list[str]) -> float:
    """Estimate how faithful the answer is to the provided evidence.

    Uses a simple word-overlap heuristic:  the fraction of non-stopword
    tokens in the answer that also appear in the concatenated evidence.

    Parameters
    ----------
    answer:
        The copilot answer text.
    evidence:
        List of source chunk texts used to produce the answer.

    Returns
    -------
    float
        A score between 0.0 (no overlap) and 1.0 (perfect overlap).
    """
    if not answer or not evidence:
        return 0.0

    _STOPWORDS = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "shall", "should", "may", "might", "must", "can",
        "could", "of", "in", "to", "for", "with", "on", "at", "from",
        "by", "about", "as", "into", "through", "during", "before",
        "after", "above", "below", "between", "and", "but", "or",
        "not", "no", "so", "if", "then", "than", "that", "this",
        "it", "its", "i", "my", "your", "his", "her", "their",
        "we", "they", "them", "our",
    }

    evidence_text = " ".join(evidence).lower()
    evidence_tokens = set(evidence_text.split()) - _STOPWORDS

    answer_tokens = [
        t for t in answer.lower().split() if t not in _STOPWORDS
    ]

    if not answer_tokens:
        return 0.0

    matched = sum(1 for t in answer_tokens if t in evidence_tokens)
    return matched / len(answer_tokens)


def citation_precision(
    predicted_citations: list[str],
    ground_truth_citations: list[str],
) -> float:
    """Compute citation precision.

    Precision = |predicted intersect ground_truth| / |predicted|

    Parameters
    ----------
    predicted_citations:
        List of citation IDs (chunk_ids or resource_ids) in the answer.
    ground_truth_citations:
        List of citation IDs from the gold standard.

    Returns
    -------
    float
        Precision score between 0.0 and 1.0, or 1.0 if predicted is empty.
    """
    if not predicted_citations:
        return 1.0  # No false positives if nothing was predicted
    pred_set = set(predicted_citations)
    gt_set = set(ground_truth_citations)
    return len(pred_set & gt_set) / len(pred_set)


def citation_recall(
    predicted_citations: list[str],
    ground_truth_citations: list[str],
) -> float:
    """Compute citation recall.

    Recall = |predicted intersect ground_truth| / |ground_truth|

    Parameters
    ----------
    predicted_citations:
        List of citation IDs in the answer.
    ground_truth_citations:
        List of citation IDs from the gold standard.

    Returns
    -------
    float
        Recall score between 0.0 and 1.0, or 1.0 if ground_truth is empty.
    """
    if not ground_truth_citations:
        return 1.0  # Nothing to recall
    pred_set = set(predicted_citations)
    gt_set = set(ground_truth_citations)
    return len(pred_set & gt_set) / len(gt_set)


def abstention_accuracy(
    predictions: list[bool],
    labels: list[bool],
) -> float:
    """Compute abstention accuracy.

    For each question, ``predictions[i]`` is ``True`` if the system
    abstained and ``labels[i]`` is ``True`` if it *should* have
    abstained.  Accuracy is the fraction of matching entries.

    Parameters
    ----------
    predictions:
        List of booleans: did the system abstain?
    labels:
        List of booleans: should the system have abstained?

    Returns
    -------
    float
        Accuracy between 0.0 and 1.0.
    """
    if not predictions:
        return 0.0
    assert len(predictions) == len(labels), "predictions and labels must be the same length"
    correct = sum(1 for p, l in zip(predictions, labels) if p == l)
    return correct / len(predictions)
