import random
import numpy as np
from collections import Counter
from typing import List, Tuple, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns

def eval_tf(
    ground_truth: List[str],
    predictions: List[str],
    options: str = "ABCD", 
    case_sensitive: bool = False,
    num_experiments: int = 1
) -> Dict[str, Any]:
    
    assert len(ground_truth) == len(predictions), \
        f"Unmatched length: ground_truth({len(ground_truth)}) != predictions({len(predictions)})"
    
    if not case_sensitive:
        ground_truth = [str(gt).upper().strip() for gt in ground_truth]
        predictions = [str(pred).upper().strip() for pred in predictions]
    else:
        ground_truth = [str(gt).strip() for gt in ground_truth]
        predictions = [str(pred).strip() for pred in predictions]
    
    n_samples = len(ground_truth)
    
    if num_experiments == 1:
        correct = []
        for gt, pred in zip(ground_truth, predictions):
            correct.append(gt == pred)
        
        accuracy = np.mean(correct)
        return accuracy * 100.
    
    results = []
    for exp_idx in range(num_experiments):

        indices = random.sample(range(n_samples), int(n_samples*0.34))
        gt_sample = [ground_truth[i] for i in indices]
        pred_sample = [predictions[i] for i in indices]

        score = []
        for gt, pred in zip(gt_sample, pred_sample):
            score.append(gt == pred)
        
        score = np.mean(score)
        results.append(score)

    mean_score = np.mean(results)
    std_score = np.std(results, ddof=1)

    return mean_score, std_score















