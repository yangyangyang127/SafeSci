import numpy as np
import random
from collections import Counter
from typing import List, Tuple, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns

def eval_mcq(
    predictions: List[str],
    ground_truth: List[str],
    options: str = "ABCD",
    case_sensitive: bool = False,
    num_experiments: int = 1
) -> Dict[str, Any]:
    
    def cal_acc(gt_list, pred_list):
        
        correct = []
        for gt, pred in zip(gt_list, pred_list):
            
            if len(gt) > 1 or len(pred) > 1:
                gt_set = set(gt)
                pred_set = set(pred)
                # correct.append(gt_set == pred_set)
                correct.append(pred_set.issubset(gt_set))
            else:
                correct.append(gt == pred)
        
        # print(correct)
        accuracy = np.mean(correct)
        return accuracy 

    assert len(ground_truth) == len(predictions), \
        f"长度不匹配: ground_truth({len(ground_truth)}) != predictions({len(predictions)})"
    
    if num_experiments == 1:
        accuracy = cal_acc(gt_sample, pred_sample)
        return accuracy * 100.
    
    n_samples = len(ground_truth)
    results = []
    for exp_idx in range(num_experiments):
        indices = random.sample(range(n_samples), int(n_samples*0.34))
        
        gt_sample = [ground_truth[i] for i in indices]
        pred_sample = [predictions[i] for i in indices]
        
        score = cal_acc(gt_sample, pred_sample)
        results.append(score)
    
    mean_score = np.mean(results)
    std_score = np.std(results, ddof=1)

    return mean_score, std_score
    