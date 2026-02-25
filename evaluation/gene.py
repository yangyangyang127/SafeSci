import numpy as np
from Bio.Align import PairwiseAligner

def levenshtein_distance(s, t):
    
    m, n = len(s), len(t)
    if m == 0:
        return n
    if n == 0:
        return m
    
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if s[i - 1] == t[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,        # deletion
                dp[i][j - 1] + 1,        # insertion
                dp[i - 1][j - 1] + cost  # substitution
            )
    return dp[m][n]


def eval_gene(gt_list, pred_list):
    
    if len(pred_list) != len(gt_list):
        raise ValueError("Unmatched length: pred_list and gt_list.")
    
    N = len(pred_list)
    if N == 0:
        return None
    
    aligner = PairwiseAligner()
    aligner.mode = 'global'
    aligner.match_score = 1.0
    aligner.mismatch_score = -1.0
    aligner.open_gap_score = -2.0
    aligner.extend_gap_score = -1.0
    
    identities = []
    avg_coverages = []
    norm_edit_dists = []
    similarity_scores = []
    
    valid_nucleotides = set('ACGTUN')
    
    for i in range(N):
        pred = str(pred_list[i]).upper().strip()
        gt = str(gt_list[i]).upper().strip()
                
        if not pred or not gt:
            identities.append(0.0)
            avg_coverages.append(0.0)
            norm_edit_dists.append(1.0)
            similarity_scores.append(0.0)
            continue
        
        alignments = aligner.align(pred, gt)
        try:
            alignment = next(alignments)
        except StopIteration:
            identities.append(0.0)
            avg_coverages.append(0.0)
            norm_edit_dists.append(1.0)
            similarity_scores.append(0.0)
            continue
        
        aligned_pred = alignment[0]
        aligned_gt = alignment[1]
        
        aligned_length = len(aligned_pred)
        matches = sum(a == b for a, b in zip(aligned_pred, aligned_gt))
        identity = (matches / aligned_length) if aligned_length > 0 else 0.0
        
        pred_covered = sum(a != '-' for a in aligned_pred) / len(pred)
        gt_covered = sum(b != '-' for b in aligned_gt) / len(gt)
        avg_coverage = (pred_covered + gt_covered) / 2
        
        edit_dist = levenshtein_distance(pred, gt)
        max_len = max(len(pred), len(gt))
        norm_edit = edit_dist / max_len if max_len > 0 else 0.0
        
        # Sequence Similarity Score
        raw_score = alignment.score
        max_possible_score = aligner.match_score * aligned_length
        similarity_score = (raw_score / max_possible_score) if max_possible_score > 0 else 0.0
        
        identities.append(identity)
        avg_coverages.append(avg_coverage)
        norm_edit_dists.append(norm_edit)
        similarity_scores.append(similarity_score)
        
    return {
        'Identity': np.mean(identities),
        'Coverage': np.mean(avg_coverages),
        'Levenshtein': np.mean(norm_edit_dists),
        'Similarity': np.mean(similarity_scores), 
    }
