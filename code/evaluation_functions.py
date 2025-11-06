import pandas as pd
import numpy as np

def evaluate_recommendations(df, ks):
    """
    Expects args:
        df: pandas df with columns ['user', 'movie', 'edge_score', 'ground_truth']
        ks (list of ints): Numbers of top recommendations to consider

    Returns NDCG@k, Recall@k, MRR@k for the given k
    """
    # Create dict of results
    results = {}
    # Group by user
    users = df['user'].unique()
    for k in ks:
        ndcg_list = []
        recall_list = []
        rr_list = []
    
        for user in users:
            user_df = df[df['user'] == user]
            # Sort by predicted score
            ranked = user_df.sort_values('edge_score', ascending=False)
            # Top k predictions
            topk = ranked.head(k)
            # Ground truth relevance values
            rel = topk['ground_truth'].values
            # Compute DCG@k
            gains = (2**rel - 1)
            discounts = np.log2(np.arange(2, k + 2))
            dcg = np.sum(gains / discounts)
    
            # Compute IDCG@k (ideal ranking)
            ideal_rel = np.sort(user_df['ground_truth'].values)[::-1][:k]
            ideal_gains = (2**ideal_rel - 1)
            idcg = np.sum(ideal_gains / discounts)
            ndcg = dcg / idcg if idcg > 0 else 0.0
            ndcg_list.append(ndcg)
    
            # Recall@k: number relevant in topk / total relevant
            total_rel = user_df['ground_truth'].sum()
            recall = rel.sum() / total_rel if total_rel > 0 else 0.0
            recall_list.append(recall)
    
            # MRR@k: reciprocal of rank of the top-ranked true edge
            rr = 0.0
            for idx, val in enumerate(rel, start=1):
                if val == 1:
                    rr = 1.0 / idx
                    break
            rr_list.append(rr)
    
        dict_of_results = {
            f'NDCG@{k}': np.mean(ndcg_list),
            f'Recall@{k}': np.mean(recall_list),
            f'MRR@{k}': np.mean(rr_list)
        }

        # Round results
        dict_of_results = {key: round(value, 4) for key, value in dict_of_results.items()}

        # Store results in the main dict
        results.update(dict_of_results)

    return results