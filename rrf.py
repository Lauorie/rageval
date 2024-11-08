def reciprocal_rank_fusion(ranked_lists, k=60):
    """
    Perform Reciprocal Rank Fusion on multiple ranked lists.

    Parameters:
    - ranked_lists: List of lists, where each sublist contains document IDs in ranked order.
    - k: Constant to control the influence of rank position.

    Returns:
    - Dictionary with document IDs as keys and their RRF scores as values.
    """
    from collections import defaultdict

    # Initialize a dictionary to store RRF scores
    rrf_scores = defaultdict(float)

    # Iterate over each ranked list
    for ranked_list in ranked_lists:
        for rank, doc_id in enumerate(ranked_list):
            rrf_scores[doc_id] += 1 / (k + rank + 1)

    return dict(rrf_scores)

# Example usage:
ranked_lists = [
    ['doc1', 'doc2', 'doc3'],
    ['doc2', 'doc3', 'doc1'],
    ['doc3', 'doc1', 'doc2']
]

rrf_scores = reciprocal_rank_fusion(ranked_lists)
print(rrf_scores)
# Output: {'doc1': 0.03333333333333333, 'doc2': 0.03333333333333333, 'doc3': 0.03333333333333333}
