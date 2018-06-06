import numpy as np
import pandas as pd


def find_topk(similarity_matrix,k=1):
    """Given a similarity matrix m,
    with m[i,j] stores the similarity score
    of item i with item j, output the top k most
    similar items for each item

    Parameters
    ----------
    similarity_matrix : np.ndarray
        a symmetric matrix.
        m[i,j] stores the similarity score of item i with item j

    k : int
        number of most similar items to return

    Returns
    -------
    top_k : np.ndarray
        shape = (n_rows,k). top_k[i,:] gives the k most similar
        items to item i

    similarity_score : np.ndarray
        shape = (n_rows,k), similarity_scores[i,:] gives the k
        similarity scores for the k most similar items to item i
    """
    row_indices = np.arange(len(similarity_matrix))[...,np.newaxis]
    top_k = np.argsort(similarity_matrix)[...,::-1][...,1:(k+1)]
    similarity_score = similarity_matrix[row_indices,top_k]
    return  top_k,similarity_score

