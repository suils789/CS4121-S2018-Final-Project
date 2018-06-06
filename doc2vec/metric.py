from sklearn.metrics import pairwise_distances
import numpy as np


__all__=["similarity_matrix"]

def similarity_matrix(X,distance='cosine'):
    """Calculate the pairwise similarity matrix for each
    row in X

    Parameters
    ----------
    X : array-like (n,k)
        each row is an observation

    distance : str
        either 'cosine' or 'euclidean'

    Returns
    -------
    similarity_matrix : np.ndarray (n,n)
        similarity_matrix[i,j] stores the cosine similarity
        of X[i,:] and X[:,j]
    """
    cosine_distances = pairwise_distances(X,metric=distance)
    return 1-cosine_distances




