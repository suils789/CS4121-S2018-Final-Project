import pandas as pd
import seaborn as sns
import numpy as np
from plotnine import *
from sklearn.decomposition import PCA,TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import issparse

__all__ = ['plot_2d_projection',
           'plot_matrix_heatmap']


def plot_2d_projection(X, y,
                       perplexity=30,
                       method='TSNE',
                       random_state=0):
    """Plot scatter plot of X points projected in a 2d space

    Parameters
    ----------
    X : array like
    y : labels
    perplexity : 5-100
        TSNE parameter
    method : 'PCA' or 'TSNE'
    random_state : 0

    Returns
    -------
    plotnine_obj : plotnine Object
    """
    try:
        X = MinMaxScaler().fit_transform(X)
    except:
        pass

    if issparse(X):
        method = 'PCA'
        X_2d = TruncatedSVD(n_components=2,
                            random_state=random_state).\
            fit_transform(X)
    elif method == 'PCA':
        X_2d = PCA(n_components=2,
                   random_state=random_state) \
            .fit_transform(X)
    elif method == 'TSNE':
        X_2d = TSNE(perplexity=perplexity,
                    init='pca',
                    random_state=random_state) \
            .fit_transform(X)

    df_X = pd.DataFrame({'First Component': X_2d[:, 0],
                         'Second Component': X_2d[:, 1],
                         'label': np.array(y).ravel()})
    if method == 'PCA':
        gg = \
            (ggplot(data=df_X,
                    mapping=aes(x='First Component',
                                y='Second Component',
                                color='label')) +
             geom_point() +
             theme_minimal()+
             theme(dpi=150,
                   figure_size=(7,4.5),
                   legend_position='bottom',
                   axis_title=element_blank())
             )
    elif method == 'TSNE':
        gg = \
            (ggplot(data=df_X,
                    mapping=aes(x='First Component',
                                y='Second Component',
                                color='label')) +
             geom_point() +
             theme_seaborn() +
             labs(
                  x="",
                  y="") +
             theme_minimal() +
             theme(axis_ticks=element_blank(),
                   axis_text=element_blank(),
                   dpi=150,
                   figure_size=(7, 4.5),
                   legend_position='bottom')
             )

    return gg



def plot_matrix_heatmap(X,xlabel=None,ylabel=None,**kwargs):
    """

    Parameters
    ----------
    X : a  matrix
    xlabel,ylabel : array-like, of shape  (n_col,),(n_row,)

    Returns
    -------

    """
    mask = np.zeros_like(X).astype('bool')
    mask[np.triu_indices_from(mask)] = True

    ax = sns.heatmap(X,
                     annot=True,
                     cmap='viridis',
                     square=True,
                     mask=mask,
                     **kwargs)
    if xlabel is not None and ylabel is not None:
        ax = sns.heatmap(X,
                         annot=True,
                         square=True,
                         cmap='viridis',
                         xticklabels = xlabel ,
                         yticklabels = ylabel,
                         mask = mask
                         **kwargs)
    ax.set_title('Heat Map of the Similarity Matrix')
    return ax
