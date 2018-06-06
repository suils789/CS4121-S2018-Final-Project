import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from . import metric
from .plot import plot_2d_projection
from .doc2vec import doc2vec
from .utils import find_topk


class DocEmbedder(BaseEstimator, TransformerMixin):
    """
    Parameters
    --------

    Methods
    --------
    fit

    fit_transform

    report_topk_docs

    scatter_plot

    Attributes
    --------
    similarity_matrix : np.ndarray
        similarity_matrix[i,j] stores the cosine similarity
        of doc i X[i,:] and doc j X[:,j]
    """

    def __init__(self):
        self._is_fitted = False
        self._similarity_matrix = None

    def fit(self, docs):
        """fit the document embedder on docs

        Parameters
        ----------
        docs : array-like
            an array of documents

        Returns
        -------
        self : self
        """
        docvecs = doc2vec(docs)
        self.docvecs_ = docvecs
        self.docs = docs
        self._is_fitted = True
        self._similarity_matrix = None
        return self

    def fit_transform(self, docs):
        """ fit on docs and transform them to vectors

        Parameters
        ----------
        docs :  array-like
            an array of documents

        Returns
        -------
        x : np.ndarray
            embedded documents. Shape = (n_docs, embedd_dim)
        """
        self.fit(docs)
        return self.docvecs_

    @property
    def similarity_matrix(self):
        assert self._is_fitted, "Not fitted"
        if hasattr(self, '_similarity_matrix') and \
                self._similarity_matrix is not None:
            return self._similarity_matrix
        x = metric.similarity_matrix(self.docvecs_)
        self.similarity_matrix = x
        return x

    @similarity_matrix.setter
    def similarity_matrix(self, value):
        self._similarity_matrix = value

    def report_topk_docs(self, document_topic=None,
                         document_title=None,
                         k=1,
                         threshold=0.95):
        """Report top k most similar docs for each doc in corpus

        Parameters
        ----------
        document_topic : array-like
            shape = (n_docs,) topics for the docs

        document_title : array-like
            shape (n_docs,). titles for the docs

        k : int
            upper limit for number of most similar documents
            to be reported for each doc

        threshold : float
            only when similarity score >threshold
            will document pairs be reported
        Returns
        -------
        None
        """

        topk, sim_scores = find_topk(self.similarity_matrix, k=k)
        print("Finding topic-similar articles by document semantic meanings")
        for i, (tops, scores) in enumerate(zip(topk, sim_scores)):
            if not np.any(scores > threshold):
                continue
            topic = document_topic[i]
            root_doc = document_title[i]
            n = len(root_doc)
            print('>' * n)
            print('Topic:', topic)
            print('Title:', root_doc)
            print()
            print('Most related articles:')
            for top, score in zip(tops, scores):
                if score > threshold:
                    topic = document_topic[top]
                    doc = document_title[top]
                    print()
                    print('Similairy score:{:.0f}%'.format(score * 100))
                    print('Topic:', topic)
                    print('Title:', doc)
                else:
                    break
            print('<' * n)
            print()

    def scatter_plot(
            self,
            topics,
            perplexcity=30,
            method='TSNE',
            save_to=None):
        """ 2D scatter plot of embedded documents

        Parameters
        ----------
        topics : array-like
            shape = (n_docs,), specifies the topic of
            each document in the corpus

        perplexcity : float
            TSNE parameter

        method : str
            eiter 'PCA' or 'TSNE'

        save_to : str, optional
            if not None, save figure to disk

        Returns
        -------
        g : plotnine Object

        """
        assert self._is_fitted, "Not fitted"
        g = plot_2d_projection(self.docvecs_, topics,
                               perplexity=perplexcity,
                               method=method)
        g.draw()
        if save_to:
            g.save(save_to)
        return g
