import gensim
from urllib.request import urlretrieve
import tempfile
import os
import numpy as np

# region utils


def maybe_make_directory(dir_path):
    """Make directory if it does not already exist
    Parameters
    ----------
    dir_path : str
        directory path. can be relative path or absolute path
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
# endregion


class LexVec:
    """ using pretrained lexvec for word embedding
    return the embedded results when called on a word
    or a lits of word (a document)
    """
    def __init__(self, pretrained_weights='wiki2015',
                 weights_fn='LexVec_weights.txt',
                 cache_path='temp'):

        self.__pretrained = {
            'wiki2015': 'http://nlpserver2.inf.ufrgs.br/alexandres/vectors/lexvec.enwiki%2bnewscrawl.300d.W.pos.vectors.gz'}

        dir = tempfile.gettempdir() if cache_path == 'temp' else cache_path
        dir = os.path.abspath(dir)
        maybe_make_directory(dir)
        weights_path = os.path.join(dir, weights_fn)
        if not os.path.isfile(weights_path):
            url = self.__pretrained[pretrained_weights]
            print('Dowloading {} from {}'.format(pretrained_weights, url)
                  )
            urlretrieve(url, weights_path)
        else:
            print('{} already exists in {}'.format(weights_fn, dir))
        self.model = gensim.models.KeyedVectors.load_word2vec_format(
            weights_path, binary=False)

    def __call__(self, x):
        if isinstance(x, str):
            return self.model[x]
        elif isinstance(x, (list, np.ndarray)):
            #word2vec_dim = len(self.model['i'])
            x = np.array(x)
            #embedded = np.zeros((x.shape+(word2vec_dim,)))
            assert  x.ndim==1
            return self.model[[w for w in x if w in self.model.vocab]]

