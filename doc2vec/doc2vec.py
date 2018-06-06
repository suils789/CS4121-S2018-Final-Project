import tensorflow as tf
import tensorflow_hub as hub
embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/1")

__all__=["doc2vec"]

def doc2vec(docs):
    """Embed each doc in docs to a vector

    Parameters
    ----------
    docs : array-like
        a array of strings

    Returns
    -------
    docvecs : np.ndarray
        the embedded document.
        each vector in docvecs corresponds to a
        doc in docs

    """
    tf.logging.set_verbosity(tf.logging.ERROR)

    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as session:
      session.run([tf.global_variables_initializer(),
                   tf.tables_initializer()
                  ])
      docvecs = session.run(embed(docs))

    print('Document Ebeddimg Done')
    print('Embedded X shape:',docvecs.shape)
    return docvecs


