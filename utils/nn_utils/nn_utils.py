import torch 
import numpy as np
import matplotlib.pyplot as plt
from gensim.models.keyedvectors import KeyedVectors

def create_embedding_weights(vocab, path, binary):
    """
    Function used to create embedding matrix from embedding file.
    """

    emb = KeyedVectors.load_word2vec_format(path, binary = binary)
    size = emb.vector_size
    vocab_size = len(vocab)+2
    vocab.insert(0,'PAD')
    vocab.insert(1,'UNK')
    embedding_matrix = np.random.uniform(-0.25, 0.25, (vocab_size, size))
    for i,word in enumerate(vocab):
      if i==0: # Embedding vector associated with padding.
        embedding_matrix[i] = np.zeros(size)
        continue
      embedding_vector = getVector(word, emb)
      if embedding_vector is not None:
          embedding_matrix[i]=embedding_vector
    return torch.FloatTensor(embedding_matrix)


def getVector(str, model):
    if str.lower() in model:
      return model[str.lower()]
    elif str in model:
      return model[str]
    else:
      return None


def plot_history(history, dataset):
  plt.plot(history['train_loss'])
  plt.plot(history['val_loss'])
  plt.title(f'{dataset} train loss v/s val loss across the epochs')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train', 'val'], loc='upper left')
  plt.show()
  plt.plot(history['train_f1'])
  plt.plot(history['val_f1'])
  plt.title(f'{dataset} train loss v/s val f1-score across the epochs')
  plt.ylabel('f1-score')
  plt.xlabel('epoch')
  plt.legend(['train', 'val'], loc='upper left')
  plt.show()