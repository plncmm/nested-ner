import torch 
import numpy as np
import matplotlib.pyplot as plt
from gensim.models.keyedvectors import KeyedVectors


def create_embedding_weights(vocab, paths):
    model = {}
    size = 0
    for i, path in enumerate(paths):
        model[i]=KeyedVectors.load_word2vec_format(path, binary=False)
        size+=model[i].vector_size
        
    vocab_size = len(vocab)+2
    vocab.insert(0,'PAD')
    vocab.insert(1,'UNK')
    embedding_matrix = np.zeros((vocab_size, size))
    for i,word in enumerate(vocab):
      embedding_vector = getVector(word, model[0], model[1]) if len(model)==2 else getUniqueVector(word, model[0])
      if embedding_vector is not None:
          embedding_matrix[i]=embedding_vector
    return torch.FloatTensor(embedding_matrix)

def getVector(str, model1, model2):
    word = str.lower()
    if word in model1 and word in model2:
      return np.concatenate((model1[word], model2[word]), axis=None)  
    if word in model1 and word not in model2:
      return np.concatenate((model1[word], np.zeros((model2.vector_size))), axis=None)
    if word in model2 and word not in model1:
      return np.concatenate((np.zeros((model1.vector_size)),model2[word]), axis=None)
    else:
      return None

def getUniqueVector(str, model):
    word = str.lower()
    if word in model:
      return model[word]
    else:
      return None


def plot_history(history):
  plt.plot(history['train_loss'])
  plt.plot(history['val_loss'])
  plt.title('Train loss v/s val loss across the epochs')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train', 'val'], loc='upper left')
  plt.show()
  plt.plot(history['train_f1'])
  plt.plot(history['val_f1'])
  plt.title('Train loss v/s val f1-score across the epochs')
  plt.ylabel('f1-score')
  plt.xlabel('epoch')
  plt.legend(['train', 'val'], loc='upper left')
  plt.show()