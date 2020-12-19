import torch 
import numpy as np
import time
from gensim.models.keyedvectors import KeyedVectors

def simplify_entity(entity):
    """
    Function used only for the Waiting List corpus, it generalizes entities so as not to have so many classes.
    
    Parameters:
    entity (string): Entity name.
    Returns:
    _ (string): Returns the simplified entity or the original depending on the entity type.
    """
    if entity in ["Laboratory_or_Test_Result", "Sign_or_Symptom", "Clinical_Finding"]:
        return "Finding"
    elif entity in ["Procedure", "Laboratory_Procedure", "Therapeutic_Procedure", "Diagnostic_Procedure"]:
        return "Procedure"
    return entity

def load_checkpoint(path):
    checkpoint = torch.load(path)
    return checkpoint

def save_checkpoint(state, path):
    torch.save(state, path)

def truncate(n, decimals=0):
    multiplier = 10 ** decimals
    return int(n * multiplier) / multiplier

def create_embedding_weights(vocab, paths):
    print("Time to load embeddings....")
    start = time.time()
    model = {}
    idx = 0
    size = 0
    for path in paths:
        model[idx]=KeyedVectors.load_word2vec_format(path, binary=False)
        size+=model[idx].vector_size
        idx+=1
    
    vocab_size = len(vocab)+2
    vocab.insert(0,'PAD')
    vocab.insert(1,'UNK')
    embedding_matrix = np.zeros((vocab_size, size))
    counter = 0
    for i,word in enumerate(vocab):
      embedding_vector = getVector(word, model[0], model[1]) if len(model)==2 else getUniqueVector(word, model[0])
      if embedding_vector is not None:
          counter+=1
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

    