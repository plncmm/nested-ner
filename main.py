import codecs
import logging
import argparse
import os
import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt 
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from torch.optim import Adam
from torch.autograd import Variable
from dataset.request_data import samples_loader_from_minio
from utils.utils import simplify_entity
from dataset.vectorizer import Vectorizer
from dataset.data_preprocesing import split_sentences, create_dataset
from dataset.data_loader import DataLoader
from model import BiLSTM
from utils.utils import save_checkpoint, load_checkpoint, create_embedding_weights
from train import train

if __name__ == "__main__":
    dtrain, dval, dtest, vocab, tags, vocab_dict = create_dataset('dataset/conll/entities.conll')
    embedding_weights = create_embedding_weights(vocab,['embeddings/clinical_embeddings.vec'])
    bilstm = BiLSTM(vocab_size = len(vocab), n_tags = len(tags), embedding_dim=300, lstm_dim=128, embedding_dropout=0.5, lstm_dropout=0.5, output_layer_dropout=0.25, lstm_layers=2, embedding_weights= embedding_weights, use_bilstm = True)
    print(f"The model has {bilstm.count_parameters():,} trainable parameters.")
    data_loader = DataLoader()
    optimizer = Adam(bilstm.parameters(), lr=0.001)
    loss_function =  nn.BCELoss()
    history = train(bilstm, optimizer, loss_function, data_loader, dtrain, dval, dtest, 50, 16, tags, vocab_dict, no_improvement=3, checkpoint_path='models/best_model.pt')
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


    
    
