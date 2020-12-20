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
from dataset.vectorizer import Vectorizer
from dataset.data_preprocesing import split_sentences, create_dataset
from dataset.data_loader import DataLoader
from model import BiLSTM
from utils.nn_utils import create_embedding_weights, plot_history
from train import train
from utils.general_utils import boolean_string

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_size', type = float, default = 0.8, required = False)
    parser.add_argument('--dev_size', type = float, default = 0.1, required = False)
    parser.add_argument('--test_size', type = float, default = 0.1, required = False)
    parser.add_argument('--bilstm',default=True, type=boolean_string)
    parser.add_argument('--static_embeddings', default=False, type=boolean_string)
    parser.add_argument('--clinical_embeddings', default=False, type=boolean_string)
    parser.add_argument('--spanish_embeddings',default=False, type=boolean_string)
    parser.add_argument('--embedding_dim', type = int, default = 300, required = False)
    parser.add_argument('--lstm_dim', type = int, default = 128, required = False)
    parser.add_argument('--lstm_layers', type = int, default = 2, required = False)
    parser.add_argument('--embedding_dropout', type = float, default = 0.5, required = False)
    parser.add_argument('--lstm_dropout', type = float, default = 0.5, required = False)
    parser.add_argument('--linear_dropout', type = float, default = 0.25, required = False)
    parser.add_argument('--max_epochs', type = int, default = 50, required = False)
    parser.add_argument('--batch_size', type = int, default = 16, required = False)
    parser.add_argument('--patience', type = int, default = 3, required = False)
    parser.add_argument('--no_improvement', type = int, default = 5, required = False)
    parser.add_argument('--learning_rate', type = int, default = 0.001, required = False)
    args = parser.parse_args()  
    train_size = args.train_size
    dev_size = args.dev_size
    test_size = args.test_size 
    static_embeddings = args.static_embeddings 
    add_clinical_embeddings = args.clinical_embeddings
    add_spanish_embeddings = args.spanish_embeddings
    embedding_dim = args.embedding_dim
    lstm_dim = args.lstm_dim 
    embedding_dropout = args.embedding_dropout
    lstm_dropout = args.lstm_dropout
    linear_dropout = args.linear_dropout
    lstm_layers = args.lstm_layers
    use_bilstm = args.bilstm
    max_epochs = args.max_epochs
    batch_size = args.batch_size 
    patience = args.patience
    no_improvement = args.no_improvement
    lr = args.learning_rate

    dtrain, dval, dtest, vocab, tags, vocab_dict = create_dataset('dataset/conll/entities.conll', train_size, dev_size, test_size)
    embedding_weights = None
    pretrained_embeddings = []
    if add_clinical_embeddings: 
        print('hola')
        pretrained_embeddings.append('embeddings/clinical_embeddings.vec')
    if add_spanish_embeddings: 
        print('hola')
        pretrained_embeddings.append('embeddings/spanish_embeddings.vec')
    print(static_embeddings)
    print(add_clinical_embeddings)
    print(add_spanish_embeddings)
    if len(pretrained_embeddings)!=0: embedding_weights = create_embedding_weights(vocab,pretrained_embeddings)
    
   
    bilstm = BiLSTM(vocab_size = len(vocab), n_tags = len(tags), embedding_dim=embedding_dim, lstm_dim=lstm_dim, \
        embedding_dropout=embedding_dropout, lstm_dropout=lstm_dropout, \
        output_layer_dropout=linear_dropout, lstm_layers=lstm_layers, embedding_weights = embedding_weights, use_bilstm = use_bilstm, static_embeddings = static_embeddings)
    print(f"The model has {bilstm.count_parameters():,} trainable parameters.")
    data_loader = DataLoader()
    optimizer = Adam(bilstm.parameters(), lr=lr)
    loss_function =  nn.BCELoss()
    entities = ['Finding', 'Procedure', 'Disease', 'Body_Part', 'Abbreviation', 'Family_Member', 'Medication']
    history = train(bilstm, optimizer, loss_function, data_loader, dtrain, dval, dtest, \
        max_epochs, batch_size, tags, vocab_dict, entities, no_improvement=no_improvement, patience = patience, checkpoint_path='models/best_model.pt')
    plot_history(history)


    
    
