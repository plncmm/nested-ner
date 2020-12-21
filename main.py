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
from eval import evaluate
torch.manual_seed(0)

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
    parser.add_argument('--learning_rate', type = float, default = 0.001, required = False)
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
    test_tokens = []
    for x in dtest['data']:
        for token in Vectorizer(vocab,tags).index_to_sentence(x):
            test_tokens.append(token)
    test_tokens = list(set(test_tokens))

    train_tokens = []
    for x in dtrain['data']:
        for token in Vectorizer(vocab,tags).index_to_sentence(x):
            train_tokens.append(token)
    train_tokens = list(set(train_tokens))

    counter = 0
    for token in test_tokens:
        if token in train_tokens:
            counter+=1
    print(counter/len(test_tokens))

    embedding_weights = None
    pretrained_embeddings = []
    if add_clinical_embeddings: 
        pretrained_embeddings.append('embeddings/clinical_embeddings.vec')
    if add_spanish_embeddings: 
        pretrained_embeddings.append('embeddings/spanish_embeddings.vec')
    #if len(pretrained_embeddings)!=0: embedding_weights = create_embedding_weights(vocab,pretrained_embeddings)
    
   
    # bilstm = BiLSTM(vocab_size = len(vocab), n_tags = len(tags), embedding_dim=embedding_dim, lstm_dim=lstm_dim, \
    #         embedding_dropout=embedding_dropout, lstm_dropout=lstm_dropout, \
    #         output_layer_dropout=linear_dropout, lstm_layers=lstm_layers, embedding_weights = embedding_weights, use_bilstm = use_bilstm, static_embeddings = static_embeddings)
    # print(f"The model has {bilstm.count_parameters():,} trainable parameters.")
    # data_loader = DataLoader()
    # optimizer = Adam(bilstm.parameters(), lr=lr)
    # loss_function =  nn.BCELoss()
    # entities = ['Finding', 'Procedure', 'Disease', 'Body_Part', 'Abbreviation', 'Family_Member', 'Medication']
    # history = train(bilstm, optimizer, loss_function, data_loader, dtrain, dval, dtest, \
    #          max_epochs, batch_size, tags, vocab_dict, entities, no_improvement=no_improvement, patience = patience, checkpoint_path='models/best_model.pt')
    # plot_history(history)
    



    dtrain, dval, dtest, vocab, tags, vocab_dict = create_dataset('dataset/conll/entities.conll', train_size, dev_size, test_size)
    
    model = BiLSTM(vocab_size = len(vocab), n_tags = len(tags), embedding_dim=embedding_dim, lstm_dim=lstm_dim, \
           embedding_dropout=embedding_dropout, lstm_dropout=lstm_dropout, \
           output_layer_dropout=linear_dropout, lstm_layers=lstm_layers, embedding_weights = embedding_weights, use_bilstm = use_bilstm, static_embeddings = static_embeddings)
    data_loader = DataLoader()
    loss_function =  nn.BCELoss()
    entities = ['Finding', 'Procedure', 'Disease', 'Body_Part', 'Abbreviation', 'Family_Member', 'Medication']
    model.load_state('models/best_model.pt')
    num_steps = (dtest['size']) // batch_size
    test_data_iterator = data_loader.iterator(dtest, batch_size, len(tags), shuffle=False)
    test_loss,test_f1 = evaluate(model, loss_function, test_data_iterator, num_steps, tags, vocab_dict, entities,  show_results = True)
    print(f"\tTest Loss: {test_loss}")
    print(f"\tTest Best F1-Score: {test_f1} \n")

    ex = []
    ex_parsed = Vectorizer(vocab, tags).sentence_to_index(ex)
    output_batch = model(torch.LongTensor([ex_parsed]), [3])
    print(output_batch)
    output_batch = output_batch.detach().numpy()
    output_batch = (output_batch > 0.5) # Aquí el 0.5 debiese ser hiperparámetro según tipo| 
    
    for elem in output_batch:
        for i, tag in enumerate(elem):
            if tag:
                print([list(tags.keys())[list(tags.values()).index(i)]])
        print('\n')    
       


                
    
    
