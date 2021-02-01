import time
import json
import torch.nn as nn
from utils.data_utils.vectorizer import Vectorizer
from utils.data_utils.data_preprocesing import create_dataset_from_conll
from utils.data_utils.data_loader import DataLoader
from utils.nn_utils.nn_utils import create_embedding_weights, plot_history
from model import BiLSTM
from torch.optim import Adam
from train import train
from eval import evaluate

if __name__ == "__main__":
    f = open('config.json', )
    params = json.load(f)
    
    embedding_weights = None
    if params['dataset'] == 'wl':
        entities = ['Finding', 'Procedure', 'Disease', 'Body_Part', 'Abbreviation', 'Family_Member', 'Medication']
        dtrain, dval, dtest, vocab, tags, vocab_dict = create_dataset_from_conll('data/wl/wl_train.conll', 'data/wl/wl_test.conll', 'data/wl/wl_dev.conll', 'wl')
        if params['pretrained_embeddings']:
            pretrained_embeddings = 'embeddings/wl/clinical.vec'
            embedding_weights = create_embedding_weights(vocab, pretrained_embeddings, False)

    if params['dataset'] == 'genia':
        entities = ['RNA', 'DNA', 'cell_line', 'cell_type', 'protein']
        dtrain, dval, dtest, vocab, tags, vocab_dict = create_dataset_from_conll('data/genia/genia_train.conll', 'data/genia/genia_test.conll', 'data/genia/genia_dev.conll', 'genia')
        if params['pretrained_embeddings']:
            pretrained_embeddings = 'embeddings/genia/bionlp.bin'
            embedding_weights = create_embedding_weights(vocab, pretrained_embeddings, True)


    ner_model = BiLSTM(vocab_size = len(vocab), n_tags = len(tags), embedding_dim=params["embedding_dim"], lstm_dim= params["lstm_dim"], \
             embedding_dropout=params["embedding_dropout"], lstm_dropout=params["lstm_dropout"], \
             output_layer_dropout=params["linear_dropout"], lstm_layers= params["lstm_layers"], embedding_weights = embedding_weights, use_bilstm = params["use_bilstm"], static_embeddings = params["static_embeddings"])
    print(f"The model has {ner_model.count_parameters():,} trainable parameters.")

    data_loader = DataLoader()
    optimizer = Adam(ner_model.parameters(), lr= params["lr"])
    loss_function =  nn.BCELoss()
    history = train(ner_model, optimizer, loss_function, data_loader, dtrain, dval, dtest, params["max_epochs"], params["batch_size"], tags, vocab_dict, entities, no_improvement=params["no_improvement"], patience = params["patience"], checkpoint_path=f"pretrained_model/{params['dataset']}/{params['dataset']}-best.pt", threshold = params['threshold'])
    plot_history(history, params['dataset'])
    
       


                
    
    
