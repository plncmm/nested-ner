import json
import torch
import torch.nn as nn
from utils.data_utils.vectorizer import Vectorizer
from utils.data_utils.data_preprocesing import create_dataset_from_conll
from utils.data_utils.data_loader import DataLoader
from utils.nn_utils.nn_utils import create_embedding_weights, plot_history
from model import SequenceMultilabelingTagger
from torch.optim import Adam
from train import train, train_full

if __name__ == "__main__":
    # Loading Hyparameters.
    f = open('params.json', )
    params = json.load(f)
    
    
    # Using Cuda/CPU.
    device = params["device"]
    available_gpu = torch.cuda.is_available()
    if available_gpu:
        print(f"GPU is available: {torch.cuda.get_device_name(0)}")
        use_device = torch.device(device)
    else:
        use_device = torch.device(device)
   
    
    # Waiting list dataset and wl pre-trained embeddings.
    if params['dataset'] == 'wl':
        entities = ['Finding', 'Procedure', 'Disease', 'Body_Part', 'Abbreviation', 'Family_Member', 'Medication']
        dtrain, dval, dtest, dtrain_full, vocab, char_vocab, tags, vocab_dict, vectorizer = create_dataset_from_conll(
                                                                                                    'data/wl/wl_multilabel_train.conll', 
                                                                                                    'data/wl/wl_multilabel_test.conll', 
                                                                                                    'data/wl/wl_multilabel_dev.conll', 
                                                                                                    'wl'
        )
        embed_path = 'embeddings/wl/clinical.vec'
    
    # GENIA dataset and BioNLP embeddings.
    if params['dataset'] == 'genia':
        entities = ['RNA', 'DNA', 'cell_line', 'cell_type', 'protein']
        dtrain, dval, dtest, dtrain_full, vocab, char_vocab, tags, vocab_dict, vectorizer = create_dataset_from_conll(
                                                                                                    'data/genia/genia_multilabel_train.conll', 
                                                                                                    'data/genia/genia_multilabel_test.conll', 
                                                                                                    'data/genia/genia_multilabel_dev.conll', 
                                                                                                    'genia'
        )
        embed_path = 'embeddings/bionlp.bin'


    # Sequence Multilabeling Model.
    tagger = SequenceMultilabelingTagger(
                    corpus = params["dataset"], 
                    vectorizer = vectorizer, 
                    n_tags = len(tags), 
                    embed_path = embed_path, 
                    use_char_embeddings = params["use_char_embeddings"],
                    use_flair_embeddings = params["use_flair_embeddings"],
                    use_bert_embeddings = params["use_bert_embeddings"],
                    use_pretrained_embeddings = params["use_pretrained_embeddings"],
                    embedding_dropout = params["embedding_dropout"], 
                    lstm_dim = params["lstm_dim"], 
                    lstm_layers = params["lstm_layers"],
                    lstm_dropout = params["lstm_dropout"],
                    use_bilstm = params["use_bilstm"], 
                    use_attn_layer = params["use_attn_layer"],
                    attn_heads = params["attn_heads"],
                    attn_dropout = params["attn_dropout"],
                    device = device
    )

    print(f"The model has {tagger.count_parameters():,} trainable parameters.")
    tagger = tagger.to(device)

    

    # Optimization functions
    optimizer = Adam(tagger.parameters(), lr = params["lr"])
    loss_function =  nn.BCELoss()
    
    
    data_loader = DataLoader()
    if params["train_with_dev"]:
        train_full(
            tagger, 
            optimizer, 
            loss_function, 
            data_loader, 
            dtrain_full, dtest, 
            params["max_epochs"], 
            params["batch_size"], 
            tags, 
            vocab_dict, 
            entities, 
            checkpoint_path=f"pretrained_model/{params['dataset']}/{params['dataset']}-best.pt", 
            threshold = params['threshold'],
            device = device
        )
    else:
        history = train(     
            tagger, 
            optimizer, 
            loss_function, 
            data_loader, 
            dtrain, 
            dval, 
            dtest, 
            params["max_epochs"], 
            params["batch_size"], 
            tags, 
            vocab_dict, 
            entities, 
            no_improvement=params["no_improvement"], 
            patience = params["patience"], 
            checkpoint_path=f"pretrained_model/{params['dataset']}/{params['dataset']}-best.pt", 
            threshold = params['threshold'],
            device = device
        )
        plot_history(history, params['dataset'])
    
       


                
    
    
