import sys 
sys.path.append('../')
from dataset.vectorizer import Vectorizer
from dataset.data_preprocesing import create_dataset
import torch 
import torch.nn as nn
import numpy as np 
from eval import evaluate
from dataset.data_loader import DataLoader
import json

def model_predict(model, text, vocab, tags):
    entities = ['Finding', 'Procedure', 'Disease', 'Body_Part', 'Abbreviation', 'Family_Member', 'Medication']
    
    ex = text.split()
    ex_parsed = Vectorizer(vocab, tags).sentence_to_index(ex)
    output_batch = model(torch.LongTensor([ex_parsed]), [len(ex_parsed)]).reshape(-1,len(tags))
    output_batch = output_batch.detach().numpy()
    output_batch = (output_batch > 0.5) # Aquí el 0.5 debiese ser hiperparámetro según tipo| 
    predictions = []

    for j, elem in enumerate(output_batch):
        ex_tags = []
        for i, tag in enumerate(elem):
            if tag:
                ex_tags.append(list(tags.keys())[list(tags.values()).index(i)])
        token_labels =  ' '.join(ex_tags) if len(ex_tags)>0 else 'O'
        predictions.append(ex[j] + ' : ' + token_labels)
    
    return predictions

