import codecs
import numpy as np
from tabulate import tabulate

def fix(pred, entities):
    new_pred = []
    total_o = 0
    for token_labels in pred:
        if 'O' in token_labels and len(token_labels)>1:
            ar = []
            for tag in token_labels:
                if tag!='O':
                    ar.append(tag)
            for entity in entities:
                if f'B-{entity}' in ar and f'I-{entity}' in ar:
                    ar.remove(f'B-{entity}')
            new_pred.append(ar)
        
        else:
            ar = []
            for tag in token_labels:
                ar.append(tag)
            for entity in entities:
                if f'B-{entity}' in ar and f'I-{entity}' in ar:
                    ar.remove(f'B-{entity}')
            new_pred.append(ar)
    return new_pred


def keep_bio_format(pred):
    new_pred = []
    new_pred.append(pred[0])
    for i, elem in enumerate(pred[1:]):
        if i==0: 
          previous_token = pred[0]
        ar = []
        for tag in elem:
            ar.append(tag)

        for tag in ar:
            if tag.startswith('I-'):
                if f'B-{tag[2:]}' not in previous_token and f'I-{tag[2:]}' not in previous_token:
                    ar.remove(tag)
                   
        previous_token = ar
        new_pred.append(ar)
  
    return new_pred

def entity_f1_score(true, pred, entities):
    prec_dict = {}
    recall_dict = {}
    f1_dict = {}

    for entity in entities:
        prec_dict[entity] = 0
        recall_dict[entity] = 0
        f1_dict[entity] = 0
    
    for entity in entities:
        true[entity].sort(key=lambda x: x[1])
        pred[entity].sort(key=lambda x: x[1])
        tp = 0
        fn = 0
        fp = 0
        
        for elem in true[entity]:
            if elem not in pred[entity]:
                fn += 1

        for elem in pred[entity]:
            if elem not in true[entity]:
                fp += 1    
          
        for elem in true[entity]:
            if elem in pred[entity]:
                tp += 1    
        
        recall = tp/(tp+fn) if (tp+fn)!=0 else 0
        recall_dict[entity] = recall

        precision = tp/(tp+fp) if (tp+fp)!=0 else 0
        prec_dict[entity] = precision
        f1_dict[entity] = ((2*precision*recall)/(precision+recall), len(true[entity])) if (precision+recall)!=0 else (0, len(true[entity]))
    
    return prec_dict, recall_dict, f1_dict

def tabulate_f1_score(prec, recall, f1, include_mean=False):
    f1_tokens = []
    for k,v in f1.items():
        if v[0]!=0:
            f1_tokens.append([k, format(round(prec[k]*100,1), '.1f'), float(format(round(recall[k]*100,1), '.1f')), format(round(v[0]*100,1), '.1f'), v[1]]) 
    return tabulate(f1_tokens, headers=['Entity type', 'Precision', 'Recall', 'F1-Score', 'Support', 'Tokens mean'])

def micro_f1_score(true, pred, entities):
    # Esta función la puedo modificar para devolver un diccionario con los micro f1-score de cada entidad.
    tp = 0
    fn = 0
    fp = 0

    for entity in entities:
        for elem in true[entity]:
            if elem not in pred[entity]:
                fn += 1
        for elem in pred[entity]:
            if elem not in true[entity]:
                fp += 1     
        for elem in true[entity]:
            if elem in pred[entity]:
                tp += 1    
    precision = tp/(tp+fp) if (tp+fp)!=0 else 0
    recall = tp/(tp+fn) if (tp+fn)!=0 else 0
    f1_score = (2*precision*recall)/(precision+recall) if (precision+recall)!=0 else 0
    return precision, recall, f1_score


def real_multi_label_format(filepath):
    f = codecs.open(filepath, 'r', 'utf-8').read()
    annotations = f.split('\n\n')
    labels = []
    for anno in annotations: 
      for line in anno.splitlines():
          line_info = line.split()
          if len(line_info)==1 or 'PAD' in line_info: 
              labels.append(['O'])
          else:
              labels.append(line_info[1:])
    return labels 