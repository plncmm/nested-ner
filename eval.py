import numpy as np 
import torch 
import codecs
import logging
from tqdm import tqdm
from metrics import entity_f1_score, tabulate_f1_score, keep_bio_format, fix, micro_f1_score
from entities import get_entities_from_multiconll
torch.manual_seed(0)
np.random.seed(0)

def evaluate(
        model, 
        loss_function, 
        data_iterator, 
        num_steps, 
        tags, 
        vocab, 
        entities, 
        threshold, 
        show_results = False
):
    model.eval()
    eval_loss = []
    pred = [] 
    real = []
    tokens = []
    real_tunned = {'tr1': [], 'tr2': [], 'tr3': [], 'tr4': [], 'tr5': [], 'tr6': [], 'tr7': []}
    pred_tunned = {'tr1': [], 'tr2': [], 'tr3': [], 'tr4': [], 'tr5': [], 'tr6': [], 'tr7': []}
    thresholds = {'tr1': 0.3, 'tr2': 0.35, 'tr3': 0.4, 'tr4': 0.45, 'tr5': 0.5, 'tr6': 0.55, 'tr7': 0.60}
    
    with torch.no_grad():
        for i in tqdm(range(num_steps), desc = f"Evaluation"):
            data_batch, labels_batch, chars_batch, lens = next(data_iterator)
            output_batch = model(data_batch, chars_batch, lens)
            loss = loss_function(output_batch.view(-1, output_batch.shape[-1]), labels_batch.view(-1, labels_batch.shape[-1]).type_as(output_batch))
            eval_loss.append(loss.item()*data_batch.shape[0])
            output_batch = output_batch.cpu().detach().numpy()
            labels_batch = labels_batch.cpu().detach().numpy()
            data_batch = data_batch.cpu().numpy()
            
            for a, b, c in zip(output_batch, labels_batch, data_batch):
                for p, r, t in zip(a, b, c):
                    if vocab[t.item()]!='PAD':
                        for k,v in thresholds.items():
                            pred_tags = [list(tags.keys())[list(tags.values()).index(i)] for i,value in enumerate(p) if value>v]
                            real_tags = [list(tags.keys())[list(tags.values()).index(i)] for i,value in enumerate(r) if value]
                            
                            if len(pred_tags)==0: 
                                pred_tags = ['O']
                            
                            pred_tunned[k].append(pred_tags)
                            real_tunned[k].append(real_tags)
                        
    best_f1 = 0
    best_threshold = 0
    best_prec = 0
    best_recall = 0
    best_pred = []
    best_true = []
    thresholds_ar = []
    for k,v in pred_tunned.items():
        original_entities = get_entities_from_multiconll(keep_bio_format(fix(real_tunned[k],entities)))
        multilabel_pred_entities = get_entities_from_multiconll(keep_bio_format(fix(pred_tunned[k], entities)))
        prec, recall, f1 = entity_f1_score(original_entities, multilabel_pred_entities, entities)
        total_precision, total_recall, total_f1 = micro_f1_score(original_entities, multilabel_pred_entities, entities)
        thresholds_ar.append(f'{k}: {total_f1}')
        if total_f1>best_f1:
            best_f1 = total_f1
            best_prec = total_precision
            best_recall = total_recall
            best_threshold = k 
            best_pred = multilabel_pred_entities
            best_true = original_entities
    prec, recall, f1 = entity_f1_score(best_true, best_pred, entities)
    print(tabulate_f1_score(prec, recall, f1))
    logging.info(f'Best threshold: {thresholds[best_threshold]}')
    logging.info(f"Recall: {format(round(best_recall*100,1), '.1f')}, Precision: {format(round(best_prec*100,1), '.1f')}, F1-Score: {format(round(best_f1*100,1), '.1f')}")
    return np.mean(np.array(eval_loss)), best_f1