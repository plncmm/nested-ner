import numpy as np 
import torch 
import codecs
from metrics import entity_f1_score, tabulate_f1_score, get_entities_from_multi_label_conll, keep_bio_format, fix, micro_f1_score
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
    real_tunned = {'tr1': [], 'tr2': [], 'tr3': [], 'tr4': []}
    pred_tunned = {'tr1': [], 'tr2': [], 'tr3': [], 'tr4': []}
    thresholds = {'tr1': 0.3, 'tr2': 0.4, 'tr3': 0.5, 'tr4': 0.6}
    if show_results: 
        output_file = codecs.open('test_predictions.conll', 'w', 'UTF-8')
     
    with torch.no_grad():
        for i in range(num_steps):
            if i%100==0: 
                print(f'Evaluation Batch: {i} / {num_steps}')
            
            data_batch, labels_batch, chars_batch, lens = next(data_iterator)
            output_batch = model(data_batch, chars_batch, lens)
            loss = loss_function(output_batch.view(-1, output_batch.shape[-1]), labels_batch.view(-1, labels_batch.shape[-1]).type_as(output_batch))
            eval_loss.append(loss.item()*data_batch.shape[0])
            output_batch = output_batch.cpu().detach().numpy()
            #output_batch = (output_batch >= threshold)  
            
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
                        
                        #if show_results:
                        #    output_file.write(f"{vocab[t.item()]} {' '.join(pred_tags)}\n")
                      
                    #else:
                    #    if show_results:
                    #        output_file.write("") 

                if show_results:
                    output_file.write("\n")
    
    best_f1 = 0
    best_threshold = 0
    thresholds_ar = []
    for k,v in pred_tunned.items():
        original_entities = get_entities_from_multi_label_conll(keep_bio_format(fix(real_tunned[k],entities)), entities)
        multilabel_pred_entities = get_entities_from_multi_label_conll(keep_bio_format(fix(pred_tunned[k], entities)), entities)
        prec, recall, f1 = entity_f1_score(original_entities, multilabel_pred_entities, entities)
        total_precision, total_recall, total_f1 = micro_f1_score(original_entities, multilabel_pred_entities, entities)
        thresholds_ar.append(f'{k}: {total_f1}')
        if total_f1>best_f1:
            best_f1 = total_f1
            best_threshold = k 
            
    print(thresholds_ar)
    print(f'El mejor threshold en este caso fue: {best_threshold}')
    # Esta linea de abajo hay que cambiarla ya que no toma el valor del mejor f1
    print(f"Recall: {format(round(total_recall*100,1), '.1f')}, Precision: {format(round(total_precision*100,1), '.1f')}, F1-Score: {format(round(best_f1*100,1), '.1f')}")
    #a = codecs.open('result.txt', 'w', 'utf-8')
    #a.write(f"Recall: {format(round(total_recall*100,1), '.1f')}, Precision: {format(round(total_precision*100,1), '.1f')}, F1-Score: {format(round(total_f1*100,1), '.1f')}")
    #if show_results: 
    #    print(tabulate_f1_score(prec, recall, f1))
    #    output_file.close()

    return np.mean(np.array(eval_loss)), best_f1