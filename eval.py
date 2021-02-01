import numpy as np 
import torch 
import codecs
from metrics import entity_f1_score, tabulate_f1_score, get_entities_from_multi_label_conll, get_multi_conll_entities, keep_bio_format, fix, micro_f1_score
torch.manual_seed(0)
np.random.seed(0)

def evaluate(model, loss_function, data_iterator, num_steps, tags, vocab, entities, threshold, show_results = False):
    model.eval()
    eval_loss = []
    pred = [] 
    real = []

    if show_results: 
      output_file = codecs.open('test_predictions.conll', 'w', 'UTF-8')
     
    with torch.no_grad():
      for _ in range(num_steps):
          data_batch, labels_batch, lens = next(data_iterator)
          output_batch = model(data_batch, lens)
          loss = loss_function(output_batch.view(-1, output_batch.shape[-1]), labels_batch.view(-1, labels_batch.shape[-1]).type_as(output_batch))
          eval_loss.append(loss.item()*data_batch.shape[0])
          output_batch = output_batch.detach().numpy()
          output_batch = (output_batch >= threshold)  
          labels_batch = labels_batch.detach().numpy()
          data_batch = data_batch.numpy()
          
          for a, b, c in zip(output_batch, labels_batch, data_batch):
            for p, r, t in zip(a, b, c):
              if vocab[t.item()]!='PAD':
                pred_tags = [list(tags.keys())[list(tags.values()).index(i)] for i,value in enumerate(p) if value]
                real_tags = [list(tags.keys())[list(tags.values()).index(i)] for i,value in enumerate(r) if value]
                if len(pred_tags)==0: pred_tags = ['O']
                pred.append(pred_tags)
                real.append(real_tags)
                if show_results:
                  output_file.write(f"{vocab[t.item()]} {' '.join(pred_tags)}\n")
                
              else:
                if show_results:
                  output_file.write("") 

            if show_results:
              output_file.write("\n")
    
    original_entities = get_entities_from_multi_label_conll(keep_bio_format(fix(real,entities)), entities)
    multilabel_pred_entities = get_entities_from_multi_label_conll(keep_bio_format(fix(pred, entities)), entities)
    prec, recall, f1 = entity_f1_score(original_entities, multilabel_pred_entities, entities)
    total_precision, total_recall, total_f1 = micro_f1_score(original_entities, multilabel_pred_entities, entities)
    print(f"Recall: {format(round(total_recall*100,1), '.1f')}, Precision: {format(round(total_precision*100,1), '.1f')}, F1-Score: {format(round(total_f1*100,1), '.1f')}")
    
    if show_results: 
      print(tabulate_f1_score(prec, recall, f1))
      output_file.close()

    return np.mean(np.array(eval_loss)), total_f1