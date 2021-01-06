import numpy as np 
import torch 
import codecs
import random
from metrics import entity_f1_score, tabulate_f1_score, accuracy_score, tabulate_dict, exact_entity_f1_score, f1_score_token
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

def evaluate(model, loss_function, data_iterator, num_steps, tags, vocab, entities, show_results=False):
    model.eval()
    eval_loss = []
    pred = [] # Prediction array used to calculate strict f1-score entity-level metrics.
    real = []
    if show_results: 
      output_file = codecs.open('testing.conll', 'w', 'UTF-8')
      output_file2 = codecs.open('real.conll', 'w', 'UTF-8')
     
    with torch.no_grad():
      for _ in range(num_steps):
          data_batch, labels_batch, lens = next(data_iterator)
          output_batch = model(data_batch, lens)
          loss = loss_function(output_batch.view(-1, output_batch.shape[-1]), labels_batch.view(-1, labels_batch.shape[-1]).type_as(output_batch))
          eval_loss.append(loss.item()*8)
          output_batch = output_batch.detach().numpy()
          output_batch = (output_batch > 0.5) # Aquí el 0.5 debiese ser hiperparámetro según tipo| 
          labels_batch = labels_batch.detach().numpy()
          data_batch = data_batch.numpy()
          for a,b,c in zip(output_batch, labels_batch, data_batch):
            for p,r,t in zip(a,b,c):
              if vocab[t.item()]!='PAD':
                pred.append([list(tags.keys())[list(tags.values()).index(i)] for i,value in enumerate(p) if value])
                real.append([list(tags.keys())[list(tags.values()).index(i)] for i,value in enumerate(r) if value])
                if show_results:
                  output_file.write(f"{vocab[t.item()]} {' '.join([list(tags.keys())[list(tags.values()).index(i)] for i,value in enumerate(p) if value])}\n")
                  output_file2.write(f"{vocab[t.item()]} {' '.join([list(tags.keys())[list(tags.values()).index(i)] for i,value in enumerate(r) if value])}\n")
              else:
                if show_results:
                  output_file.write("")
                  output_file2.write("")  
            if show_results:
              output_file.write("\n")
              output_file2.write("\n")
              
    table, strict_f1_score = entity_f1_score(real, pred, entities)
    if show_results: 
      #print(f'El accuracy a nivel de token es: {round(accuracy_score(real, pred, ignore = None),2)}')
      #print(f"El accuracy en tokens múltiples es: {round(accuracy_score(real, pred, ignore = 'O', multiple=True),2)}")
      print(tabulate_dict(f1_score_token(real, pred,  entities)))
      print(tabulate_f1_score(exact_entity_f1_score(real, pred, entities), include_mean=True))
      print(tabulate_f1_score(table, include_mean=True))

      

      output_file.close()
      output_file2.close()
    return np.mean(np.array(eval_loss)), strict_f1_score
      

    