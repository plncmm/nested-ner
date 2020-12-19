import numpy as np 
import torch 
import codecs
from metrics import entity_f1_score, tabulate_f1_score, accuracy_score, tabulate_dict, exact_entity_f1_score, f1_score_token

def evaluate(model, loss_function, data_iterator, num_steps, tags, vocab, show_results=False):
    model.eval()
    eval_loss = []
    f1 = []
    pred = []
    real = []
    if show_results: 
      output_file = codecs.open('testing.conll', 'w', 'UTF-8')
      output_file2 = codecs.open('real.conll', 'w', 'UTF-8')
    with torch.no_grad():
      for _ in range(num_steps):
          data_batch, labels_batch, lens = next(data_iterator)
          output_batch = model.forward(data_batch, lens)
          output_batch = output_batch.view(-1, output_batch.shape[-1])
          labels_batch = labels_batch.view(-1, labels_batch.shape[-1])
          loss = loss_function(output_batch, labels_batch.type_as(output_batch))
          eval_loss.append(loss.item())
          output_batch = output_batch.detach().numpy()
          output_batch = (output_batch > 0.5) # Aquí el 0.5 debiese ser hiperparámetro según tipo| 
          labels_batch = labels_batch.detach().numpy()

          for i, (a, b) in enumerate(zip(output_batch, labels_batch)):
            if vocab[data_batch.view(output_batch.shape[0], -1).numpy()[i][0]]!='PAD':
              pred.append([list(tags.keys())[list(tags.values()).index(i)] for i,value in enumerate(a) if value])
              real.append([list(tags.keys())[list(tags.values()).index(i)] for i,value in enumerate(b) if value])
              if show_results:
                output_file.write(f"{vocab[data_batch.view(output_batch.shape[0], -1).numpy()[i][0]]} {' '.join([list(tags.keys())[list(tags.values()).index(i)] for i,value in enumerate(a) if value])}\n")
                output_file2.write(f"{vocab[data_batch.view(output_batch.shape[0], -1).numpy()[i][0]]} {' '.join([list(tags.keys())[list(tags.values()).index(i)] for i,value in enumerate(b) if value])}\n")
            else:
              if show_results:
                output_file.write("")
                output_file2.write("")      
          
    entities = ['Finding', 'Procedure', 'Disease', 'Body_Part', 'Abbreviation', 'Family_Member', 'Medication']
    table, strict_f1_score = entity_f1_score(real, pred, entities)
    if show_results: 
      print(f'El accuracy a nivel de token es: {round(accuracy_score(real, pred, ignore = None),2)}')
      print(f"El accuracy en tokens múltiples es: {round(accuracy_score(real, pred, ignore = 'O', multiple=True),2)}")
      print(tabulate_dict(f1_score_token(real, pred,  entities)))
      print(tabulate_f1_score(exact_entity_f1_score(real, pred, entities), include_mean=True))
      print(tabulate_f1_score(table, include_mean=True))
      output_file.close()
      output_file2.close()
    return np.mean(np.array(eval_loss)), strict_f1_score
      

    