import numpy as np 
import time 
import torch
from eval import evaluate
from metrics import entity_f1_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
torch.manual_seed(0)
def epoch(model, optimizer, loss_function, data_iterator, num_steps, tags, vocab, entities):
    model.train() 
    epoch_loss = []
    f1 = []
    real = []
    pred = []
    for i in range(num_steps):
        train_batch, labels_batch, lens = next(data_iterator) # En cada batch disponible obtengo el batch y los labels asociados a ese batch
        optimizer.zero_grad()
        output_batch = model.forward(train_batch, lens)       # Luego entreno el modelo con una pasada del forward del train_batch batch_size x n_tokens x n_tags
        output_batch = output_batch.view(-1, output_batch.shape[-1]) # batch_size*n_tokens x n_tags
        labels_batch = labels_batch.view(-1, labels_batch.shape[-1])   # 
        loss = loss_function(output_batch, labels_batch.type_as(output_batch))
        epoch_loss.append(loss.item()*8)
        output_batch = output_batch.detach().numpy()
        output_batch = (output_batch > 0.5) # Aquí el 0.5 debiese ser hiperparámetro según tipo| 
        labels_batch = labels_batch.detach().numpy()
        for i, (a, b) in enumerate(zip(output_batch, labels_batch)):
            if vocab[train_batch.view(output_batch.shape[0], -1).numpy()[i][0]]!='PAD':
              pred.append([list(tags.keys())[list(tags.values()).index(i)] for i,value in enumerate(a) if value])
              real.append([list(tags.keys())[list(tags.values()).index(i)] for i,value in enumerate(b) if value])
        loss.backward()
        optimizer.step()
    table, strict_f1_score = entity_f1_score(real, pred, entities)
    return np.mean(np.array(epoch_loss)), strict_f1_score


def train(model, optimizer, loss_function, data_loader, train_data, val_data, test_data, max_epochs, batch_size, tags, vocab, entities, no_improvement, patience, checkpoint_path):
    history = {"num_params": model.count_parameters(),
    'train_loss': [],
    'train_f1': [],
    'val_loss': [],
    'val_f1': []
    }
    
    best_val_loss = np.inf
    best_epoch = None
    # scheduler object from pytorch
    # reduce learning rate by a factor of 0.3 if there is no performance
    # improvement after 3 epochs
    lr_scheduler = ReduceLROnPlateau(
        optimizer=optimizer,
        patience=patience,
        factor=0.3,
        mode="max",
        verbose=True
    )
    
    n_epoch = 1
    n_stagnant = 0  # preparation for early stopping
    stop = False


    while not stop:
        start_time = time.time()
        num_steps = (train_data['size']) // batch_size
        train_data_iterator = data_loader.iterator(train_data, batch_size, len(tags), shuffle=False)
        train_loss, train_f1 = epoch(model, optimizer, loss_function, train_data_iterator, num_steps, tags, vocab, entities)
        print(f"\tTrain Loss: {train_loss} ")
        print(f"\tTrain F1-Score: {train_f1}\n")
        history['train_loss'].append(train_loss)
        history['train_f1'].append(train_f1)
        num_steps = (val_data['size']) // batch_size
        val_data_iterator = data_loader.iterator(val_data, batch_size, len(tags), shuffle=False)
        val_loss, val_f1 = evaluate(model, loss_function, val_data_iterator, num_steps, tags, vocab, entities,  show_results = False)
        print(f"\tValidation Loss: {val_loss}")
        print(f"\tValidation F1-Score: {val_f1}\n")
        end_time = time.time()
        epoch_time = end_time - start_time
        print(f'Epoch time: {epoch_time}s')
        lr_scheduler.step(val_f1)
        if (1.01 * val_loss) < (best_val_loss):
                print(f"Epoch {n_epoch:5d}: found better Val loss: {val_loss:.4f} (Train loss: {train_loss:.4f}), saving model...")
                model.save_state(checkpoint_path)
                best_val_loss = val_loss
                best_epoch = n_epoch
                n_stagnant = 0
        else:
            n_stagnant += 1
        history['val_loss'].append(val_loss)
        history['val_f1'].append(val_f1)
        if n_epoch >= max_epochs:
                print(f"Reach maximum number of epoch: {n_epoch}, stop training.")
                stop = True
        elif no_improvement is not None and n_stagnant >= no_improvement:
            print(f"No improvement after {n_stagnant} epochs, stop training.")
            stop = True
        else:
            n_epoch += 1
    
    if checkpoint_path:
        model.load_state(checkpoint_path)
        
    num_steps = (test_data['size']) // batch_size
    test_data_iterator = data_loader.iterator(test_data, batch_size, len(tags), shuffle=False)
    test_loss,test_f1 = evaluate(model, loss_function, test_data_iterator, num_steps, tags, vocab, entities,  show_results = True)
    print(f"\tTest Loss: {test_loss}")
    print(f"\tTest Best F1-Score: {test_f1} \n")
    return history
    
      
            

    
   