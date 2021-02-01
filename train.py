import numpy as np 
import time 
import torch
from eval import evaluate
from metrics import entity_f1_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from metrics import get_entities_from_multi_label_conll, get_multi_conll_entities, micro_f1_score, keep_bio_format, fix
torch.manual_seed(0)
def epoch(model, optimizer, loss_function, data_iterator, num_steps, tags, vocab, entities, threshold):
    model.train() 
    epoch_loss = []
    f1 = []
    real = []
    pred = []
    for _ in range(num_steps):
        train_batch, labels_batch, lens = next(data_iterator)
        optimizer.zero_grad()
        output_batch = model.forward(train_batch, lens) 
        loss = loss_function(output_batch.view(-1, output_batch.shape[-1]), labels_batch.view(-1, labels_batch.shape[-1]).type_as(output_batch))
        epoch_loss.append(loss.item()*train_batch.shape[0])
        output_batch = output_batch.detach().numpy()
        output_batch = (output_batch > threshold) 
        labels_batch = labels_batch.detach().numpy()
        train_batch = train_batch.numpy()
        
        for a, b, c in zip(output_batch, labels_batch, train_batch):
            for p, r, t in zip(a, b, c):
              if vocab[t.item()]!='PAD':
                pred_tags = [list(tags.keys())[list(tags.values()).index(i)] for i,value in enumerate(p) if value]
                real_tags = [list(tags.keys())[list(tags.values()).index(i)] for i,value in enumerate(r) if value]
                if len(pred_tags)==0: pred_tags = ['O']
                pred.append(pred_tags)
                real.append(real_tags)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
        optimizer.step()
    
    original_entities = get_entities_from_multi_label_conll(keep_bio_format(fix(real,entities)), entities)
    multilabel_pred_entities = get_entities_from_multi_label_conll(keep_bio_format(fix(pred, entities)), entities)
    total_precision, total_recall, total_f1 = micro_f1_score(original_entities, multilabel_pred_entities, entities)
    return np.mean(np.array(epoch_loss)), total_f1


def train(model, optimizer, loss_function, data_loader, train_data, val_data, test_data, max_epochs, batch_size, tags, vocab, entities, no_improvement, patience, checkpoint_path, threshold):
    history = {"num_params": model.count_parameters(),
    'train_loss': [],
    'train_f1': [],
    'val_loss': [],
    'val_f1': []
    }
    
    best_val_loss = np.inf
    best_f1_score = 0
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
        train_data_iterator = data_loader.iterator(train_data, batch_size, len(tags), shuffle=True)
        train_loss, train_f1 = epoch(model, optimizer, loss_function, train_data_iterator, num_steps, tags, vocab, entities, threshold)
        print(f"\tTrain Loss: {train_loss} ")
        print(f"\tTrain F1-Score: {train_f1}\n")
        history['train_loss'].append(train_loss)
        history['train_f1'].append(train_f1)
        num_steps = (val_data['size']) // batch_size
        val_data_iterator = data_loader.iterator(val_data, batch_size, len(tags), shuffle=True)
        val_loss, val_f1 = evaluate(model, loss_function, val_data_iterator, num_steps, tags, vocab, entities, threshold, show_results = True)
        print(f"\tValidation Loss: {val_loss}")
        print(f"\tValidation F1-Score: {val_f1}\n")
        end_time = time.time()
        epoch_time = end_time - start_time
        print(f'Epoch time: {epoch_time}s')
        lr_scheduler.step(val_f1)
        if val_f1 > best_f1_score:
                print(f"Epoch {n_epoch:5d}: found better Val f1: {val_f1:.4f} (Train f1: {train_f1:.4f}), saving model...")
                model.save_state(checkpoint_path)
                best_f1_score = val_f1
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
    test_loss,test_f1 = evaluate(model, loss_function, test_data_iterator, num_steps, tags, vocab, entities, threshold,  show_results = True)
    print(f"\tTest Loss: {test_loss}")
    print(f"\tTest Best F1-Score: {test_f1} \n")
    return history
    
      
            

    
   