import numpy as np
import random
import torch 
from torch.autograd import Variable

class DataLoader:
    def pad_labels(self, n, n_tags):
        final_array = []
        padded_array = [0]*(n_tags-1)
        padded_array.insert(0,1)
        for i in range(n):
            final_array.append(padded_array)
        return final_array

    def iterator(self, data, batch_size, n_tags, shuffle=False):
        order = list(range(data['size']))
        if shuffle:
            random.seed(230)
            random.shuffle(order)
        
        for i in range((data['size'])//batch_size):
            #voy aquí
            lens = []
            batch_sentences = [data['data'][idx] for idx in order[i*batch_size:(i+1)*batch_size]]
            batch_tags = [data['labels'][idx] for idx in order[i*batch_size:(i+1)*batch_size]]
            batch_max_len = max([len(s) for s in batch_sentences])
            batch_data = np.zeros((len(batch_sentences), batch_max_len))
            batch_labels = np.zeros((len(batch_sentences), batch_max_len, n_tags))
            for j in range(len(batch_sentences)): # For each sentence in the batch
                cur_len = len(batch_sentences[j]) # First we get the current sentence len
                lens.append(cur_len)
                batch_data[j][:cur_len] = batch_sentences[j] # Then we fill the batch data with the sentence until the len position, the rest are padding words
                batch_labels[j][:cur_len] = batch_tags[j][:][:cur_len] # The same for labels
                n = batch_max_len - cur_len # We obtain the number of padding tokens with the aim of filling the labels with 1 0 0 0......
                if batch_max_len!=cur_len: batch_labels[j][cur_len:] = self.pad_labels(n, n_tags)
                # Me falta poner que en la posición 0 haya un 1 para los token que sean PADS aquí.
            batch_data, batch_labels = torch.LongTensor(batch_data), torch.LongTensor(batch_labels)

            if torch.cuda.is_available():
                batch_data, batch_labels = batch_data.cuda(), batch_labels.cuda()

            batch_data, batch_labels = Variable(batch_data), Variable(batch_labels)
            yield batch_data, batch_labels, lens

