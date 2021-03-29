import numpy as np
import random
import torch 

class DataLoader:

    def pad_labels(self, n, n_tags):  
        sentence_labels = []

        # Labels associated with padding words.
        token_labels = [0] * (n_tags-1)
        token_labels.insert(0, 1)
        
        for i in range(n):
            sentence_labels.append(token_labels)

        return sentence_labels

    def iterator(self, data, batch_size, n_tags, shuffle = True, device='cuda'):
        order = list(range(data['size']))
        if shuffle:
            random.shuffle(order)
        
        for i in range((data['size'])//batch_size):
            lens = []
            batch_sentences = [data['data'][idx] for idx in order[i*batch_size:(i+1)*batch_size]]
            batch_chars = [data['chars'][idx] for idx in order[i*batch_size:(i+1)*batch_size]]
            batch_char_max_len = max([len(word) for sent in batch_chars for word in sent])
            batch_tags = [data['labels'][idx] for idx in order[i*batch_size:(i+1)*batch_size]]
            batch_max_len = max([len(s) for s in batch_sentences])
            batch_data = np.zeros((len(batch_sentences), batch_max_len))
            batch_chars_data = np.zeros((len(batch_sentences), batch_max_len, batch_char_max_len))
            batch_labels = np.zeros((len(batch_sentences), batch_max_len, n_tags))

            for j in range(len(batch_sentences)): 
                cur_len = len(batch_sentences[j]) 
                lens.append(cur_len)
                batch_data[j][:cur_len] = batch_sentences[j] 
                batch_labels[j][:cur_len] = batch_tags[j][:][:cur_len] 
                
                for i, word in enumerate(batch_chars[j]):
                    batch_chars_data[j][i][:len(word)] = word
                
                n = batch_max_len - cur_len 
                
                if batch_max_len!=cur_len: 
                    batch_labels[j][cur_len:] = self.pad_labels(n, n_tags)

            batch_data, batch_labels, batch_chars_data = torch.LongTensor(batch_data), torch.LongTensor(batch_labels), torch.LongTensor(batch_chars_data)

            if torch.cuda.is_available():
                batch_data, batch_labels, batch_chars_data = batch_data.to(device), batch_labels.to(device), batch_chars_data.to(device)

            yield batch_data, batch_labels, batch_chars_data, lens

