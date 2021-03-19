import numpy as np
import random
import torch 

class DataLoader:

    def pad_labels(self, n, n_tags):  
        """
        Function used to pad labels. A 1 is placed in the first position, which corresponds to the pad index.
        """

        sentence_labels = []
        token_labels = [0] * (n_tags-1)
        token_labels.insert(0, 1)
        
        for i in range(n):
            sentence_labels.append(token_labels)

        return sentence_labels

    def iterator(self, data, batch_size, n_tags, shuffle = True):
        """
        Returns an iterator on the batches of the dataset.
        """
        
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

            # Tengo que hacer ahora el padding tanto a las oraciones como a los chars y estaríamos listos para entrenar los
            # character embeddings.

            for j in range(len(batch_sentences)): # For each sentence in the batch
                
                cur_len = len(batch_sentences[j]) # First we get the current sentence len
                lens.append(cur_len)
                batch_data[j][:cur_len] = batch_sentences[j] # Then we fill the batch data with the sentence until the len position, the rest are padding words
                batch_labels[j][:cur_len] = batch_tags[j][:][:cur_len] # The same for labels
                
                
                for i, word in enumerate(batch_chars[j]):
                    batch_chars_data[j][i][:len(word)] = word
                
              
                
                

                # Probablemente deba hacer un for sobre todas las palabras que hayan en esa oración y realizar un padding o truncar según
                # el max len de los chars en este batch actual no?

                n = batch_max_len - cur_len # We obtain the number of padding tokens with the aim of filling the labels with 1 0 0 0......
                if batch_max_len!=cur_len: batch_labels[j][cur_len:] = self.pad_labels(n, n_tags)
            batch_data, batch_labels, batch_chars_data = torch.LongTensor(batch_data), torch.LongTensor(batch_labels), torch.LongTensor(batch_chars_data)

            if torch.cuda.is_available():
                batch_data, batch_labels, batch_chars_data = batch_data.cuda(), batch_labels.cuda(), batch_chars_data.cuda()

            
            yield batch_data, batch_labels, batch_chars_data, lens

