import codecs
import random
import math
import sys
sys.path.append('')
from utils.dataset_utils import simplify_entity
from dataset.vectorizer import Vectorizer
from sklearn import model_selection
import numpy as np
random.seed(0)
np.random.seed(0)


def fix(labels):
    new_labels = labels
    for i, sentence in enumerate(new_labels):
        if len(sentence)==1:
            continue
        previus_token_labels = sentence[0]
        for j, token in enumerate(sentence[1:]):
            actual_token_entities = token
            for tag in actual_token_entities:
                if tag.startswith('I-'):
                    if f'B-{tag[2:]}' not in previus_token_labels and f'I-{tag[2:]}' not in previus_token_labels:
                        new_labels[i][j+1].remove(tag)
                        new_labels[i][j+1].append(f'B-{tag[2:]}')
                        continue
            previus_token_labels = token
    return new_labels
            
def create_dataset(conll_path, train_size, dev_size, test_size):
    f = codecs.open(conll_path, 'r', 'UTF-8')
    text = f.read()
    annotations = text.split('\n\n')[:-1]
    sentences = [[line.split(' ')[:-1][0] for line in anno.splitlines()] for anno in annotations]
    labels = [[list(map(lambda x: simplify_entity(x), line.split(' ')[1:])) for line in anno.splitlines()] for anno in annotations]
    labels = fix(labels)
    vocab_file = codecs.open('vocab.txt', 'w', 'UTF-8')
    vocab = sorted(list(set([token for sent in sentences for token in sent])))
    for word in vocab: 
        vocab_file.write(word + '\n')
    vocab_file.close()
    

    tags_file = codecs.open('tags.txt', 'w', 'UTF-8')
    tags = sorted(list(set([value for sentence_labels in labels for label in sentence_labels for value in label])))
    for tag in tags:
        tags_file.write(tag+'\n')
    tags_file.close()
    
    vectorizer = Vectorizer(vocab, tags)
    (x_train, x_test, y_train, y_test) = model_selection.train_test_split(sentences, labels, random_state=1, test_size = test_size)
    (x_train, x_val, y_train, y_val) = model_selection.train_test_split(x_train, y_train, random_state=1, test_size = dev_size)
    x_train, y_train = vectorizer.transform_to_index(x_train, y_train)
    x_val, y_val = vectorizer.transform_to_index(x_val, y_val)
    x_test, y_test = vectorizer.transform_to_index(x_test, y_test)
    dtrain = {'data': x_train, 'labels': y_train, 'size': len(x_train)}
    dval = {'data': x_val, 'labels': y_val, 'size': len(x_val)}
    dtest = {'data': x_test, 'labels': y_test, 'size': len(x_test)}
    return dtrain, dval, dtest, vocab, vectorizer.tag_to_index, vectorizer.index_to_word
    