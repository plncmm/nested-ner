import codecs
import random
import math

import sys
sys.path.append('')
from utils.utils import simplify_entity
from dataset.vectorizer import Vectorizer


def split_sentences(sentences, labels, train_size, val_size, test_size):
    n_examples = len(sentences)
    print(f'The total number of sentences is: {n_examples}')
    n_train = math.floor(n_examples*train_size)
    n_val =  math.floor(n_examples*val_size)
    n_test =  n_examples - n_train - n_val 
    #c = list(zip(sentences, labels))
    #random.shuffle(c)
    #sentences, labels = zip(*c)
    x_train, y_train = sentences[:n_train], labels[:n_train]
    x_val, y_val = sentences[n_train: n_train+n_val], labels[n_train: n_train+n_val]
    x_test, y_test = sentences[-n_test:], labels[-n_test:]
    return x_train, y_train, x_val, y_val, x_test, y_test

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
            
def create_dataset(conll_path):
    f = codecs.open(conll_path, 'r', 'UTF-8')
    text = f.read()
    annotations = text.split('\n\n')[:-1]
    sentences = [[line.split(' ')[:-1][0] for line in anno.splitlines()] for anno in annotations]
    labels = [[list(map(lambda x: simplify_entity(x), line.split(' ')[1:])) for line in anno.splitlines()] for anno in annotations]
    labels = fix(labels)
    vocab = list(set([token for sent in sentences for token in sent]))
    tags = list(set([value for sentence_labels in labels for label in sentence_labels for value in label]))
    vectorizer = Vectorizer(vocab, tags)
    x_train, y_train, x_val, y_val, x_test, y_test = split_sentences(sentences, labels, 0.8, 0.1, 0.1)
    print(f"The total number of tags is: {len(tags)}")
    print(f"The vocabulary size is: {len(vocab)}")
    x_train, y_train = vectorizer.transform_to_index(x_train, y_train)
    x_val, y_val = vectorizer.transform_to_index(x_val, y_val)
    x_test, y_test = vectorizer.transform_to_index(x_test, y_test)
    dtrain = {'data': x_train, 'labels': y_train, 'size': len(x_train)}
    dval = {'data': x_val, 'labels': y_val, 'size': len(x_val)}
    dtest = {'data': x_test, 'labels': y_test, 'size': len(x_test)}
    return dtrain, dval, dtest, vocab, vectorizer.tag_to_index, vectorizer.index_to_word
    