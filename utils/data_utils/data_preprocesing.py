import codecs
import sys
sys.path.append('')
from utils.wl_utils.wl_utils import simplify_entity
from utils.data_utils.vectorizer import Vectorizer

def create_vocab_and_tags(sentences, labels,  dataset):
    vocab_file = codecs.open(f'data/{dataset}/vocab.txt', 'w', 'UTF-8')
    
    vocab = sorted(list(set([token for sent in sentences for token in sent])))
    for word in vocab: 
        vocab_file.write(word + '\n')
    vocab_file.close()

    tags_file = codecs.open(f'data/{dataset}/tags.txt', 'w', 'UTF-8')
    tags = sorted(list(set([value for sentence_labels in labels for label in sentence_labels for value in label])))
    for tag in tags:
        tags_file.write(tag+'\n')
    tags_file.close()

    char_file = codecs.open(f'data/{dataset}/char_vocab.txt', 'w', 'UTF-8')
    char_vocab = sorted(list(set([char for sent in sentences for token in sent for char in token])))
    print(f'Char vocab len: {len(char_vocab)}')
    for char in char_vocab: 
        char_file.write(char + '\n')
    char_file.close()
    return vocab, tags, char_vocab

def get_annotations(path):
    f = codecs.open(path, 'r', 'UTF-8').read()
    return f.split('\n\n')

def get_sentences_and_labels(annotations):
    sentences = [[line.split()[0] for line in anno.splitlines()] for anno in annotations]
    labels = [[list(map(lambda x: x, line.split()[1:])) for line in anno.splitlines()] for anno in annotations]
    return sentences, labels


def create_dataset_from_conll(train_path, test_path, dev_path, dataset):
    """
    Function used to transform dataset partitions to neural network inputs format.
    """

    train_annotations = get_annotations(train_path)
    test_annotations = get_annotations(test_path)
    dev_annotations = get_annotations(dev_path)

    train_sentences, train_labels = get_sentences_and_labels(train_annotations)
    test_sentences, test_labels =  get_sentences_and_labels(test_annotations)
    dev_sentences, dev_labels = get_sentences_and_labels(dev_annotations)

    vocab, tags, char_vocab = create_vocab_and_tags(train_sentences + dev_sentences + test_sentences, train_labels + dev_labels + test_labels, dataset)
    
    vectorizer = Vectorizer(vocab, tags, char_vocab)
    x_train, y_train, chars_train = vectorizer.transform_to_index(train_sentences, train_labels)
    x_val, y_val, chars_val = vectorizer.transform_to_index(dev_sentences, dev_labels)
    x_test, y_test, chars_test = vectorizer.transform_to_index(test_sentences, test_labels)
    dtrain = {'data': x_train, 'labels': y_train, 'chars': chars_train, 'size': len(x_train)}
    dval = {'data': x_val, 'labels': y_val, 'chars': chars_val, 'size': len(x_val)}
    dtest = {'data': x_test, 'labels': y_test, 'chars': chars_test, 'size': len(x_test)}
    dtrain_full = {'data': x_train + x_val, 'labels': y_train + y_val, 'chars': chars_train + chars_val, 'size': len(x_train) + len(x_val)}
    
    return dtrain, dval, dtest, dtrain_full, vocab, vectorizer.tag_to_index, vectorizer.index_to_word    