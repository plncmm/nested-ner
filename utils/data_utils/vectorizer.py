class Vectorizer():
    
    """
        Initializes a Vectorizer
        :param vocab: word vocabulary
        :param tags: entity types
        :param char_vocab: char vocabulary

    """
    
    def __init__(self, vocab, tags, char_vocab):
        self.vocab = vocab
        self.tags = tags
        self.char_vocab = char_vocab
        self.word_to_index = self.word_to_index()
        self.index_to_word = self.index_to_word()
        self.tag_to_index = self.tag_to_index()
        self.index_to_tag = self.index_to_tag()
        self.char_to_index = self.char_to_index()
        self.index_to_char = self.index_to_char()
        
    def char_to_index(self):
        char_to_index = {k: v+2 for (k, v) in zip(self.char_vocab, range(len(self.char_vocab)))}
        char_to_index["PAD"] = 0
        char_to_index["UNK"] = 1
        return char_to_index

    def index_to_char(self):
        index_to_char = {v+2: k for (k, v) in zip(self.char_vocab, range(len(self.char_vocab)))}
        index_to_char[0] = "PAD"
        index_to_char[1] = "UNK"
        return index_to_char    

    def word_to_index(self):
        word_to_index = {k: v+2 for (k, v) in zip(self.vocab, range(len(self.vocab)))}
        word_to_index["PAD"] = 0
        word_to_index["UNK"] = 1
        return word_to_index

    def index_to_word(self):
        index_to_word = {v+2: k for (k, v) in zip(self.vocab, range(len(self.vocab)))}
        index_to_word[0] = "PAD"
        index_to_word[1] = "UNK"
        return index_to_word
    
    def tag_to_index(self):
        tag_to_index = {k: v+1 for (k, v) in zip(self.tags, range(len(self.tags)))}
        tag_to_index["PAD"] = 0
        return tag_to_index

    def index_to_tag(self):
        index_to_tag = {v+1: k for (k, v) in zip(self.tags, range(len(self.tags)))}
        index_to_tag[0] = "PAD"
        return index_to_tag

    def sentence_to_index(self, sent):
        new_sentence = list(map(lambda x: self.word_to_index[x] if x in self.vocab else self.word_to_index['UNK'], sent))
        return new_sentence
    
    def index_to_sentence(self, sent):
        new_sentence = list(map(lambda x: self.index_to_word[x], sent))
        return new_sentence

    def char_sentence_to_index(self, sent):
        new_char_sentence = []
        for word in sent:
            word_array = []
            for char in word:
                if char in self.char_vocab:
                    word_array.append(self.char_to_index[char])
                else:
                    word_array.append(self.char_to_index['UNK'])
            new_char_sentence.append(word_array)
        return new_char_sentence
    
    def index_to_char_sentence(self, sent):
        new_char_sentence = []
        for word in sent:
            word_array = []
            for char in word:
                word_array.append(self.index_to_char[char])
            new_char_sentence.append(word_array)
        return new_char_sentence

    

    def labels_to_index(self, tags):
        total_tags = len(self.tags)+1
        sentence_labels = [0]*total_tags
        for tag in tags:
            sentence_labels[self.tag_to_index[tag]]=1
        return sentence_labels

    def index_to_labels(self, tags):
        new_labels = list(map(lambda x: self.index_to_tag[x], tags))
        return new_labels

    def transform_to_index(self, sents, labels):
        new_sents = [self.sentence_to_index(sent) for sent in sents]
        char_sents = [self.char_sentence_to_index(sent) for sent in sents]
        new_labels = [[self.labels_to_index(tag) for tag in tags] for tags in labels]
        return new_sents, new_labels, char_sents
