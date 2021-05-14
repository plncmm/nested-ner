import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, FlairEmbeddings, CharacterEmbeddings, TransformerWordEmbeddings, BertEmbeddings
from typing import List
from functools import reduce
from flair.data import Sentence, Token
from gensim.models import KeyedVectors
import re
import numpy as np
torch.manual_seed(0)


class SequenceMultilabelingTagger(nn.Module):
    """Sequence Multilabeling for Nested NER. 2021"""
    
    def __init__(
                self, 
                corpus, 
                vectorizer, 
                n_tags, 
                embed_path, 
                use_char_embeddings,
                use_flair_embeddings,
                use_bert_embeddings,
                use_pretrained_embeddings,
                embedding_dropout, 
                lstm_dim, 
                lstm_layers,
                lstm_dropout,
                use_bilstm, 
                use_attn_layer,
                attn_heads,
                attn_dropout,
                device
    ):

        """
        Initializes a SequenceMultilabelingTagger for Nested Named Entity Recognition.
        :param corpus: corpus vocabulary
        :param vectorizer: vectorizer object used to transform index to word
        :param n_tags: number of entity types
        :param embed_path: pre-trained embedding path
        :param use_char_embeddings: if True use RNN Character Embeddings
        :param use_flair_embeddings: if True use flair contextualized embeddings
        :param use_bert_embeddings: if True use bert contextualized embeddings
        :param use_pretrained_embeddings: if True use pre-trained word embeddings
        :param embedding_dropout: embedding dropout probability
        :param lstm_dim: number of hidden states
        :param lstm_layers: number of LSTM layers
        :param lstm_dropout: lstm dropout probability
        :param use_bilstm: if True use BiLSTM, otherwise use simple LSTM
        :param use_attn_layer: if True add an attention layer
        :param attn_heads: number of heads in attention layer
        :param attn_dropout: attention dropout probability
        :param device: 'cuda' or 'cpu'
        """


        super(SequenceMultilabelingTagger, self).__init__()
        self.device = device
        self.vectorizer = vectorizer
        self.use_pretrained_embeddings: bool = use_pretrained_embeddings
        self.use_char_embeddings: bool = use_char_embeddings
        self.use_flair_embeddings: bool = use_flair_embeddings
        self.use_bert_embeddings: bool = use_bert_embeddings
        self.use_attn_layer: bool = use_attn_layer

        # Embeddings: Pre-trained Word Embeddings/Character Embeddings/Contextualized Embeddings
        embedding_types: List[TokenEmbeddings] = []

        if use_pretrained_embeddings: 
            embedding_types.append(W2vWordEmbeddings(embed_path))

        if use_char_embeddings:
            embedding_types.append(CharacterEmbeddings())
        
        if use_bert_embeddings:

            if corpus == 'genia':
                embedding_types.append(TransformerWordEmbeddings(
                    'bert-large-cased', 
                    layers = 'all', 
                    layer_mean = True, 
                    subtoken_pooling = 'mean'
                ))

            if corpus == 'wl':
                embedding_types.append(TransformerWordEmbeddings(
                    'dccuchile/bert-base-spanish-wwm-cased', 
                    layers = 'all', 
                    layer_mean = True, 
                    subtoken_pooling = 'mean'
                ))

        if use_flair_embeddings:
            if corpus == 'genia':
                embedding_types.append(FlairEmbeddings('news-forward'))
                embedding_types.append(FlairEmbeddings('news-backward'))
            if corpus == 'wl':
                embedding_types.append(FlairEmbeddings('spanish-forward'))
                embedding_types.append(FlairEmbeddings('spanish-backward'))

        self.flair_embeddings: StackedEmbeddings = StackedEmbeddings(embeddings = embedding_types)
        

        self.embedding_dropout = nn.Dropout(embedding_dropout)

        # BiLSTM.
        self.lstm = nn.LSTM(
            self.flair_embeddings.embedding_length, 
            lstm_dim, 
            batch_first = True, 
            bidirectional = use_bilstm, 
            num_layers = lstm_layers, 
            dropout = lstm_dropout
        )
        
        # Attention Layer.
        self.attn = nn.MultiheadAttention(
            embed_dim = lstm_dim * 2,
            num_heads= attn_heads,
            dropout = attn_dropout
        )

        # Flatten layer. 
        self.output_layer = nn.Linear(lstm_dim * 2, n_tags)
        self.init_weights()

    def get_flair_embeddings(self, sents, vectorizer, flair_embeddings):
        new_words = []
        sentences = []
        for sent in sents:
            sentence = Sentence()
            tokens: List[Token] = [Token(vectorizer.index_to_word[token.item()]) for token in sent]
            if len(tokens) != len(sent):
                raise ValueError("tokens length does not match sent length")
            sentence.tokens = tokens
            flair_embeddings.embed(sentence)
            sent_embs = []
            for token in sentence:
              sent_embs.append(token.embedding.cpu().detach().numpy())
            new_words.append(sent_embs)
        return torch.tensor(new_words)

    def forward(self, words, chars, lens):
        # Embeddings (Batch_size, Batch_sent_max_len, Embedding_dim)
        embeddings = self.get_flair_embeddings(words, self.vectorizer, self.flair_embeddings).to(self.device)
        embeddings = self.embedding_dropout(embeddings)

        # BiLSTM (Batch_size, Batch_sent_max_len, Hidden units)
        packed_embedded = pack_padded_sequence(embeddings, lens, batch_first=True, enforce_sorted=False)
        lstm, _ = self.lstm(packed_embedded)
        encoder_output, _ = pad_packed_sequence(lstm, batch_first=True)    

        # Attention layer (Batch_size, Batch_sent_max_len, _)
        if self.use_attn_layer:
            key_padding_mask = torch.as_tensor(words == 0).permute(1, 0)
            encoder_output, _ = self.attn(
                encoder_output, 
                encoder_output, 
                encoder_output, 
                key_padding_mask = key_padding_mask
            )  

        # Flatten layer (Batch_size, Batch_sent_max_len, tags)
        output = self.output_layer(encoder_output)  
        output = torch.sigmoid(output) 
        return output

    def save_state(self, path):
        torch.save(self.state_dict(), path)

    def load_state(self, path):
        self.load_state_dict(torch.load(path))

    def init_weights(self):
        for name, param in self.named_parameters():
          nn.init.normal_(param.data, mean=0, std=0.1)
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

  
class W2vWordEmbeddings(TokenEmbeddings):

    def __init__(self, embeddings):
        self.name = embeddings
        self.static_embeddings = False
        self.precomputed_word_embeddings = KeyedVectors.load_word2vec_format(embeddings, binary=False)
        self.__embedding_length: int = self.precomputed_word_embeddings.vector_size
        super().__init__()

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:
        for i, sentence in enumerate(sentences):
            for token, token_idx in zip(sentence.tokens, range(len(sentence.tokens))):
                token: Token = token
                if token.text in self.precomputed_word_embeddings:
                    word_embedding = self.precomputed_word_embeddings[token.text]
                elif token.text.lower() in self.precomputed_word_embeddings:
                    word_embedding = self.precomputed_word_embeddings[token.text.lower()]
                elif re.sub('\d', '#', token.text.lower()) in self.precomputed_word_embeddings:
                    word_embedding = self.precomputed_word_embeddings[re.sub('\d', '#', token.text.lower())]
                elif re.sub('\d', '0', token.text.lower()) in self.precomputed_word_embeddings:
                    word_embedding = self.precomputed_word_embeddings[re.sub('\d', '0', token.text.lower())]
                else:
                    word_embedding = np.zeros(self.embedding_length, dtype='float')
                word_embedding = torch.FloatTensor(word_embedding)
                token.set_embedding(self.name, word_embedding)
        return sentences