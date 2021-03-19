import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
torch.manual_seed(0)

class BiLSTM(nn.Module):
  def __init__(self, vocab_size, n_tags, embedding_dim, lstm_dim, embedding_dropout, lstm_dropout, output_layer_dropout, lstm_layers, embedding_weights, use_bilstm, static_embeddings):
    super(BiLSTM, self).__init__()
    self.vocab_size = vocab_size
    self.n_tags = n_tags
    self.num_layers = lstm_layers
    self.lstm_dim = lstm_dim
    self.static_embeddings = static_embeddings
    self.embedding_layer = nn.Embedding(self.vocab_size+2, embedding_dim, padding_idx=0)
    self.embedding_layer_dropout = nn.Dropout(embedding_dropout)

    self.char_emb_dim = 25

    self.char_emb = nn.Embedding(
        num_embeddings = 123,
        embedding_dim = 25,
        padding_idx = 0
    )

    self.char_cnn = nn.Conv1d(
        in_channels= 25,
        out_channels= 25 * 5,
        kernel_size=3,
        groups=25  # different 1d conv for each embedding dim
    )
    self.cnn_dropout = nn.Dropout(0.5)
    self.lstm = nn.LSTM(embedding_dim, lstm_dim, batch_first=True, bidirectional=use_bilstm, num_layers=lstm_layers, dropout=lstm_dropout)
    self.attn = nn.MultiheadAttention(
            embed_dim = lstm_dim * 2,
            num_heads= 20,
            dropout = 0.25
        )
    self.output_layer = nn.Linear(lstm_dim * 2, self.n_tags)
    self.output_layer_dropout = nn.Dropout(output_layer_dropout)
    self.init_weights()
    self.init_embeddings(embedding_dim, embedding_weights)

  def save_state(self, path):
      torch.save(self.state_dict(), path)

  def load_state(self, path):
      self.load_state_dict(torch.load(path))

  def init_weights(self):
    for name, param in self.named_parameters():
      nn.init.normal_(param.data, mean=0, std=0.1)
    

  def init_embeddings(self, embedding_dim, pretrained=None):
    self.embedding_layer.weight.data[0] = torch.zeros(embedding_dim)
    if pretrained is not None:
          self.embedding_layer = nn.Embedding.from_pretrained(
              embeddings=torch.as_tensor(pretrained),
              padding_idx=0,
              freeze=self.static_embeddings
          )

  def count_parameters(self):
    return sum(p.numel() for p in self.parameters() if p.requires_grad)

  def forward(self, words, chars, lens):
    embedding_out = self.embedding_layer_dropout(self.embedding_layer(words)) # dim: batch_size x batch_max_len x embedding_dim

    char_emb_out = self.embedding_layer_dropout(self.char_emb(chars))
    batch_size, sent_len, word_len, char_emb_dim = char_emb_out.shape
    char_cnn_max_out = torch.zeros(batch_size, sent_len, self.char_cnn.out_channels)
    # for character embedding, we need to iterate over sentences
    for sent_i in range(sent_len):
        # sent_char_emb = [batch size, word length, char emb dim]
        sent_char_emb = char_emb_out[:, sent_i, :, :]  # get the character field of sent i
        # sent_char_emb_p = [batch size, char emb dim, word length]
        sent_char_emb_p = sent_char_emb.permute(0, 2, 1)  # the channel (char emb dim) has to be the last dimension
        # char_cnn_sent_out = [batch size, out channels * char emb dim, word length - kernel size + 1]
        char_cnn_sent_out = self.char_cnn(sent_char_emb_p)
        char_cnn_max_out[:, sent_i, :], _ = torch.max(char_cnn_sent_out, dim=2)  # max pooling over the word length dimension
    char_cnn = self.cnn_dropout(char_cnn_max_out)
    # concat word and char embedding
    # char_cnn_p = [sentence length, batch size, char emb dim * num filter]

    word_features = torch.cat((embedding_out, char_cnn), dim=2)


    packed_embedded = pack_padded_sequence(embedding_out, lens, batch_first=True, enforce_sorted=False)
    lstm, _ = self.lstm(packed_embedded)
    lstm_output, _ = pad_packed_sequence(lstm, batch_first=True)    

    key_padding_mask = torch.as_tensor(words == 0).permute(1, 0)
    attn_out, attn_weight = self.attn(lstm_output, lstm_output, lstm_output, key_padding_mask=key_padding_mask)  

    output = self.output_layer_dropout(self.output_layer(lstm_output))  
    output = torch.sigmoid(output) 
    return output