import torch
import torch.nn as nn
from torch.autograd import Variable
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
    self.lstm = nn.LSTM(embedding_dim, lstm_dim, batch_first=True, bidirectional=use_bilstm, num_layers=lstm_layers, dropout=lstm_dropout)
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
  

  def init_hidden(self, batch_size):
        h, c = (Variable(torch.zeros(self.num_layers * 2, batch_size, self.lstm_dim)),
                Variable(torch.zeros(self.num_layers * 2, batch_size, self.lstm_dim)))
        return h, c

  def forward(self, input, lens):
    h_0, c_0 = self.init_hidden(input.shape[0])
    emb = self.embedding_layer_dropout(self.embedding_layer(input)) # dim: batch_size x batch_max_len x embedding_dim
    packed_embedded = pack_padded_sequence(emb, lens, batch_first=True, enforce_sorted=False)
    lstm, _ = self.lstm(packed_embedded, (h_0, c_0))
    lstm_unpacked, _ = pad_packed_sequence(lstm, batch_first=True)            
    lstm_output = lstm_unpacked.reshape(-1, lstm_unpacked.shape[2])  # dim: batch_size*batch_max_len x lstm_hidden_dim
    output = self.output_layer_dropout(self.output_layer(lstm_output))  
    output = torch.sigmoid(output) 
    return output