# Import
import torch
import torch.nn as nn


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, features, captions):
        embeddings = self.dropout(self.embed(captions))
        embeddings = torch.cat((features.unsqueeze(0), embeddings), dim=0)
        hiddens, _ = self.lstm(embeddings)
        outputs = self.linear(hiddens)
        return outputs


# if __name__ == "__main__":
#     text_D = DecoderRNN(embed_size=16, hidden_size=16, vocab_size=100, num_layers=1)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     text_D.to(device)
#     x = torch.tensor([1, 5, 6, 4, 3, 9, 5, 2, 0],[1, 5, 6, 4, 3, 9, 5, 2, 0]).to(device)
#
#     trg = torch.tensor([1, 7, 4, 3, 5, 9, 2, 0, 1]).to(device)
#     out = text_D(x, trg)
#     print(out.shape)
