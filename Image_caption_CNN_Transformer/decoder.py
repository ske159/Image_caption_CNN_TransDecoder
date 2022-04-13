"""
src: (S, E)(S,E) for unbatched input, (S, N, E)(S,N,E) if batch_first=False or (N, S, E) if batch_first=True.
tgt: (T, E)(T,E) for unbatched input, (T, N, E)(T,N,E) if batch_first=False or (N, T, E) if batch_first=True.
src_mask: (S, S)(S,S).
tgt_mask: (T, T)(T,T).
memory_mask: (T, S)(T,S).
src_key_padding_mask: (S)(S) for unbatched input otherwise (N, S)(N,S).
tgt_key_padding_mask: (T)(T) for unbatched input otherwise (N, T)(N,T).
memory_key_padding_mask: (S)(S) for unbatched input otherwise (N, S)(N,S).
output: (T, E)(T,E) for unbatched input, (T, N, E)(T,N,E) if batch_first=False or (N, T, E) if batch_first=True.
output = transformer_model(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)
"""

import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, embed_size, heads, dropout, num_layers, caption_vocab_size, device):
        super(Decoder, self).__init__()
        self.device = device
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=embed_size, nhead=heads, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)
        self.captions_embedding = nn.Embedding(caption_vocab_size, embed_size)
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(embed_size, caption_vocab_size)

    def make_trg_mask(self, trg):
        trg_len = trg.shape[0]
        trg_mask = torch.tril(torch.ones((trg_len, trg_len)))
        return trg_mask.to(self.device)

    def forward(self, captions, images):
        caption_mask = self.make_trg_mask(captions)
        images = images.unsqueeze(0)
        captions = self.captions_embedding(captions)
        out = self.transformer_decoder(captions, images, tgt_mask=caption_mask)
        out = self.fc_out(out)
        return out


# if __name__ == '__main__':
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     embed_size = 512
#     heads = 8
#     dropout = 0.10
#     num_layers = 6
#     caption_vocab_size = 10
#     caption = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0], [1, 8, 7, 3, 4, 5, 6, 7, 2]]).to(device)
#
#     image = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0], [1, 5, 6, 2, 4, 7, 6, 2]]).to(device)
#     # tgt_key_padding_mask = 0
#     dcoder = Decoder(embed_size, heads, dropout, num_layers, caption_vocab_size, device=device).to(device)
#
#     out = dcoder(caption.transpose(0, 1), image.transpose(0, 1))
#     print(out.shape)
