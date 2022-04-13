# Import
import torch
import torch.nn as nn
from encoder import EncoderCNN
from decoder import Decoder


# ResNet50 + Transformer auto encoder(AE) which hook encoder and decoder together


class AttentionAE(nn.Module):
    def __init__(self, num_layers, embed_size, heads, caption_vocab_size, device, dropout):
        super(AttentionAE, self).__init__()
        self.encoder = EncoderCNN(embed_size=embed_size)
        self.decoder = Decoder(num_layers=num_layers, embed_size=embed_size,
                               heads=heads, caption_vocab_size=caption_vocab_size,
                               device=device, dropout=dropout)
        self.device = device

    def forward(self, images, captions):
        features = self.encoder(images)  # return features shape:[N, embed_size] where N is batch_size
        outputs = self.decoder(captions, features)
        return outputs

    @torch.no_grad()
    def caption_image(self, image, vocabulary, max_length=50):
        outputs = [vocabulary.stoi["<SOS>"]]
        image_tensor = self.encoder(image)
        for i in range(max_length):
            trg_tensor = torch.LongTensor(outputs).unsqueeze(1).to(self.device)
            output = self.decoder(trg_tensor, image_tensor)
            best_guess = output.argmax(2)[-1, :].item()
            outputs.append(best_guess)

            if best_guess == vocabulary.stoi["<EOS>"]:
                break
        image_caption = [vocabulary.itos[idx] for idx in outputs]
        return image_caption[1:]
