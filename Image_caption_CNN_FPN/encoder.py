# Import
import torch.nn as nn
import torchvision.models as models
import torch



class EncoderCNN(nn.Module):
    def __init__(self, embed_size, train_CNN=False):
        super(EncoderCNN, self).__init__()
        self.train_CNN = train_CNN
        self.resnet50 = models.resnet50(pretrained=True)
        self.resnet50.fc = nn.Linear(in_features=2048, out_features=embed_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, images):
        features = self.dropout(self.relu(self.resnet50(images)))
        return features


# def test():
#     net = EncoderCNN(256)
#     x = torch.randn(1, 3, 224, 224)
#     y = net(x).to('cuda')
#     print(y.shape)
#
#
# test()
