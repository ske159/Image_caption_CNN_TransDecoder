# Import
import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import functional as F


class EncoderCNN(nn.Module):
    def __init__(self, embed_size, train_CNN=False):
        super(EncoderCNN, self).__init__()
        self.train_CNN = train_CNN
        self.resnet50 = models.resnet50(pretrained=True)
        self.resnet50.fc = nn.Linear(in_features=256, out_features=embed_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

        # list(resnet50.children()) consists of following modules
        #   [0] = Conv2d, [1] = BatchNorm2d, [2] = ReLU, [3] = MaxPool2d,
        #   [4] = Sequential(Bottleneck...),
        #   [5] = Sequential(Bottleneck...),
        #   [6] = Sequential(Bottleneck...),
        #   [7] = Sequential(Bottleneck...),
        #   [8] = AvgPool2d, [9] = Linear

        children = list(self.resnet50.children())
        self.layer1 = nn.Sequential(*children[:4])
        self.layer2 = children[4]
        self.layer3 = children[5]
        self.layer4 = children[6]
        self.layer5 = children[7]

        # 侧向引出的下采样分段特征
        num_feature_out = 256
        in_channel = 256
        self.lateral_c2 = nn.Conv2d(in_channels=in_channel, out_channels=num_feature_out, kernel_size=1)
        self.lateral_c3 = nn.Conv2d(in_channels=in_channel * 2, out_channels=num_feature_out, kernel_size=1)
        self.lateral_c4 = nn.Conv2d(in_channels=in_channel * 4, out_channels=num_feature_out, kernel_size=1)
        self.lateral_c5 = nn.Conv2d(in_channels=in_channel * 8, out_channels=num_feature_out, kernel_size=1)

        # 去混淆卷积 Reduce the aliasing effect
        self.de_aliasing_p2 = nn.Conv2d(in_channels=num_feature_out, out_channels=num_feature_out, kernel_size=3, padding=1)
        self.de_aliasing_p3 = nn.Conv2d(in_channels=num_feature_out, out_channels=num_feature_out, kernel_size=3, padding=1)
        self.de_aliasing_p4 = nn.Conv2d(in_channels=num_feature_out, out_channels=num_feature_out, kernel_size=3, padding=1)

    def forward(self, images):






        # 下采样方向，不同layer的卷积运算结果
        c1 = self.layer1(images)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        c5 = self.layer5(c4)

        # 上采样方向计算pyramid，层间特征融合，feature fusion form different layer
        p5 = self.lateral_c5(c5)
        p4 = self.lateral_c4(c4) + F.interpolate(input=p5, size=(c4.shape[2], c4.shape[3]), mode='nearest')
        p3 = self.lateral_c3(c3) + F.interpolate(input=p4, size=(c3.shape[2], c3.shape[3]), mode='nearest')
        p2 = self.lateral_c2(c2) + F.interpolate(input=p3, size=(c2.shape[2], c2.shape[3]), mode='nearest')

        # 去混淆
        p2_de_aliasing = self.de_aliasing_p2(p2)

        # mapping to embed_size
        x = self.resnet50.avgpool(p2_de_aliasing)
        x = x.reshape(x.shape[0], -1)
        features = self.resnet50.fc(x)
        features = self.dropout(self.relu(features))
        return features
