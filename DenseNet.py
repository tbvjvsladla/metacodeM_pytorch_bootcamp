import torch
import torch.nn as nn
from torchvision import models
from trans_weight import TransWeight

class Botteleneck(nn.Module):
    # Dense block의 레이어들이 출력하는 FeatureMap 크기: K = Growh_rate
    def __init__(self, in_ch, growth_rate): 
        super(Botteleneck, self).__init__()

        self.conv1x1 = nn.Sequential(
            nn.BatchNorm2d(in_ch),
            nn.ReLU(),
            nn.Conv2d(in_ch, out_channels=4*growth_rate, kernel_size=1, bias=False)
        )

        self.conv3x3 = nn.Sequential(
            nn.BatchNorm2d(4*growth_rate),
            nn.ReLU(),
            nn.Conv2d(in_channels=4*growth_rate, out_channels=growth_rate, 
                      kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x):
        identy = x
        x = self.conv1x1(x)
        x = self.conv3x3(x)

        x = torch.cat([identy, x], dim=1)
        return x
    
class TransitionModule(nn.Module):
    def __init__(self, in_ch):
        super(TransitionModule, self).__init__()
        # Composition Function을 적용하여 BN -> AF -> Conv순
        self.tr_layer = nn.Sequential(
            nn.BatchNorm2d(in_ch),
            nn.ReLU(),
            nn.Conv2d(in_ch, in_ch//2, kernel_size=1, stride=1, bias=False)
        )
        # Feature Map 크기 감소
        self.ave_pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.tr_layer(x)
        x = self.ave_pool(x)
        return x
    

class DenseBlock(nn.Module):
    def __init__(self, num_block, in_ch, growth_rate):
        super(DenseBlock, self).__init__()
        self.dense_ch = in_ch

        self.layers = nn.ModuleList() #리스트 처럼 레이어를 선언

        for _ in range(num_block):
            layer = Botteleneck(self.dense_ch, growth_rate)

            self.layers.append(layer) #선언한 레이어를 리스트에 삽입
            self.dense_ch += growth_rate

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x
    

class DenseNet(nn.Module):
    def __init__(self, block_list, growth_rate, n_classes=1000):
        super(DenseNet, self).__init__()

        assert len(block_list) == 4, "블럭 개수 확인 요"

        self.growth_rate = growth_rate

        self.stem = nn.Sequential( # DenseNet의 헤드 레이어
            nn.Conv2d(in_channels=3, out_channels=2*self.growth_rate,
                      kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(2*self.growth_rate),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        dense_ch = 2*self.growth_rate
        trans_ch_list = []
        
        self.dense_blocks = nn.ModuleDict()
        for i, num_block in enumerate(block_list):
            block_name = f'dense_block{i+1}'
            dense_block = DenseBlock(num_block, dense_ch, self.growth_rate)
            self.dense_blocks[block_name] = dense_block
            trans_ch_list.append(dense_block.dense_ch)
            dense_ch = dense_block.dense_ch // 2

        self.trans_blocks = nn.ModuleDict()
        for i, trans_ch in enumerate(trans_ch_list[:-1]):
            block_name = f'trans_block{i+1}'
            self.trans_blocks[block_name] = TransitionModule(trans_ch)

        last_module = nn.Sequential(
            nn.BatchNorm2d(trans_ch_list[-1]),
            nn.ReLU()
        )
        self.trans_blocks['trans_block4'] = last_module

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(trans_ch_list[-1], n_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        for i in range(4):
            dense_block = self.dense_blocks[f'dense_block{i+1}']
            trans_block = self.trans_blocks[f'trans_block{i+1}']
            x = dense_block(x)
            x = trans_block(x)
        x = self.classifier(x)
        return x
    

modul_list  = [
    ('stem', 'features[:4]'),
    ('dense_blocks.dense_block1', 'features.denseblock1'),
    ('trans_blocks.trans_block1', 'features.transition1'),
    ('dense_blocks.dense_block2', 'features.denseblock2'),
    ('trans_blocks.trans_block2', 'features.transition2'),
    ('dense_blocks.dense_block3', 'features.denseblock3'),
    ('trans_blocks.trans_block3', 'features.transition3'),
    ('dense_blocks.dense_block4', 'features.denseblock4'),
    ('trans_blocks.trans_block4', 'features.norm5'),
    ('classifier', 'classifier')
]

def DenseNet121(n_classes=1000, pretrained=False):
    cus_model = DenseNet(block_list=[6, 12, 24, 16], growth_rate=32, n_classes=n_classes)
    if pretrained:
        pr_model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        trans_weight = TransWeight(cus_model, pr_model, modul_list)
        trans_weight.transfer_parm()
    return cus_model


def DenseNet169(n_classes=1000, pretrained=False):
    cus_model = DenseNet(block_list=[6, 12, 32, 32], growth_rate=32, n_classes=n_classes)
    if pretrained:
        pr_model = models.densenet169(weights=models.DenseNet169_Weights.IMAGENET1K_V1)
        trans_weight = TransWeight(cus_model, pr_model, modul_list)
        trans_weight.transfer_parm()
    return cus_model


def DenseNet201(n_classes=1000, pretrained=False):
    cus_model = DenseNet(block_list=[6, 12, 48, 32], growth_rate=32, n_classes=n_classes)
    if pretrained:
        pr_model = models.densenet201(weights=models.DenseNet201_Weights.IMAGENET1K_V1)
        trans_weight = TransWeight(cus_model, pr_model, modul_list)
        trans_weight.transfer_parm()
    return cus_model