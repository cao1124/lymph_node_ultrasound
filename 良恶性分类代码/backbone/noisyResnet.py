import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights
from torchvision.models import resnet50


def Gaussian(x, gau_mean, gau_var, noise_str):
    #credit: http://blog.moephoto.tech/pytorch%e7%94%9f%e6%88%90%e5%8a%a0%e6%80%a7%e9%ab%98%e6%96%af%e7%99%bd%e5%99%aa%e5%a3%b0awgn/
    s1,s2,s3,s4 = x.shape
    means= gau_mean * torch.ones(s1,s2,s3,s4)
    stds = gau_var * torch.ones(s1,s2,s3,s4)
    gaussian_noise = (torch.normal(means, stds)).to(x.device)
    return noise_str * gaussian_noise + x

def Impulse(x,prob): #salt_and_pepper noise. strength has no meaning here, use the probality [0,1] to control
    #credit: https://blog.csdn.net/jzwong/article/details/109159682
    noise_tensor=torch.rand(x.size())
    salt=(torch.max(x.clone())).detach()
    pepper=(torch.min(x.clone())).detach()
    x_clone = x.clone()
    #x[noise_tensor<prob/2]=salt    #cause in-place graident computation error
    #x[noise_tensor>prob + prob/2]=pepper  #cause in-place graident computation error
    x_clone[noise_tensor<prob/2]=salt
    x_clone[noise_tensor>1-prob/2]=pepper
    return x_clone


class NoisyResnet(nn.Module):

    def __init__(self, num_classes, checkpoint=None,
                 noise_type=None, noise_strength=None, noise_prob=None,
                 noise_layer=None, sub_noise_layer=None, pretrain=False):
        super(NoisyResnet, self).__init__()

        self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT if pretrain else None)
        # self.noise_layer: 添加噪声的stage
        self.noise_layer = noise_layer
        # self.sub_noise_layer: 添加噪声的stage中的layer
        self.sub_noise_layer = sub_noise_layer
        # self.noise_str: 噪声强度
        self.noise_str = noise_strength
        self.noise_type = noise_type
        self.noise_prob = noise_prob

        if checkpoint:
            self.resenet.load_state_dict(torch.load(checkpoint))

        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
        self.resnet.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def apply_noise(self, x, x_copy, layer_block, sub_layer):
        if self.noise_type == 'linear':
            noise = (x_copy - x.detach())
            x = layer_block[sub_layer](x + self.noise_str * noise)
        elif self.noise_layer == 'gaussian':
            x = Gaussian(x, 0, 0.1, self.noise_str)
            x = layer_block[sub_layer](x)
        elif self.noise_type == 'impulse':
            x = Impulse(x, self.noise_prob)
            x = layer_block[sub_layer](x)
        return x

    def _process_layers(self, x, layer_block, layer_num):
        if self.noise_layer == layer_num:
            x_copy = x.detach()
            # [X1, X2, ..... XN] => [X2, X3, ..... XN, X1]
            x_copy = torch.cat((x_copy[1:], x_copy[0].unsqueeze(0)), dim=0)
            for sub_layer in range(len(layer_block)):
                if self.sub_noise_layer == sub_layer + 1:
                    x = self.apply_noise(x, x_copy, layer_block, sub_layer)
                else:
                    x = layer_block[sub_layer](x)
                    x_copy = x.detach()
                    x_copy = torch.cat((x_copy[1:], x_copy[0].unsqueeze(0)), dim=0)
        else:
            x = layer_block(x)
        return x

    def forward(self, x):
        # 降维处理
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        for layer in [1, 2, 3, 4]:
            x = self._process_layers(x, getattr(self.resnet, f'layer{layer}'), layer)

        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.resnet.fc(x)
        return x

def nnResnet_50_default(pretrained, num_classes):
    return NoisyResnet(num_classes=num_classes,
                       noise_layer=4,
                       sub_noise_layer=3,
                       noise_type='linear',
                       noise_strength=0.1,
                       noise_prob=0.1,
                       pretrain=pretrained)



