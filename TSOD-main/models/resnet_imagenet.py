import torch
import torch.nn as nn
from torchvision.models.resnet import ResNet
from typing import Callable, Optional
from torch import Tensor


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width, affine=True)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width, affine=True)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet_ImageNet(ResNet):
    def __init__(self, block, num_blocks, num_classes=1000, return_features=False, proj=False):
        super(ResNet_ImageNet, self).__init__(block, num_blocks, num_classes=num_classes)
        self.return_features = return_features
        self.penultimate_layer_dim = self.fc.weight.shape[1]
        print('self.penultimate_layer_dim:', self.penultimate_layer_dim)
        if proj:
            self.projection = nn.Sequential(nn.Linear(self.penultimate_layer_dim, self.penultimate_layer_dim), nn.ReLU(), nn.Linear(self.penultimate_layer_dim, 128))
        self.bn1 = nn.BatchNorm2d(64, affine=False)
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                m.register_parameter("weight", None)
                m.register_parameter("bias", None)
                m.affine = False

    def forward_features(self, x):
        c1 = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        h1 = self.layer1(c1) # (64,32,32)
        h2 = self.layer2(h1) # (128,16,16)
        h3 = self.layer3(h2) # (256,8,8)
        h4 = self.layer4(h3) # (512,4,4)
        p4 = self.avgpool(h4) # (512,1,1)
        p4 = torch.flatten(p4, 1) # (512)
        return p4

    def forward_classifier(self, p4):
        logits = self.fc(p4) # (10)
        return logits

    def forward(self, x):
        p4 = self.forward_features(x)
        logits = self.forward_classifier(p4)

        if self.return_features:
            return logits, p4
        else:
            return logits

    def forward_projection(self, p4):
        projected_f = self.projection(p4) # (10)
        return projected_f



def ResNet50(num_classes=1000, return_features=False, proj=True):
    return ResNet_ImageNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, return_features=return_features, proj=proj)

if __name__ == '__main__':
    from thop import profile
    net = ResNet50(num_classes=10)
    x = torch.randn(1,3,224,224)
    flops, params = profile(net, inputs=(x, ))
    y = net(x)
    print(y.size())
    print('GFLOPS: %.4f, #params: %.4fM' % (flops/1e9, params/1e6)) # GFLOPS: 4.1095, #params: 23.5285M

    bn_parameter_number, fc_parameter_number, all_parameter_number = 0, 0, 0
    for name, p in net.named_parameters():
        if 'bn' in name:
            bn_parameter_number += p.numel()
        if 'fc' in name:
            fc_parameter_number += p.numel()
        if 'projection' not in name:
            all_parameter_number += p.numel()

    all_size = all_parameter_number * 4 /1e6 
    bn_size = bn_parameter_number * 4 /1e6 
    fc_size = fc_parameter_number * 4 /1e6 

    print('all_size: %s MB' % (all_size), 2*all_size)
    print('bn_size: %s MB' % (all_size+bn_size), bn_size)
    print('fc_size: %s MB' % (all_size+fc_size), fc_size)
    print('both_size: %s MB' % (all_size+bn_size+fc_size))

'''
module.conv1.weight         
module.bn1.weight           
module.bn1.bias             
module.layer1.0.conv1.weight
module.layer1.0.bn1.weight  
module.layer1.0.bn1.bias    
module.layer1.0.conv2.weight
module.layer1.0.bn2.weight  
module.layer1.0.bn2.bias    
module.layer1.0.conv3.weight
module.layer1.0.bn3.weight  
module.layer1.0.bn3.bias                                                                                                                                  
module.layer1.0.downsample.0.weight
module.layer1.0.downsample.1.weight
module.layer1.0.downsample.1.bias
module.layer1.1.conv1.weight
module.layer1.1.bn1.weight  
module.layer1.1.bn1.bias    
module.layer1.1.conv2.weight
module.layer1.1.bn2.weight  
module.layer1.1.bn2.bias           
module.layer1.1.conv3.weight       
module.layer1.1.bn3.weight  
module.layer1.1.bn3.bias    
module.layer1.2.conv1.weight
module.layer1.2.bn1.weight
module.layer1.2.bn1.bias    
module.layer1.2.conv2.weight
module.layer1.2.bn2.weight
module.layer1.2.bn2.bias
module.layer1.2.conv3.weight
module.layer1.2.bn3.weight
module.layer1.2.bn3.bias
module.layer2.0.conv1.weight       
module.layer2.0.bn1.weight       
module.layer2.0.bn1.bias    
module.layer2.0.conv2.weight
module.layer2.0.bn2.weight  
module.layer2.0.bn2.bias    
module.layer2.0.conv3.weight
module.layer2.0.bn3.weight         
module.layer2.0.bn3.bias           
module.layer2.0.downsample.0.weight
module.layer2.0.downsample.1.weight
module.layer2.0.downsample.1.bias
module.layer2.1.conv1.weight
module.layer2.1.bn1.weight  
module.layer2.1.bn1.bias    
module.layer2.1.conv2.weight
module.layer2.1.bn2.weight  
module.layer2.1.bn2.bias    
module.layer2.1.conv3.weight
module.layer2.1.bn3.weight  
module.layer2.1.bn3.bias    
module.layer2.2.conv1.weight
module.layer2.2.bn1.weight  
module.layer2.2.bn1.bias    
module.layer2.2.conv2.weight
module.layer2.2.bn2.weight  
module.layer2.2.bn2.bias    
module.layer2.2.conv3.weight                                                                                                                              
module.layer2.2.bn3.weight
module.layer2.2.bn3.bias    
module.layer2.3.conv1.weight
module.layer2.3.bn1.weight
module.layer2.3.bn1.bias    
module.layer2.3.conv2.weight
module.layer2.3.bn2.weight
module.layer2.3.bn2.bias    
module.layer2.3.conv3.weight       
module.layer2.3.bn3.weight         
module.layer2.3.bn3.bias
module.layer3.0.conv1.weight
module.layer3.0.bn1.weight
module.layer3.0.bn1.bias    
module.layer3.0.conv2.weight
module.layer3.0.bn2.weight
module.layer3.0.bn2.bias    
module.layer3.0.conv3.weight
module.layer3.0.bn3.weight
module.layer3.0.bn3.bias    
module.layer3.0.downsample.0.weight
module.layer3.0.downsample.1.weight
module.layer3.0.downsample.1.bias
module.layer3.1.conv1.weight
module.layer3.1.bn1.weight
module.layer3.1.bn1.bias    
module.layer3.1.conv2.weight
module.layer3.1.bn2.weight
module.layer3.1.bn2.bias           
module.layer3.1.conv3.weight       
module.layer3.1.bn3.weight       
module.layer3.1.bn3.bias    
module.layer3.2.conv1.weight
module.layer3.2.bn1.weight
module.layer3.2.bn1.bias    
module.layer3.2.conv2.weight
module.layer3.2.bn2.weight
module.layer3.2.bn2.bias    
module.layer3.2.conv3.weight
module.layer3.2.bn3.weight
module.layer3.2.bn3.bias    
module.layer3.3.conv1.weight
module.layer3.3.bn1.weight
module.layer3.3.bn1.bias    
module.layer3.3.conv2.weight
module.layer3.3.bn2.weight
module.layer3.3.bn2.bias    
module.layer3.3.conv3.weight
module.layer3.3.bn3.weight
module.layer3.3.bn3.bias
module.layer3.4.conv1.weight
module.layer3.4.bn1.weight
module.layer3.4.bn1.bias
module.layer3.4.conv2.weight
module.layer3.4.bn2.weight
module.layer3.4.bn2.bias  
module.layer3.4.conv3.weight
module.layer3.4.bn3.weight         
module.layer3.4.bn3.bias 
module.layer3.5.conv1.weight
module.layer3.5.bn1.weight
module.layer3.5.bn1.bias
module.layer3.5.conv2.weight
module.layer3.5.bn2.weight
module.layer3.5.bn2.bias
module.layer3.5.conv3.weight
module.layer3.5.bn3.weight
module.layer3.5.bn3.bias
module.layer4.0.conv1.weight
module.layer4.0.bn1.weight
module.layer4.0.bn1.bias
module.layer4.0.conv2.weight
module.layer4.0.bn2.weight
module.layer4.0.bn2.bias
module.layer4.0.conv3.weight
module.layer4.0.bn3.weight
module.layer4.0.bn3.bias
module.layer4.0.downsample.0.weight
module.layer4.0.downsample.1.weight
module.layer4.0.downsample.1.bias
module.layer4.1.conv1.weight
module.layer4.1.bn1.weight
module.layer4.1.bn1.bias
module.layer4.1.conv2.weight
module.layer4.1.bn2.weight
module.layer4.1.bn2.bias
module.layer4.1.conv3.weight
module.layer4.1.bn3.weight
module.layer4.1.bn3.bias
module.layer4.2.conv1.weight
module.layer4.2.bn1.weight
module.layer4.2.bn1.bias
module.layer4.2.conv2.weight
module.layer4.2.bn2.weight
module.layer4.2.bn2.bias
module.layer4.2.conv3.weight
module.layer4.2.bn3.weight
module.layer4.2.bn3.bias
module.fc.weight
module.fc.bias
module.projection.0.weight
module.projection.0.bias
module.projection.2.weight
module.projection.2.bias
'''