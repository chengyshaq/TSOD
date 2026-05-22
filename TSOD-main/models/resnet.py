import torch
import torch.nn as nn
import torch.nn.functional as F
"""
    总结：
        这是一个基本的残差块（BasicBlock），包含两个卷积层、两个批量归一化层和一个残差连接（shortcut）。
        该模块实现了常见的残差网络（ResNet）中的基本块结构。
"""
# 解释：定义一个名为 BasicBlock 的类，继承自 nn.Module。nn.Module 是 PyTorch 中所有神经网络模块的基类，所有自定义的网络模块都需要继承它。
class BasicBlock(nn.Module):
    # 解释：初始化方法（构造函数），用于定义这个模块的各个层。in_planes 是输入的通道数，mid_planes 是中间层的通道数，out_planes 是输出的通道数，norm 是归一化层类型，stride 是步幅，bn_momentum 是批量归一化的动量。
    def __init__(self, in_planes, mid_planes, out_planes, norm, stride=1, bn_momentum=0.1):
        # 解释：调用父类（nn.Module）的初始化方法，确保 BasicBlock 类继承了 nn.Module 的所有功能。
        super(BasicBlock, self).__init__()
        # 解释：定义一个卷积层 conv1，输入通道数为 in_planes，输出通道数为 mid_planes，卷积核大小为 3x3，步幅为 stride，填充为 1，禁用偏置项。
        self.conv1 = nn.Conv2d(in_planes, mid_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        # 解释：定义一个批量归一化层 bn1，该层的输入通道数为 mid_planes，affine=False 表示该层没有可学习的参数，momentum=bn_momentum 设置批量归一化的动量。
        self.bn1 = norm(mid_planes, affine=False, momentum=bn_momentum)
        # 解释：定义第二个卷积层 conv2，输入通道数为 mid_planes，输出通道数为 out_planes，卷积核大小为 3x3，步幅为 1，填充为 1，禁用偏置项。
        self.conv2 = nn.Conv2d(mid_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        # 解释：定义第二个批量归一化层 bn2，输入通道数为 out_planes，affine=False 表示没有可学习的参数，momentum=bn_momentum 设置批量归一化的动量。
        self.bn2 = norm(out_planes, affine=False, momentum=bn_momentum)
        # 解释：初始化一个空的快捷连接（shortcut）层，使用 nn.Sequential() 表示按顺序组成的层，暂时为空。
        self.shortcut = nn.Sequential()
        # 解释：如果步幅 stride 不为 1，或者输入通道数 in_planes 不等于输出通道数 out_planes，则需要调整快捷连接的维度，以匹配卷积后的输出。
        if stride != 1 or in_planes != out_planes:
            # 解释：如果条件成立，定义一个快捷连接层，使用 1x1 的卷积层来调整输入的通道数，并通过归一化层调整输出通道数，确保与卷积层输出匹配。
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False), norm(out_planes))
    # 解释：定义前向传播方法，x 是输入张量，方法中定义了如何将输入通过网络进行计算。
    def forward(self, x):
        # 解释：通过 conv1 层进行卷积操作，然后通过批量归一化层 bn1 进行归一化处理。
        out = self.bn1(self.conv1(x))
        # 解释：对输出应用 ReLU 激活函数，增加非线性。
        out = F.relu(out)
        # 解释：通过 conv2 层进行卷积操作，然后通过批量归一化层 bn2 进行归一化处理。
        out = self.bn2(self.conv2(out))
        # 解释：将快捷连接的输出添加到当前的 out。这是残差连接的核心，目的是将输入 x 直接加到输出上，从而缓解梯度消失问题。
        out += self.shortcut(x)
        # 解释：对最终的输出应用 ReLU 激活函数。
        out = F.relu(out)
        return out



"""
    总结：
    该代码实现了一个基于残差网络（ResNet）的深度神经网络，能够进行图像分类，并且提供了特征提取和特征投影的功能。
"""
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, pooling='avgpool', norm=nn.BatchNorm2d, return_features=False, proj_dim=128, bn_momentum=0.1, feature_dim=512):
        super(ResNet, self).__init__()
        if pooling == 'avgpool':
            self.pooling = nn.AvgPool2d(4)
        elif pooling == 'maxpool':
            self.pooling = nn.MaxPool2d(4)
        else:
            raise Exception('Unsupported pooling: %s' % pooling)
        self.in_planes = 64
        self.return_features = return_features
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm(64, affine=False, momentum=bn_momentum)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, norm=norm, bn_momentum=bn_momentum)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, norm=norm, bn_momentum=bn_momentum)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, norm=norm, bn_momentum=bn_momentum)
        self.layer4 = self._make_layer(block, feature_dim, num_blocks[3], stride=2, norm=norm, bn_momentum=bn_momentum)
        self.linear = nn.Linear(feature_dim, num_classes)
        self.projection = nn.Sequential(nn.Linear(feature_dim, feature_dim), nn.ReLU(), nn.Linear(feature_dim, proj_dim))

    def _make_layer(self, block, planes, num_blocks, norm, stride, bn_momentum):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, planes, norm, stride, bn_momentum))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward_features(self, x):
        c1 = F.relu(self.bn1(self.conv1(x))) # (3,32,32)
        h1 = self.layer1(c1) # (64,32,32)
        h2 = self.layer2(h1) # (128,16,16)
        h3 = self.layer3(h2) # (256,8,8)
        h4 = self.layer4(h3) # (512,4,4)
        p4 = self.pooling(h4) # (512,1,1)
        p4 = p4.view(p4.size(0), -1) # (512)
        return p4

    def forward_classifier(self, p4):
        # (10)
        logits = self.linear(p4)
        return logits

    def forward(self, x):
        p4 = self.forward_features(x)
        logits = self.forward_classifier(p4)
        if self.return_features:
            return logits, p4
        else:
            return logits

    def forward_projection(self, p4):
        # (10)
        projected_f = self.projection(p4)
        return projected_f


"""
    总结：
        这段代码定义了一个名为 ResNet18 的函数，该函数用于创建一个 ResNet-18 模型。
        用户可以通过传递不同的参数来定制模型的类别数、池化方法、归一化方法等。
        函数内部调用了先前定义的 ResNet 类，并将 ResNet-18 特有的配置（如残差块的数量）传递给它。
"""
def ResNet18(num_classes=10, pooling='avgpool', norm=nn.BatchNorm2d, return_features=False, proj_dim=128, feature_dim=512):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, pooling=pooling, norm=norm, return_features=return_features, proj_dim=proj_dim, feature_dim=feature_dim)


if __name__ == '__main__':
    from thop import profile
    net = ResNet18(num_classes=10, return_features=True)
    x = torch.randn(1, 3, 32, 32)
    flops, params = profile(net, inputs=(x, ))
    y, features = net(x)
    print(y.size())
    print('GFLOPS: %.4f, model size: %.4fMB' % (flops/1e9, params/1e6))



'''
conv1.weight       
bn1.weight       
bn1.bias                  
layer1.0.conv1.weight                                                                                                                                     
layer1.0.bn1.weight     
layer1.0.bn1.bias    
layer1.0.conv2.weight
layer1.0.bn2.weight
layer1.0.bn2.bias    
layer1.1.conv1.weight
layer1.1.bn1.weight
layer1.1.bn1.bias
layer1.1.conv2.weight
layer1.1.bn2.weight
layer1.1.bn2.bias
layer2.0.conv1.weight
layer2.0.bn1.weight
layer2.0.bn1.bias  
layer2.0.conv2.weight
layer2.0.bn2.weight                
layer2.0.bn2.bias
layer2.0.shortcut.0.weight
layer2.0.shortcut.1.weight
layer2.0.shortcut.1.bias
layer2.1.conv1.weight                 
layer2.1.bn1.weight                                                          
layer2.1.bn1.bias                                                            
layer2.1.conv2.weight                                                        
layer2.1.bn2.weight                                                          
layer2.1.bn2.bias                                                                                                                                         
layer3.0.conv1.weight                 
layer3.0.bn1.weight                                                          
layer3.0.bn1.bias                                                            
layer3.0.conv2.weight                                                        
layer3.0.bn2.weight                                                          
layer3.0.bn2.bias                                                                                                                                         
layer3.0.shortcut.0.weight            
layer3.0.shortcut.1.weight         
layer3.0.shortcut.1.bias                                                     
layer3.1.conv1.weight                                                                                                                                     
layer3.1.bn1.weight
layer3.1.bn1.bias 
layer3.1.conv2.weight                
layer3.1.bn2.weight             
layer3.1.bn2.bias                    
layer4.0.conv1.weight          
layer4.0.bn1.weight                                                          
layer4.0.bn1.bias                                                                                                                                         
layer4.0.conv2.weight                                                        
layer4.0.bn2.weight
layer4.0.bn2.bias
layer4.0.shortcut.0.weight
layer4.0.shortcut.1.weight
layer4.0.shortcut.1.bias
layer4.1.conv1.weight
layer4.1.bn1.weight  
layer4.1.bn1.bias  
layer4.1.conv2.weight
layer4.1.bn2.weight  
layer4.1.bn2.bias  
linear.weight    
linear.bias          
projection.0.weight  
projection.0.bias  
projection.2.weight
projection.2.bias 
'''