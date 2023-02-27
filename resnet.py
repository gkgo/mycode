# This ResNet network was designed following the practice of the following papers:
# TADAM: Task dependent adaptive metric for improved few-shot learning (Oreshkin et al., in NIPS 2018) and
# A Simple Neural Attentive Meta-Learner (Mishra et al., in ICLR 2018).
import torch.nn.functional as F
import torch
import torch.nn as nn

def featureL2Norm(feature):
    epsilon = 1e-6
    norm = torch.pow(torch.sum(torch.pow(feature, 2), 1) + epsilon, 0.5).unsqueeze(1).expand_as(feature)
    return torch.div(feature, norm)

class mySelfCorrelationComputation(nn.Module):
    def __init__(self, kernel_size=(5, 5), padding=2):
        super(mySelfCorrelationComputation, self).__init__()
        self.kernel_size = kernel_size
        self.unfold = nn.Unfold(kernel_size=kernel_size, padding=padding)
        self.relu = nn.ReLU(inplace=False)
        self.bn1 = nn.BatchNorm2d(640)

        self.conv1x1_in = nn.Sequential(nn.Conv2d(640, 64, kernel_size=1, bias=False, padding=0),
                                        nn.BatchNorm2d(64),
                                        nn.ReLU(inplace=True))
        self.embeddingFea = nn.Sequential(nn.Conv2d(1664, 640,
                                                     kernel_size=1, bias=False, padding=0),
                                           nn.BatchNorm2d(640),
                                           nn.ReLU(inplace=True))
        self.conv1x1_out = nn.Sequential(
            nn.Conv2d(640, 640, kernel_size=1, bias=False, padding=0),
            nn.BatchNorm2d(640))
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv1x1_in(x)
        b, c, h, w = x.shape
        x0 = self.relu(x)
        x = x0
        x = F.normalize(x, dim=1, p=2)
        identity = x
        x = self.unfold(x)
        x = x.view(b, c, self.kernel_size[0], self.kernel_size[1], h, w)  # b, c, u, v, h, w
        x = x * identity.unsqueeze(2).unsqueeze(2)
        x = x.view(b, -1, h, w)
        feature_gs = featureL2Norm(x)

        # concatenate
        feature_cat = torch.cat([identity, feature_gs], 1)

        # embed
        feature_embd = self.embeddingFea(feature_cat)
        feature_embd = self.conv1x1_out(feature_embd)
        feature_embd = self.dropout(feature_embd)
        return feature_embd

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def gaussian_normalize( x, dim, eps=1e-05):
    x_mean = torch.mean(x, dim=dim, keepdim=True)
    x_var = torch.var(x, dim=dim, keepdim=True)  # 求dim上的方差
    x = torch.div(x - x_mean, torch.sqrt(x_var + eps))  # （x原始-x平均）/根号下x_var
    return x



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(0.1)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv3x3(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.maxpool = nn.MaxPool2d(stride)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        out = self.maxpool(out)

        return out
def normalize_feature(x):
    return x - x.mean(1).unsqueeze(1)  # x-x.mean(1)行求平均值并在channal维上增加一个维度

class ResNet(nn.Module):

    def __init__(self, block, num_classes=100, zero_init_residual=False):
        self.inplanes = 3
        super(ResNet, self).__init__()

        self.layer1 = self._make_layer(block, 64, stride=2)
        self.layer2 = self._make_layer(block, 160, stride=2)
        self.layer3 = self._make_layer(block, 320, stride=2)
        self.layer4 = self._make_layer(block, 640, stride=2)
        self.scr_module = mySelfCorrelationComputation(kernel_size=(5,5), padding=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(640, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)

        x = self.layer2(x)

        x = self.layer3(x)

        x = self.layer4(x)
#____________________________________
#         identity = x

#         x = self.scr_module(x)


#         x = x + identity

#         x = F.relu(x, inplace=True)

#         b, c, h, w = x.shape
#         x = normalize_feature(x)

#         y = F.normalize(x, p=2, dim=1, eps=1e-8)

#         d_s = y.view(b, c, -1)
#         d_s = gaussian_normalize(d_s, dim=2)

#         d_s = F.softmax(d_s /2, dim=2)
#         d_s = d_s.view(b,c,h, w)

#         x1 = d_s + x

#         x = x1.mean(dim=[-1, -2])
#         x = self.fc(x)
#_______________________________________________________
        identity = x
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        
        # Self-correlation module
        feat_corr = self.scr_module(identity)
        feat_corr = feat_corr.mean([2, 3])
        feat_corr = self.fc(feat_corr)
        out = x + feat_corr

        return out

def resnet12():
    return ResNet(BasicBlock)
