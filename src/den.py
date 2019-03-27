import torch
import torch.nn as nn
from torchvision.models.resnet import Bottleneck, conv1x1, conv3x3

INPUT_HEIGHT, INPUT_WIDTH = 224, 224
INPUT_DIM = (3, INPUT_HEIGHT, INPUT_WIDTH)

TARGET_HEIGHT, TARGET_WIDTH = 25, 32
TARGET_DIM = (TARGET_HEIGHT, TARGET_WIDTH)
TARGET_SIZE = TARGET_HEIGHT * TARGET_WIDTH

def conv1x3(in_planes, out_planes):
    """1x3 convolution with width padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=(1, 3),
                     padding=(0, 1), bias=False)


def conv3x1(in_planes, out_planes):
    """3x1 convolution with height padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=(3, 1),
                     padding=(1, 0), bias=False)


class ModifiedBottleneck(nn.Module):
    """A modified basic block of the ResNet architecture.

    Contains additional 1x3 and 3x1 convolutions which extracts features
    from the final output of the block.
    """

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, special=64):
        super().__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        self.conv1x3 = conv1x3(planes * self.expansion, special)
        self.conv3x1 = conv3x1(special, special)

    def forward(self, x):
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

        aux = self.conv1x3(out)
        aux = self.relu(aux)
        aux = self.conv3x1(aux)
        aux = self.relu(aux)
        self.aux = aux.view(aux.shape[0], -1)

        return out

class DepthEstimationNetwork(nn.Module):
    """Depth Estimation Network based on a modified ResNet-152 network.

    Expects input in the standard ImageNet shape (224x224 color images).
    Output is a 25x32 depth map.
    """

    def __init__(self):
        super().__init__()

        block = Bottleneck
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, 3)
        self.layer2 = self._make_layer(block, 128, 8, stride=2)
        # 20 normal + 16 modified
        self.layer3 = self._make_layer(block, 256, 36, stride=2)
        for i in range(16):
            self.layer3[20 +
                        i] = ModifiedBottleneck(self.inplanes, 256, special=8)
        self.layer4 = self._make_layer(ModifiedBottleneck, 512, 3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(36544, TARGET_SIZE)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        aux3_outs = (block.aux for block in self.layer3[20:])
        aux4_outs = (block.aux for block in self.layer4)

        x = torch.cat([*aux3_outs, *aux4_outs, x], dim=1)
        x = self.fc(x)

        return x

if __name__ == '__main__':
    # Test model constructor
    model = DepthEstimationNetwork()
    model.eval()

    # Run a forward pass to ensure all dimensions match
    random_image = torch.randn(INPUT_DIM)
    batch = random_image.unsqueeze(0)

    with torch.no_grad():
        model(batch)
