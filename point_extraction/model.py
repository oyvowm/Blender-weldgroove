from numpy.core.numeric import identity
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn.modules.batchnorm import BatchNorm1d
from torch.nn.modules.container import Sequential




class SimpleNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(2,8,2,2) # 320  ut
        self.bn1 = nn.BatchNorm1d(8)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(8,64,2,2) # 160 ut
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, 2, 2) # 80 ut
        self.bn3 = nn.BatchNorm1d(128)
        self.conv4 = nn.Conv1d(128, 256, 2, 2) # 40 ut
        self.bn4 = nn.BatchNorm1d(256)
        self.fc1 = nn.Linear(256*40, 640)
        self.fc2 = nn.Linear(640, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))

        x = self.conv2(x)
        x = F.relu(self.bn2(x))

        x = self.conv3(x)
        x = F.relu(self.bn3(x))

        x = self.conv4(x)
        x = F.relu(self.bn4(x))

        x = x.flatten(1)
        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        x = self.fc3(x)
        x = x.reshape(x.shape[0], -1, 2)
        return x



class SimpleNetwork2(nn.Module):
    """
    added more layers
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(2,8,2,2) # 320  ut
        self.bn1 = nn.BatchNorm1d(8)
        self.conv2 = nn.Conv1d(8,64,2,2) # 160 ut
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, 2, 2) # 80 ut
        self.bn3 = nn.BatchNorm1d(128)
        self.conv4 = nn.Conv1d(128, 256, 2, 2) # 40 ut
        self.bn4 = nn.BatchNorm1d(256)
        self.conv5 = nn.Conv1d(256, 512, 2, 2) # 20 ut
        self.bn5 = nn.BatchNorm1d(512)
        self.conv6 = nn.Conv1d(512, 512, 2, 2) # 10 ut
        self.bn6 = nn.BatchNorm1d(512)
        self.conv7 = nn.Conv1d(512, 512, 2, 2) # 5 ut
        self.bn7 = nn.BatchNorm1d(512)
        self.conv8 = nn.Conv1d(512, 512, 3, 1) # 3 ut
        self.bn8 = nn.BatchNorm1d(512)

        self.fc1 = nn.Linear(512*3, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))

        x = self.conv2(x)
        x = F.relu(self.bn2(x))

        x = self.conv3(x)
        x = F.relu(self.bn3(x))

        x = self.conv4(x)
        x = F.relu(self.bn4(x))

        x = self.conv5(x)
        x = F.relu(self.bn5(x))

        x = self.conv6(x)
        x = F.relu(self.bn6(x))
        #print(x.shape)
        x = self.conv7(x)
        x = F.relu(self.bn7(x))
        #print(x.shape)
        x = self.conv8(x)
        x = F.relu(self.bn8(x))

        #print(x.shape)

        x = x.flatten(1)
        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        x = self.fc3(x)
        x = x.reshape(x.shape[0], 2, -1)
        return x

class SimpleNetwork3(nn.Module):
    """
    identical to SimpleNetwork but with larger filters in the convolutional layers
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(2,8,5,2,2) # 320  ut
        self.bn1 = nn.BatchNorm1d(8)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(8,64,5,2,2) # 160 ut
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, 5, 2, 2) # 80 ut
        self.bn3 = nn.BatchNorm1d(128)
        self.conv4 = nn.Conv1d(128, 256, 5, 2, 2) # 40 ut
        self.bn4 = nn.BatchNorm1d(256)

        self.fc1 = nn.Linear(256*40, 640)
        self.fc2 = nn.Linear(640, 128)
        self.fc3 = nn.Linear(128, 10)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))

        x = self.conv2(x)
        x = F.relu(self.bn2(x))

        x = self.conv3(x)
        x = F.relu(self.bn3(x))

        x = self.conv4(x)
        x = F.relu(self.bn4(x))

        x = x.flatten(1)
        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        x = self.fc3(x)
        x = x.reshape(x.shape[0], 2, -1)
        return x

class FeedForwardNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1280, 640),
            nn.ReLU(),
            nn.Linear(640, 320),
            nn.ReLU(),
            nn.Linear(320, 100),
            nn.ReLU(),
            nn.Linear(100, 10)
        )

    def forward(self, x):
        x = x.flatten(1)
        x = self.layers(x)
        x = x.reshape(x.shape[0], 2, -1)
        return x




def conv1d2x2(in_chans, out_chans, kernel=3, stride=1, padding=1):
    return nn.Conv1d(in_chans, out_chans, kernel, stride, padding)

def conv1d1x1(in_chans, out_chans, kernel=1, stride=1):
    return nn.Conv1d(in_chans, out_chans, kernel, stride, bias=False)

class ResidualBlock(nn.Module):
    def __init__(self, inchans, outchans, kernel, padding, stride=1) -> None:
        super().__init__()
        self.conv1 = conv1d2x2(inchans, outchans, kernel=kernel, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm1d(outchans)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv1d2x2(outchans, outchans, kernel=kernel, padding=padding)
        self.bn2 = nn.BatchNorm1d(outchans)
        self.downsample = None

        if inchans != outchans or stride != 1:
            self.downsample = nn.Sequential(
                conv1d1x1(inchans, outchans, stride=stride),
                nn.BatchNorm1d(outchans)
            )
            self.conv1 = conv1d2x2(inchans, outchans, kernel=7, stride=2, padding=3)

    def forward(self, x):
        #print(x.shape)
        identity  = x
        #print(x.shape)
        out = self.conv1(x)
        #print(out.shape)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        #print(out.shape)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)

        return out

class ResidualNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(2, 16, 9, 2, 4) # 320  ut
        self.bn1 = nn.BatchNorm1d(16)
        self.relu = nn.ReLU(inplace=True)
        #self.avgpool = nn.AvgPool1d(2)
        self.pool = nn.MaxPool1d(2)

        self.layer1 = self.make_layer(16, 64, 6, 3, 1)
        self.layer2 = self.make_layer(64, 128, 7, 3, 1)
        self.layer3 = self.make_layer(128, 256, 7, 3, 1)
        self.layer4 = self.make_layer(256, 512, 5, 3, 1)

        self.fc1 = nn.Linear(512 * 20, 100)
        self.fc2 = nn.Linear(100, 10)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        
        for m in self.modules():
            if isinstance(m, ResidualBlock):
                nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def make_layer(self, inchans, outchans, num_blocks, kernel, padding):
        layers = []

        layers.append(ResidualBlock(inchans, outchans, 7, 3, 2))

        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(outchans, outchans, kernel, padding))
        
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        #x = self.pool(x)
        x = self.layer2(x)
        #x = self.pool(x)
        x = self.layer3(x)
        #x = self.pool(x)
        x = self.layer4(x)
        #x = self.pool(x)

        x = x.flatten(1)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = x.reshape(x.shape[0], -1, 2)
        return x


if __name__ == '__main__':

    x = torch.rand(10, 2 , 640)

    model = ResidualNetwork()
    #model = FeedForwardNet()
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print('num parameters:',pytorch_total_params)
    #print(model)
    a = model(x)
    print(a.shape)