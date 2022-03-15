from numpy.core.numeric import identity
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn.modules.batchnorm import BatchNorm1d
from torch.nn.modules.container import Sequential
from dataset import LaserPointDataset
from torch.utils.data import DataLoader

def conv1d2x2(in_chans, out_chans, kernel=3, stride=1, padding=1):
    return nn.Conv1d(in_chans, out_chans, kernel, stride, padding)

def conv1d1x1(in_chans, out_chans, kernel=1, stride=1):
    return nn.Conv1d(in_chans, out_chans, kernel, stride, bias=False)

class ResidualBlock(nn.Module):
    def __init__(self, inchans, outchans, kernel=3, padding=1, stride=1) -> None:
        super().__init__()
        self.conv1 = conv1d2x2(inchans, outchans)
        self.bn1 = nn.BatchNorm1d(outchans)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv1d2x2(outchans, outchans)
        self.bn2 = nn.BatchNorm1d(outchans)
        self.downsample = None

        if inchans != outchans or stride != 1:
            self.downsample = nn.Sequential(
                conv1d1x1(inchans, outchans, stride=stride),
                nn.BatchNorm1d(outchans)
            )
            self.conv1 = conv1d2x2(inchans, outchans, kernel=kernel, stride=stride, padding=padding)

    def forward(self, x):
        #print(x.shape)
        identity = x
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
        #self.conv1 = nn.Conv1d(2, 16, 21, 5, 10) # 128  ut
        #self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        #self.avgpool = nn.AvgPool1d(2)
        self.pool = nn.MaxPool1d(2)

        self.d1 = self.make_layer(2, 16, 3, 21, 10, 5) # 128
        self.d2 = self.make_layer(16, 32, 4, 11, 5, 1) # 64

        self.layer1 = self.make_layer(32, 64, 4, stride=1) # 32
        self.layer2 = self.make_layer(64, 128, 5, stride=1) # 16
        self.layer3 = self.make_layer(128, 256, 4, stride=1) # 8
        self.layer4 = self.make_layer(256, 512, 2, stride=1) # 4
        #self.holistic_layer = nn.Conv1d(512, 64, 5)
        #self.layer4 = self.make_layer(512, 1024, 4) # 2
        #self.layer5 = self.make_layer(1024, 1024, 3) # 1
        #self.layer6 = self.make_layer(512, 512, 3, 3, 1)

        self.fc1 = nn.Linear(512 * 4, 512)
        self.fc2 = nn.Linear(512, 100)
        self.fc3 = nn.Linear(100, 10)
        #self.fc3 = nn.Linear(256, 10)
        #nn.Dropout2d()

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        
        for m in self.modules():
            if isinstance(m, ResidualBlock):
                nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def make_layer(self, inchans, outchans, num_blocks, kernel=3, padding=1, stride=2):
        layers = []
        if stride != 1:
            layers.append(ResidualBlock(inchans, outchans, kernel, padding, stride))
        else:
            layers.append(ResidualBlock(inchans, outchans))

        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(outchans, outchans))
        
        return nn.Sequential(*layers)

    def forward(self, x):
        #x = self.conv1(x)
        #x = self.bn1(x)
        #x = self.relu(x)
        x = (self.d1(x))
        #print(x.shape)
        x = self.pool(self.d2(x))
        #print(x.shape)

        x = self.pool(self.layer1(x))
        #print(x.shape)
        #x = self.pool(x)
        x = self.pool(self.layer2(x))
        #print(x.shape)
        #x = self.pool(x)
        x = self.pool(self.layer3(x))
        x = self.pool(self.layer4(x))
        
        #y = self.holistic_layer(x)
        #y = self.bn1(y)
        #y = self.relu(y)
        #print(y[0])
        #y = y.expand(-1, -1, 5)
        #print(y[0,:,1])
        #print(x.shape)
        #print(y.shape)
        #x = torch.cat((x,y), dim=1)
        #print(x.shape)
        #print(x.shape)
        #x = self.pool(x)
        #x = self.layer4(x)
        #print(x.shape)
        #x = self.pool(x)
        #x = self.layer5(x)
        #x = self.layer6(x)


        x = x.flatten(1)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = x.reshape(x.shape[0], -1, 2)
        return x



class ResidualNetwork2(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        #self.conv1 = nn.Conv1d(2, 16, 21, 5, 10) # 128  ut
        #self.bn1 = nn.BatchNorm1d(16)
        self.relu = nn.ReLU(inplace=True)
        #self.avgpool = nn.AvgPool1d(2)
        #self.pool = nn.MaxPool1d(2)

        self.d1 = self.make_layer(2, 16, 4, 21, 10, 4) # 160
        self.d2 = self.make_layer(16, 64, 5, 11, 5, 4) # 40

        self.layer1 = self.make_layer(64, 128, 6) # 20
        self.layer2 = self.make_layer(128, 256, 5) # 10
        self.layer3 = self.make_layer(256, 512, 4) # 5

        self.downsample = nn.Sequential(
            self.make_layer(512, 256, 1, 3, 1, 1),
            self.make_layer(256, 128, 1, 3, 1, 1),
            self.make_layer(128, 32, 1, 3, 1, 1),
            self.make_layer(32, 8, 1, 3, 1, 1),
            self.make_layer(8, 2, 1, 3, 1, 1),
        )

        #self.layer4 = self.make_layer(512, 256, 1, 3, 1, 1) # (5 x 2)
        #self.layer5 = self.make_layer(256, 128, 1, 3, 1, 1)
        #self.layer6 = self.make_layer(128, 32, 1, 3, 1, 1)
        #self.layer7 = self.make_layer(32, 8, 1, 3, 1, 1)
        #self.layer8 = self.make_layer(8, 2, 1, 3, 1, 1)
        
        #self.layer4 = self.make_layer(512, 1024, 4) # 2
        #self.layer5 = self.make_layer(1024, 1024, 3) # 1
        #self.layer6 = self.make_layer(512, 512, 3, 3, 1)

        #self.fc1 = nn.Linear(512 * 5, 100)
        #self.fc2 = nn.Linear(100, 10)
        #self.fc3 = nn.Linear(256, 10)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        
        for m in self.modules():
            if isinstance(m, ResidualBlock):
                nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def make_layer(self, inchans, outchans, num_blocks, kernel=3, padding=1, stride=2):
        layers = []
        if stride != 1:
            layers.append(ResidualBlock(inchans, outchans, kernel, padding, stride))
        else:
            layers.append(ResidualBlock(inchans, outchans))

        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(outchans, outchans))
        
        return nn.Sequential(*layers)

    def forward(self, x):
        #x = self.conv1(x)
        #x = self.bn1(x)
        #x = self.relu(x)
        x = self.d1(x)
        #print(x.shape)
        x = self.d2(x)
        #print(x.shape)

        x = self.layer1(x)
        #print(x.shape)
        #x = self.pool(x)
        x = self.layer2(x)
        #print(x.shape)
        #x = self.pool(x)
        x = self.layer3(x)
        #print(x.shape)
        #x = self.pool(x)
        #x = self.layer4(x)
        #print(x.shape)
        #x = self.pool(x)
        x = self.downsample(x)
        #print(x.shape)
        #x = self.relu(x)
        #x = self.fc3(x)
        #print(x[0])
        x = torch.permute(x, (0, 2, 1)).contiguous()
        x = x.reshape(x.shape[0], -1, 2)
        #print(x[0])
        return x


if __name__ == '__main__':

    x = torch.rand(10, 2 , 640)

    data = LaserPointDataset('/home/oyvind/Blender-weldgroove/render', noise=True, corrected=True, normalization='dataset', shuffle_gt_and_est=True)
    loader = DataLoader(data, 64, shuffle=True, num_workers=4)
    print(loader)
    model = ResidualNetwork()
    #model = SimpleNetwork24()
    a = (model.modules())
    
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print('num parameters:',pytorch_total_params)
    #print(model)
    a = model(x)
    print(a.shape)