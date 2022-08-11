from numpy.core.numeric import identity
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn.modules.batchnorm import BatchNorm1d
from torch.nn.modules.container import Sequential
from dataset import LaserPointDataset
from torch.utils.data import DataLoader
from torchinfo import summary
from time import time


class SimpleNetwork23o(nn.Module):
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
        x = x.reshape(x.shape[0], -1, 2)
        return x

class SimpleNetwork23(nn.Module):
    """
    added more layers
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(2, 16, 21, 10, 10) # 64  ut
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 64, 3, 1, 1) # 32 ut
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, 3, 1, 1) # 16 ut
        self.bn3 = nn.BatchNorm1d(128)
        self.conv4 = nn.Conv1d(128, 256, 3, 1, 1) # 8 ut
        self.bn4 = nn.BatchNorm1d(256)
        self.conv5 = nn.Conv1d(256, 512, 3, 1, 1) # 4 ut
        self.bn5 = nn.BatchNorm1d(512)
        #self.conv6 = nn.Conv1d(512, 512, 3, 2, 1) # 10 ut
        #self.bn6 = nn.BatchNorm1d(512)
        #self.conv7 = nn.Conv1d(512, 512, 3, 2, 1) # 5 ut
        #self.bn7 = nn.BatchNorm1d(512)
        #self.conv8 = nn.Conv1d(512, 512, 3, 1) # 3 ut
        #self.bn8 = nn.BatchNorm1d(512)
        self.pool = nn.MaxPool1d(2)

        self.fc1 = nn.Linear(512*4, 256)
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
        #print(x.shape)
        x = F.relu(self.bn1(x))

        x = self.conv2(x)
        #print(x.shape)
        x = self.pool(F.relu(self.bn2(x)))

        x = self.conv3(x)
        #print(x.shape)
        x = self.pool(F.relu(self.bn3(x)))

        x = self.conv4(x)
        #print(x.shape)
        x = self.pool(F.relu(self.bn4(x)))

        x = self.conv5(x)
        #print(x.shape)
        x = self.pool(F.relu(self.bn5(x)))


        #print(x.shape)

        x = x.flatten(1)
        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        x = self.fc3(x)
        x = x.reshape(x.shape[0], -1, 2)
        return x

class SimpleNetwork23b(nn.Module):
    """
    added more layers
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(2, 16, 21, 5, 10) # 128  ut
        self.bn1 = nn.BatchNorm1d(16)
        self.a = nn.Conv1d(16, 32, 3, 1, 1) # 32 ut
        self.abn = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, 3, 1, 1) # 32 ut
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, 3, 1, 1) # 16 ut
        self.bn3 = nn.BatchNorm1d(128)
        self.conv4 = nn.Conv1d(128, 256, 3, 1, 1) # 8 ut
        self.bn4 = nn.BatchNorm1d(256)
        self.conv5 = nn.Conv1d(256, 512, 3, 1, 1) # 4 ut
        self.bn5 = nn.BatchNorm1d(512)
        #self.conv6 = nn.Conv1d(512, 512, 3, 2, 1) # 10 ut
        #self.bn6 = nn.BatchNorm1d(512)
        #self.conv7 = nn.Conv1d(512, 512, 3, 2, 1) # 5 ut
        #self.bn7 = nn.BatchNorm1d(512)
        #self.conv8 = nn.Conv1d(512, 512, 3, 1) # 3 ut
        #self.bn8 = nn.BatchNorm1d(512)
        self.pool = nn.MaxPool1d(2)

        self.fc1 = nn.Linear(512*4, 256)
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
        #print(x.shape)
        x = F.relu(self.bn1(x))

        x = self.pool(F.relu(self.abn(self.a(x))))

        x = self.conv2(x)
        #print(x.shape)
        x = self.pool(F.relu(self.bn2(x)))

        x = self.conv3(x)
        #print(x.shape)
        x = self.pool(F.relu(self.bn3(x)))

        x = self.conv4(x)
        #print(x.shape)
        x = self.pool(F.relu(self.bn4(x)))

        x = self.conv5(x)
        #print(x.shape)
        x = self.pool(F.relu(self.bn5(x)))


        #print(x.shape)

        x = x.flatten(1)
        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        x = self.fc3(x)
        x = x.reshape(x.shape[0], -1, 2)
        return x

def depthwise_conv(in_chan, out_chan):
    return nn.Sequential(
        nn.Conv1d(in_chan, in_chan, 3, 1, 1, groups=in_chan, bias=False),
        nn.BatchNorm1d(in_chan),
        nn.ReLU(inplace=True),

        nn.Conv1d(in_chan, out_chan, 1, 1, bias=False),
        nn.BatchNorm1d(out_chan),
        nn.ReLU(inplace=True),
        nn.MaxPool1d(2),
    )

def conv_block(in_chan, out_chan, use_batchnorm=True, use_dropout=False, *args, **kwargs):
    layers = [nn.Conv1d(in_chan, out_chan, *args, **kwargs)]
    if use_batchnorm:
        layers.append(nn.BatchNorm1d(out_chan))
    layers.append(nn.ReLU(inplace=True))
    if use_dropout:
        layers.append(nn.Dropout(0.3, inplace=False))
    if in_chan != out_chan and in_chan < 256:
        layers.append(nn.MaxPool1d(2))
    #layers.append(nn.Conv1d(out_chan, out_chan, 1, 1))
    #layers.append(nn.BatchNorm1d(out_chan))
    #layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)

class SimpleNetwork23d2(nn.Module):
    """
    added more layers
    """
    def __init__(self):
        super().__init__() # 15, 7
        self.conv1 = nn.Conv1d(2, 8, 15, 4, 7) # 160  ut
        self.bn1 = nn.BatchNorm1d(8)

        self.encoding_dims = [8, 16, 32, 64, 64, 128, 128, 256, 256, 512]


        self.encoder = nn.Sequential(
            *[conv_block(in_chan, out_chan, kernel_size=3, padding=1) for
            in_chan, out_chan in zip(self.encoding_dims, self.encoding_dims[1:])]
        )


        self.regressor = nn.Sequential(
            nn.Linear(512*5, 512), 
            nn.ReLU(inplace=True),
            nn.Linear(512, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 10)
        )


        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        #print(x.shape)
        x = F.relu(self.bn1(x))

        x = self.encoder(x)
       # print(x.shape)        
        x = x.flatten(1)
        x = self.regressor(x)
        x = x.reshape(x.shape[0], -1, 2)
        return x


class SimpleNetwork23d3(nn.Module):
    """
    added more layers
    """
    def __init__(self):
        super().__init__() # 15, 7
        self.conv1x1 = nn.Conv1d(2,1,1,1,0)
        self.bn1x1 = nn.BatchNorm1d(1)
        self.conv1 = nn.Conv1d(1, 4, 15, 4, 7) # 160  ut
        self.bn1 = nn.BatchNorm1d(4)

        self.encoding_dims = [2, 4, 8, 16, 32, 64, 128, 256]


        #self.encoder = nn.Sequential(
        #    *[conv_block(in_chan, out_chan, kernel_size=3, padding=1) for
        #    in_chan, out_chan in zip(self.encoding_dims, self.encoding_dims[1:])]
        #)
        self.encoder = nn.Sequential(
        *[depthwise_conv(in_chan, out_chan) for 
        in_chan, out_chan in zip(self.encoding_dims, self.encoding_dims[1:])]
        )

        self.regressor = nn.Sequential(
            nn.Linear(256*5, 256), 
            nn.ReLU(inplace=True),
            nn.Linear(256, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 10)
        )


        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        #x = F.relu(self.bn1x1(self.conv1x1(x)))
        #x = self.conv1(x)
        #print(x.shape)
        #x = F.relu(self.bn1(x))
        #print(x.shape)
        x = self.encoder(x)

        #print(x.shape)        
        x = x.flatten(1)
        x = self.regressor(x)
        x = x.reshape(x.shape[0], -1, 2)
        return x



if __name__ == '__main__':

    x = torch.rand(10, 2 , 640)

    data = LaserPointDataset('/home/oyvind/Blender-weldgroove/render', noise=True, corrected=True, normalization='dataset', shuffle_gt_and_est=True)
    loader = DataLoader(data, 64, shuffle=True, num_workers=4)
    print(loader)
    #model = ResidualNetwork()
    model = SimpleNetwork23b()
    
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print('num parameters:',pytorch_total_params)
    #print(model)
    start = time()
    a = model(x)
    print(f'inference time: {time() - start}')
    print(a.shape)

    summary(model, (10, 2, 640))