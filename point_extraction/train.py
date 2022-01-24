from pickle import TRUE
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import dataset
import model
from conv2d import Conv2DNetwork
from torch.utils.data import DataLoader
import time
from loss_functions import EuclideanLoss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = "cpu"
# Assuming that we are on a CUDA machine, this should print a CUDA device:

print(device)

PATH = '/home/oyvind/Blender-weldgroove/ResNet_EuclideanLoss_94kernel_bs64.pth' # storre; 8841574 param

config = {
    "num_epochs": 8000,  
    "batch_size": 64, 
    "lr": 0.001,
    "continue_training": True,
}
# loss at epoch: 999 = 0.002346100521125746 1
# num parameters: 13 369 958

#loss at epoch: 996 = 0.001686061208602041 epoch time: 0.9587793350219727 lr: [0.000125]
#loss at epoch: 997 = 0.001756916740718721 epoch time: 0.9566116333007812 lr: [0.000125]
#loss at epoch: 998 = 0.001890811024085534 epoch time: 0.9343113899230957 lr: [0.000125]
#loss at epoch: 999 = 0.001884016224225475

data = dataset.LaserPointDataset('/home/oyvind/Blender-weldgroove/render', noise=True, corrected=True, normalization='dataset')
test_set = dataset.LaserPointDataset('/home/oyvind/Blender-weldgroove/render', noise=True, corrected=True, normalization='dataset', test=True)

loader = DataLoader(data, config['batch_size'], shuffle=True, num_workers=4)
test_loader = DataLoader(test_set, config['batch_size'], num_workers=4)
#criterion = nn.MSELoss()
#criterion = nn.L1Loss(reduction='sum')
#criterion = nn.L1Loss()
criterion = EuclideanLoss()

#net = model.SimpleNetwork3()
net = model.ResidualNetwork()
#net = Conv2DNetwork()
#net = model.FeedForwardNet()
net.to(device)
net.train()
#optimizer = optim.SGD(net.parameters(), config["lr"], momentum=0.9)
optimizer = optim.Adam(net.parameters(), lr=1e-3, betas=(0.9,0.999), weight_decay=1e-5)
#lambda1 = lambda epoch: 0.75 ** (epoch // 100)
#scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
milestone = [i * 2000 for i in range(1, 10)]
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestone, gamma=0.5)

#scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, eta_min=0, last_epoch=-1)
start_epoch = 0
if config['continue_training']:

    checkpoint = torch.load(PATH)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])




#print(len(loader))
for epoch in range(start_epoch+1, config["num_epochs"]):
    net.train()
    running_loss = 0.0
    start = time.time()
    for i, data in enumerate(loader):
        x, y = data[0].to(device), data[-1].to(device)

        #print('sdfgd',y.shape[0])
        outputs = net(x)
        
        optimizer.zero_grad()

        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        #if i % 20 == 19:
        #    print(f'loss at epoch: {epoch} and mini-batch: {i+1} = {running_loss / 20}')
        #    running_loss = 0.0
    
    print(f'training loss at epoch: {epoch} = {running_loss / (i+1)} epoch time: {time.time()-start} lr: {scheduler.get_last_lr()}')
    
    #if (epoch+1) % 100:
    scheduler.step()

    if (epoch+1) % 20 == 0:
        net.eval()
        test_loss = 0.0
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                x, y = data[0].to(device), data[-1].to(device)
                
                outputs = net(x)
                loss = criterion(outputs, y)
                test_loss += loss.item()
        
        print(f'test loss: = {test_loss / (i + 1)}')
        print('##########################')

    if (epoch+1) % 500 == 0:
        torch.save({
                    'model_state_dict': net.state_dict(),
                    'epoch': epoch,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    },PATH)
