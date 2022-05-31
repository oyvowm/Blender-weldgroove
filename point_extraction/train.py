from pickle import TRUE
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import dataset
import model
import resnet
from conv2d import Conv2DNetwork
from torch.utils.data import DataLoader
import time
from loss_functions import EuclideanLoss
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = "cpu"

print(device)

#torch.manual_seed(1) # set seed for reproducibility

PATH = '/home/oyvind/Blender-weldgroove/ResNet_NewSet125.pth' #24 # 26

config = {
    "num_epochs": 1200,  
    "batch_size": 64, 
    "lr": 1e-3,
    "continue_training": True,
}
# loss at epoch: 999 = 0.002346100521125746 1
# num parameters: 13 369 958

data = dataset.LaserPointDataset('/home/oyvind/Blender-weldgroove/render', noise=True, corrected=True, normalization='dataset', shuffle_gt_and_est=True)
test_set = dataset.LaserPointDataset('/home/oyvind/Blender-weldgroove/render', noise=True, corrected=True, normalization='dataset', test=True, shuffle_gt_and_est=True)

loader = DataLoader(data, config['batch_size'], shuffle=True, num_workers=4)
test_loader = DataLoader(test_set, config['batch_size'], num_workers=4)
#criterion = nn.MSELoss()
#criterion = nn.L1Loss(reduction='sum')
#criterion = nn.L1Loss()
criterion = EuclideanLoss()

net = model.SimpleNetwork23d2()
#net = resnet.ResidualNetwork()
#net = model.ResidualNetwork2()
#net = Conv2DNetwork()
#net = model.FeedForwardNet()
net.to(device)
net.train()
#optimizer = optim.SGD(net.parameters(), config["lr"], momentum=0.9)
optimizer = optim.Adam(net.parameters(), lr=config['lr'], betas=(0.9,0.999), weight_decay=5e-6) # 6
#lambda1 = lambda epoch: 0.75 ** (epoch // 100)
#scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
milestone = [i * 200 for i in range(1, 50)]
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestone, gamma=0.5)

#scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, eta_min=0, last_epoch=-1)
start_epoch = 0
if config['continue_training']:
    try:
        checkpoint = torch.load(PATH)
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    except:
        print('no previous checkpoint found -- training from scratch')


try:
    test_losses = checkpoint['test_losses']
    training_losses = checkpoint['training_losses']
except:
    test_losses = []
    training_losses= []
for epoch in range(start_epoch+1, config["num_epochs"]):
    net.train()
    running_loss = 0.0
    start = time.time()
    for i, data in enumerate(loader):
        #print(i)
        x, y = data[0].to(device), data[-1].to(device)

        outputs = net(x)
        
        optimizer.zero_grad()
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        #if i % 20 == 19:
        #    print(f'loss at epoch: {epoch} and mini-batch: {i+1} = {running_loss / 20}')
        #    running_loss = 0.0
    
    loss = running_loss / (i+1)
    training_losses.append(loss)
    print(f'training loss at epoch: {epoch} = {loss} epoch time: {time.time()-start} lr: {scheduler.get_last_lr()}')
    
    #if (epoch+1) % 100:
    scheduler.step()

    if (epoch+1) % 20 == 0:
        net.eval()
        test_loss = 0.0
        with torch.no_grad():
            for i, d in enumerate(test_loader):
                x, y = d[0].to(device), d[-1].to(device)
                outputs = net(x)
                #print(outputs.shape)
                #print(y.shape)
                loss = criterion(outputs, y)
                test_loss += loss.item()
        
        test_loss = test_loss / (i + 1)
        test_losses.append(test_loss)
        #print(test_losses)
        print(min(test_losses))
        print(f'test loss: = {test_loss}')
        print('##########################')

        #if test_loss < (sum(test_losses[-5:-1]) / 4) - 0.0003 and test_loss < test_losses[-2]:
        if len(test_losses) > 1 and test_loss < min(test_losses[:-1]):
            print('saving model...')
            torch.save({
                'model_state_dict': net.state_dict(),
                'epoch': epoch,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'test_losses': test_losses,
                'training_losses': training_losses,
                },PATH)

            if epoch > 400:
                break


    if (epoch+1) % 100 == 0:
        torch.save({
                    'model_state_dict': net.state_dict(),
                    'epoch': epoch,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'test_losses': test_losses,
                    'training_losses': training_losses,
                    },PATH)
#test_xs = np.linspace(0, test_losses[-1], test_losses[-1])
plt.plot(training_losses)
plt.plot(np.arange(0, epoch+1, 20), test_losses)
plt.show()
