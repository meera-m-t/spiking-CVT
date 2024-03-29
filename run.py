import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from cvt_spiking import CvT
import torch
import torch.nn as nn
import time
from torch.utils.data import DataLoader
import torchvision
import numpy as np
from SpykeTorch import snn
from SpykeTorch import functional as sf
from SpykeTorch import visualization as vis
from SpykeTorch import utils
from torchvision import transforms
from torch.autograd import Variable

use_cuda = True
def train_unsupervise(network, data, layer_idx):
    network.train()
    for i in range(len(data)):
        data_in = data[i]
        if use_cuda:
            data_in = data_in.cuda()
        network(data_in, layer_idx)
        network.stdp(layer_idx)

def train_rl(network, data, target):
    network.train()
    perf = np.array([0,0,0]) # correct, wrong, silence
    for i in range(len(data)):
        data_in = data[i]
        target_in = target[i]
        if use_cuda:
            data_in = data_in.cuda()
            target_in = target_in.cuda()
        d = network(data_in, 3)
        if d != -1:
            if d == target_in:
                perf[0]+=1
                network.reward()
            else:
                perf[1]+=1
                network.punish()
        else:
            perf[2]+=1
    return perf/len(data)




def test(network, data, target):
    network.eval()
    perf = np.array([0,0,0]) # correct, wrong, silence
    for i in range(len(data)):
        data_in = data[i]
        target_in = target[i]
        if use_cuda:
            data_in = data_in.cuda()
            target_in = target_in.cuda()
        d = network(data_in, 3)
        if d != -1:
            if d == target_in:
                perf[0]+=1
            else:
                perf[1]+=1
        else:
            perf[2]+=1
    return perf/len(data)
        
        

class S1Transform:
    def __init__(self, filter, timesteps = 1):
        self.resize = transforms.Resize(256)
        self.centercrop = transforms.CenterCrop(256)  
        self.grayscale = transforms.Grayscale()                 
        self.to_tensor = transforms.ToTensor()
        self.filter = filter
        self.temporal_transform = utils.Intensity2Latency(timesteps)
        self.cnt = 0
    def __call__(self, image):
        if self.cnt % 1000 == 0:
            print(self.cnt,"OOOOOOOOOOOOO")
        self.cnt+=1
        image = self.resize(image)
        image = self.centercrop(image)  
        image = self.grayscale(image)             
        image = self.to_tensor(image) * 255
        image.unsqueeze_(0)
        image = self.filter(image)
        image = sf.local_normalization(image, 8)
        temporal_image = self.temporal_transform(image)   
        return temporal_image.sign().byte()

kernels = [ utils.DoGKernel(7,1,2),
            utils.DoGKernel(7,2,1),]
filter = utils.Filter(kernels, padding = 3, thresholds = 50)
s1c1 = S1Transform(filter)

TRANSFORM_IMG = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225] )
    ])
trainset = utils.CacheDataset(torchvision.datasets.ImageNet('/media/sameerahtalafha/easystore/data/', split='train', transform=s1c1))
ImageNet_trainLoader = DataLoader(trainset, batch_size=1,shuffle=True,  num_workers=4)

valset = utils.CacheDataset(torchvision.datasets.ImageNet('/media/sameerahtalafha/easystore/data/', split='val', transform=s1c1))
ImageNet_valLoader = DataLoader(valset, batch_size=1, shuffle=True,  num_workers=4)                                  
    

mozafari = CvT(2, 100, 1000, (50,50), 1000, (0.0001, -0.0035), (-0.0001, 0.00006), 0.5)
if use_cuda:
    mozafari.cuda()

# Training The First Layer
print("Training the first layer")
if os.path.isfile("saved_l1.net"):
    mozafari.load_state_dict(torch.load("saved_l1.net"))
else:
    for epoch in range(10):
        print("Epoch", epoch)
        iter = 0

        for step, (data,targets) in enumerate (ImageNet_trainLoader):
            # print(step,"uuuuuuuuuuuuuuuuuu")
            data= Variable(data)   # batch x (image)
            targets= Variable(targets)   # batch y (target)  
            # print(data.shape,"pppppppppppppp")          
            # print(targets.shape,"uuuuuuuuuuuuuuuu")

            # print("Iteration", iter)
            train_unsupervise(mozafari, data, 1)
            # print("Done!")
            iter+=1
            if iter == 16:
                break

    torch.save(mozafari.state_dict(), "saved_l1.net")
# Training The Second Layer
print("Training the second layer")
if os.path.isfile("saved_l2.net"):
    mozafari.load_state_dict(torch.load("saved_l2.net"))
else:
    for epoch in range(10):
        print("Epoch", epoch)
        iter = 0
        for data,targets in ImageNet_trainLoader:
            # print("Iteration", iter)
            train_unsupervise(mozafari, data, 2)
            # print("Done!")
            iter+=1
            if iter == 16:
                # print("yessssssssssssssssss")
                break           
            
    torch.save(mozafari.state_dict(), "saved_l2.net")

# initial adaptive learning rates
apr = mozafari.stdp3.learning_rate[0][0].item()
anr = mozafari.stdp3.learning_rate[0][1].item()
app = mozafari.anti_stdp3.learning_rate[0][1].item()
anp = mozafari.anti_stdp3.learning_rate[0][0].item()

adaptive_min = 0
adaptive_int = 1
apr_adapt = ((1.0 - 1.0 / 10) * adaptive_int + adaptive_min) * apr
anr_adapt = ((1.0 - 1.0 / 10) * adaptive_int + adaptive_min) * anr
app_adapt = ((1.0 / 10) * adaptive_int + adaptive_min) * app
anp_adapt = ((1.0 / 10) * adaptive_int + adaptive_min) * anp

# perf
best_train = np.array([0.0,0.0,0.0,0.0]) # correct, wrong, silence, epoch
best_test = np.array([0.0,0.0,0.0,0.0]) # correct, wrong, silence, epoch

# Training The Third Layer
zz = 0
print("Training the third layer")
for epoch in range(30):
    print("Epoch #:", epoch)
    perf_train = np.array([0.0,0.0,0.0])
    for data,targets in ImageNet_trainLoader:
        perf_train_batch = train_rl(mozafari, data, targets)
        print(perf_train_batch)
        #update adaptive learning rates
        apr_adapt = apr * (perf_train_batch[1] * adaptive_int + adaptive_min)
        anr_adapt = anr * (perf_train_batch[1] * adaptive_int + adaptive_min)
        app_adapt = app * (perf_train_batch[0] * adaptive_int + adaptive_min)
        anp_adapt = anp * (perf_train_batch[0] * adaptive_int + adaptive_min)
        mozafari.update_learning_rates(apr_adapt, anr_adapt, app_adapt, anp_adapt)
        perf_train += perf_train_batch
        zz += 1
        if zz == 16:
            break
    perf_train /= len(ImageNet_trainLoader)
    if best_train[0] <= perf_train[0]:
        best_train = np.append(perf_train, epoch)
    print("Current Train:", perf_train)
    print("   Best Train:", best_train)

    for data,targets in ImageNet_valLoader :
        # print("hellpppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppp")
        perf_test = test(mozafari, data, targets)
        if best_test[0] <= perf_test[0]:
            best_test = np.append(perf_test, epoch)
            torch.save(mozafari.state_dict(), "saved.net")
        print(" Current Test:", perf_test)
        print("    Best Test:", best_test)
        
        break
