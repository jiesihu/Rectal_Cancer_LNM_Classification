#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import shutil
import tempfile
import sys
import argparse
import time
from datetime import datetime
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import numpy
import pandas as pd

from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

from monai.apps import download_and_extract
from monai.config import print_config
from monai.metrics import ROCAUCMetric
from monai.networks.nets import DenseNet121,resnet101,resnet18,resnet34,resnet50,resnet10,resnet152
from monai.transforms import (
    Activations,
    AddChannel,
    AsDiscrete,
    Compose,
    LoadImage,
    RandFlip,
    RandRotate,
    RandZoom,
    ScaleIntensity,
    ToTensor,
    Spacing,
    Resize,
    RandAffine,
    RandRicianNoise,
    RandShiftIntensity,
    RandScaleIntensity,
    RandAdjustContrast,
)
from monai.data import Dataset, DataLoader
from monai.utils import set_determinism
print_config()


import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils import get_network, get_training_dataloader, get_test_dataloader,     WarmUpLR,most_recent_folder, most_recent_weights, last_epoch, best_acc_weights,    get_img_info,split_the_patient,calculateModelScores,integrate_patient,integrate_patient,AddGaussianNoise
from metric import calculateModelScores
from dataloader import Tumor_Dataset


# In[2]:


parser = argparse.ArgumentParser()
parser.add_argument('-net', type=str, required=True, help='net type')
parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
parser.add_argument('-acc', action='store_true', default=False, help='save best acc')
parser.add_argument('-loss', action='store_true', default=False, help='save best loss')
parser.add_argument('-AUC', action='store_true', default=False, help='save best AUC')
parser.add_argument('-b', type=int, default=8, help='batch size for dataloader')
parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
parser.add_argument('-weight_decay', type=float, default=5e-3, help='initial weight decay')
parser.add_argument('-resume', action='store_true', default=False, help='resume training')
parser.add_argument('-opt', type=str,default='SGD',choices = ['SGD','Adam'], help='Set the optimizer')
parser.add_argument('-interval', type=int, default=5, help='the epoch to start record best model')
parser.add_argument('-total_epoch', type=int, default=300, help='the total epoch')
parser.add_argument('-notes', type=str, help='notes for the model')
parser.add_argument('-seed', type=int, default=5, help='seed for reproducibility')
parser.add_argument('-data_path', type=str, help='notes for the model')
parser.add_argument('-pretrained', action='store_true', default=False)
parser.add_argument('-pretrained_path', type=str, default = './Med3D_pretrain/', help='where pre-trained files saved')
args = parser.parse_args(args = ['-net', 'Med3D_resnet101', '-gpu','-b','8','-lr','0.00001','-AUC',
                                '-opt','Adam','-interval','1','-total_epoch','200',
                                 '-weight_decay','0.0025','-data_path','./sample_data/'])
val_interval = args.interval
num_class = 2
num_worker = 0
shuffle = True
batch_size = args.b
epoch_num = args.total_epoch
class_names = ['Non metastasis','Metastasis']
print(args)


# In[3]:


#  Set random seed
torch.manual_seed(args.seed)
import random
random.seed(args.seed)
np.random.seed(args.seed)
set_determinism(seed=args.seed)


# In[4]:


# # Load data
x_train = get_img_info(os.path.join(args.data_path,'train/'))
x_val = get_img_info(os.path.join(args.data_path,'val/'))



# In[5]:


# Transform
# img_size = 128 
train_transforms = Compose([
#     AddChannel(),
    ScaleIntensity(),
    RandRotate(range_x=15, prob=0.5, keep_size=True,padding_mode = 'border'),# reflection, zeros
    RandFlip(prob = 0.3,spatial_axis =0),
    RandFlip(prob = 0.3,spatial_axis =1),
    RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5, keep_size=True),
    RandAffine(translate_range=30,prob = 0.2,padding_mode = 'border'),
#     RandRicianNoise(prob = 0.1,mean = 0,std = 0.05,channel_wise = True,),
#     RandShiftIntensity(offsets = 0.1,prob= 1),
    RandAdjustContrast(prob = 0.3,gamma = (0.7,2)),
#     Resize(spatial_size = (img_size,img_size)),
    ToTensor(),
    AddGaussianNoise(0,0.05,p = 0.2)
])

val_transforms = Compose([
#     AddChannel(),
    ScaleIntensity(),
#     Resize(spatial_size = (img_size,img_size)),
    ToTensor()
])

act = Activations(softmax=True)
to_onehot = AsDiscrete(to_onehot=True, n_classes=num_class)


# In[6]:


# Get dataloaders
transform_train = train_transforms
transform_test = val_transforms

train_ds = Tumor_Dataset(x_train,transform=transform_train)
val_ds = Tumor_Dataset(x_val,transform=transform_test)

train_loader = DataLoader(
    train_ds, shuffle=shuffle, num_workers=num_worker, batch_size=batch_size)
val_loader = DataLoader(
    val_ds, shuffle=False, num_workers=num_worker, batch_size=batch_size)

print("Training dataset:",len(train_ds))
print("Validation dataset:",len(val_ds))


# In[7]:


# ## Plot some samples 
# display some data
print('Read Training set')
import matplotlib.pyplot as plt
temp = val_ds[0]
print('Name:',temp[-1])
print('Max:',np.max(temp[0].numpy()))
print('Min:',np.min(temp[0].numpy()))
print('Mean:',np.mean(temp[0].numpy()))
print('Std:',np.std(temp[0].numpy()))
print('Size:',temp[0].numpy().shape)
plt.subplots(3, 3, figsize=(8, 8))
for i,k in enumerate(np.random.randint(len(val_ds), size=9)):
    arr = train_ds[k][0].numpy()
    plt.subplot(3, 3, i + 1)
    plt.title('Label '+str(train_ds[k][-3]))
    plt.imshow(arr[0,0,:,:], cmap='gray')
plt.tight_layout()
plt.show()
# display some data
print('Read Val set')
import matplotlib.pyplot as plt
temp = val_ds[0]
print('Name:',temp[-1])
print('Max:',np.max(temp[0].numpy()))
print('Min:',np.min(temp[0].numpy()))
print('Mean:',np.mean(temp[0].numpy()))
print('Std:',np.std(temp[0].numpy()))
print('Size:',temp[0].numpy().shape)
plt.subplots(3, 3, figsize=(8, 8))
for i,k in enumerate(np.random.randint(len(val_ds), size=9)):
    arr = val_ds[k][0].numpy()
    plt.subplot(3, 3, i + 1)
    plt.title('Label '+str(val_ds[k][-3]))
    plt.imshow(arr[0,0,:,:], cmap='gray')
plt.tight_layout()
plt.show()


# In[8]:


# Define network and optimizer
# Load model
device = torch.device("cuda:0")
if args.net == 'Med3D_resnet101':
    model = resnet101(num_classes = 2,n_input_channels = 1).to(device)
    pretrained_model = os.path.join(args.pretrained_path,'resnet_101.pth')
    print('Use Med3D_resnet101!')
elif args.net == 'Med3D_resnet50':
    model = resnet50(num_classes = 2,n_input_channels = 1).to(device)
    pretrained_model = os.path.join(args.pretrained_path,'resnet_50_23dataset.pth')
    print('Use Med3D_resnet50!')
elif args.net == 'Med3D_resnet18':
    model = resnet18(num_classes = 2,n_input_channels = 1).to(device)
    pretrained_model = os.path.join(args.pretrained_path,'resnet_18_23dataset.pth')
    print('Use Med3D_resnet18!')
elif args.net == 'Med3D_resnet152':
    model = resnet152(num_classes = 2,n_input_channels = 1).to(device)
    pretrained_model = os.path.join(args.pretrained_path,'resnet_152.pth')
    print('Use Med3D_resnet152!')


# In[9]:


if args.pretrained:
    # load pretrained med3D
    state_dict = torch.load(pretrained_model)['state_dict']

    for key in list(state_dict.keys()):
        new_key = '.'.join(key.split('.')[1:])
        state_dict[new_key] = state_dict[key]
        del state_dict[key]

    model_dict = model.state_dict()
    state_dict = {
            k: v for k, v in state_dict.items() if (k in model_dict) and (model_dict[k].shape == state_dict[k].shape)}
    print('Length of update dic:',len(state_dict))
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)


# In[10]:


print(model)


# In[11]:


# Load loss functin, Opt

def count_weight():
    label = []
    for i in range(0,len(train_ds)):
        label.append(train_ds[i][1])
    weight1 = len(label)/sum(label)
    weight0 = len(label)/(len(label)-sum(label))
    s = weight1/2+weight0/2
    return torch.FloatTensor([weight0/s,weight1/s]).cuda()
weight_ = count_weight()
print('Weight:',weight_)


# In[12]:


loss_function = torch.nn.CrossEntropyLoss(weight = weight_)


# In[13]:


#Setting Opt.
training_par = filter(lambda p: p.requires_grad,model.parameters())
if args.opt == 'Adam':
    optimizer = optim.Adam(training_par, lr=args.lr,weight_decay=args.weight_decay) # weight_decay = L2 norm
    print('Adam opt!')
else: 
    optimizer = optim.SGD(training_par, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay) # weight_decay = L2 norm
    print('SGD opt!')
print('LR:',args.lr,'\t weight decay',args.weight_decay)


# In[14]:


def report_performance(val_loader,metric_values_val,dataset = 'Validation',integrate = False):
    y_true_l = list()
    y_pred_l = list()
    y_prob_l = list()
    y_name_l = list()
    with torch.no_grad():
            y_pred = torch.tensor([], dtype=torch.float32, device=device)
            y = torch.tensor([], dtype=torch.long, device=device)
            for val_data in val_loader: #val_dataloader
                val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
                output_temp = model(val_images)
                y_pred = torch.cat([y_pred, output_temp], dim=0)
                y = torch.cat([y, val_labels], dim=0)
                
                prob = output_temp
                pred = prob.argmax(dim=1)
                # record the result
                for i in range(len(pred)):
                    y_true_l.append(val_labels[i].item())
                    y_pred_l.append(pred[i].item())
                    y_prob_l.append(nn.Softmax(dim=1)(prob)[i][1].item())
                    y_name_l.append(val_data[3][i])
                    
            y_onehot = [to_onehot(i) for i in y]
            y_pred_act = [act(i) for i in y_pred]
            auc_metric(y_pred_act, y_onehot)
            auc_result = auc_metric.aggregate()
            auc_metric.reset()
            del y_pred_act, y_onehot
            if metric_values_val is not None:
                metric_values_val.append(auc_result)
            acc_value = torch.eq(y_pred.argmax(dim=1), y)
            acc_metric = acc_value.sum().item() / len(acc_value)

            print(dataset+f" current epoch: {epoch + 1} current AUC: {auc_result:.4f}"
                  f" current accuracy: {acc_metric:.4f}")
            
            if integrate:
                # Integrate AUC
                result,result['label'],result['pred'],result['prob'],result['name'] = {},y_true_l,y_pred_l,y_prob_l,y_name_l
                result_inte = integrate_patient(result)
                score = calculateModelScores(result_inte['label'], result_inte['pred'],result_inte['prob'], 
                                           dataset='Test',verbal = False)
                score['label'],score['pred'],score['prob'],score['name'] = result_inte['label'], result_inte['pred'],result_inte['prob'],result_inte['name']
                print(dataset+f" final AUC: {score['auc']:.4f}"
                  f" final accuracy: {score['accuracy']:.4f}")
                return epoch,auc_result,acc_metric,score
def get_interval(epoch,args,interval_long = 10, epoch_cut_off = 50):
    if epoch>epoch_cut_off:
        return args.interval
    else:
        return interval_long


# In[15]:


# change auc output
# add train & val auc result
best_metric = -1
best_metric_epoch = -1
epoch_loss_values = list()
auc_metric = ROCAUCMetric()
Train_data = {}
Val_data = {}
Train_data['metric_values'] = list()
Train_data['metric_values_final'] = list()
Val_data['metric_values'] = list()
Val_data['metric_values_final'] = list()


# In[16]:


#time of running the script
DATE_FORMAT = '%A_%d_%B_%Y_%Hh_%Mm_%Ss'
TIME_NOW = datetime.now().strftime(DATE_FORMAT)
log_path = os.path.join('./logs/', args.net,TIME_NOW)
print(log_path)
if not os.path.exists(log_path):
        os.makedirs(log_path)


# In[18]:



for epoch in range(epoch_num):
    print('-' * 10)
    print(f"epoch {epoch + 1}/{epoch_num}")
    model.train()
    epoch_loss = 0
    step = 0
    for batch_data in train_loader:
        step += 1
        inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        if step%50==0:
            print(f"{step}/{len(train_ds) // train_loader.batch_size}, train_loss: {loss.item():.4f}")
        epoch_len = len(train_ds) // train_loader.batch_size
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

    if (epoch + 1) % get_interval(epoch,args) == 0:
        model.eval()
        epoch,auc_result,acc_metric,score = report_performance(train_loader,Train_data['metric_values'],
                                                               dataset = 'Train:',integrate = True)
        score['epoch'] = epoch
        Train_data['metric_values_final'].append(score)
        
    if (epoch + 1) % get_interval(epoch,args) == 0:
        model.eval()
        epoch,auc_result,acc_metric,score = report_performance(val_loader,Val_data['metric_values'],
                                                               dataset = 'Validation:',integrate = True)
        score['epoch'] = epoch
        Val_data['metric_values_final'].append(score)
        
        if score['auc'] > best_metric:
            best_metric = score['auc']
            best_metric_epoch = epoch + 1
            torch.save(model.state_dict(),os.path.join(log_path,'best_metric_model.pth') )
            print('saved new best metric model')
        if score['auc'] >0.75:
            torch.save(model.state_dict(),os.path.join(log_path,'best_metric_model'+str(epoch)+'.pth') )
        print(f"Test: current epoch: {epoch + 1} current AUC: {score['auc']:.4f}"
              f" current accuracy: {acc_metric:.4f} best AUC: {best_metric:.4f}"
              f" at epoch: {best_metric_epoch}")
print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")


# In[21]:


# ## Plot the loss and metric

# In[ ]:


plt.figure('train', (12, 12))
plt.subplot(2, 2, 1)
plt.title("Epoch Average Loss")
x = [i + 1 for i in range(len(epoch_loss_values))]
y = epoch_loss_values
plt.xlabel('epoch')
plt.plot(x, y)

plt.subplot(2, 2, 2)
plt.title("Validation: Area under the ROC curve")
x = [val_interval * (i + 1) for i in range(len(Val_data['metric_values']))]
y = Val_data['metric_values']
y_train = Train_data['metric_values']
#y_val = Val_data['metric_values']
plt.xlabel('epoch')
plt.plot(x, y_train,label='Train')
plt.plot(x, y,label='Val')
plt.legend()

plt.subplot(2, 2, 3)
y_train_final = [i['auc'] for i in Train_data['metric_values_final']]
#y_val_final = [i['auc'] for i in Val_data['metric_values_final']]
y_val_final = [i['auc'] for i in Val_data['metric_values_final']]
plt.title("Validation: Area under the ROC curve")
plt.xlabel('epoch')
plt.plot(x, y_train_final,label='Train final')
plt.plot(x, y_val_final,label='val final')
plt.legend()
plt.show()


# # Save the result

# In[ ]:


result_all_epoch,result_all_epoch['args'], = {},vars(args)
result_all_epoch['Train'] = Train_data['metric_values_final']
result_all_epoch['Val'] = Val_data['metric_values_final']

# save all the result
import pickle
with open(os.path.join(log_path,'result.pickle'),'wb') as handle:
    pickle.dump(result_all_epoch,handle,protocol = pickle.HIGHEST_PROTOCOL)
# save the setting
import json
json_str = json.dumps(vars(args),indent = 4)
with open(os.path.join(log_path,'setting.json'),'w') as json_file:
    json_file.write(json_str)


# In[ ]:




