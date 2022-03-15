import numpy as np
import pandas as pd

def find_patient(filenames):
    # find number of patient
    files_temp = []
    for i in filenames:
        temp = int(i.split('/')[-1].split('_')[0][7:])
        files_temp.append(temp)
    files_temp = np.array(list(set(files_temp)))
    return files_temp
def integrate_patient(result):# Garbage design
    
    # build dict to store prob
    result2 = {}
    label2 = {}
    for i in find_patient(result['name']):
        result2[i] = []

    for i in range(len(result['label'])):
        temp = int(result['name'][i].split('/')[-1].split('_')[0][7:])
        result2[temp].append(result['prob'][i])
        label2[temp]=result['label'][i]
    # average prob
    for i in result2.keys():
        result2[i] = np.mean(result2[i],axis=0)
    # rearrage
    result3 = {}
    result3['label'],result3['pred'],result3['prob'],result3['name']=[],[],[],[]
    for i in result2.keys():
        try:
            result3['name'].append(i)
        except:
            pass
        result3['prob'].append(result2[i])
        result3['label'].append(label2[i])
    result3['prob'] = np.array(result3['prob'])
    result3['pred'] = 1*(result3['prob']>0.5)
    result3['label'] = np.array(result3['label'])
    return result3


# -*- coding: utf-8 -*-
"""
 helper function

author baiyu
"""
import os
import sys
import re
import datetime

import numpy

import torch
from torch.optim.lr_scheduler import _LRScheduler
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
#Self designed#######################################################################################################################
# Data preprocessing
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

def get_img_info(data_dir):
        data_info = list()

        # get image and label with .png
        img_names = os.listdir(data_dir)
        img_names = list(filter(lambda x: x.endswith('.png'), img_names))
        label_df = pd.read_csv(data_dir+'Label.csv')
        label_df.index = label_df['Name']

        # 遍历图片
        for i in range(len(img_names)):
            img_name = img_names[i]
            path_img = os.path.join(data_dir, img_name)

            temp_name = img_name[7:-4]
            label1 = label_df['label1'][temp_name]
            label2 = label_df['label2'][temp_name]

            data_info.append((path_img, int(label1),int(label2)))

        return data_info

def get_img_info2(data_dir):
        data_info = list()

        # get image and label with .png
        img_names = os.listdir(data_dir)
        img_names = list(filter(lambda x: x.endswith('.png'), img_names))
        label_df = pd.read_csv(data_dir+'Label.csv')
        label_df.index = label_df['Name']

        # 遍历图片
        for i in range(len(img_names)):
            img_name = img_names[i]
            path_img = os.path.join(data_dir, img_name)

            temp_name = img_name[7:-4]
            label1 = label_df['label1'][temp_name]
            label2 = label_df['label2'][temp_name]

            data_info.append((path_img, int(label1),int(label2),label_df['TNM'][temp_name]))

        return data_info


def split_the_patient(filenames,split_rate = 0.2, balance_sample = False):
    # find number of patient
    files_temp = []
    for i in filenames:
        temp = (int(i[0][20:].split('_')[0]),i[1],i[2])
        files_temp.append(temp)
    files_temp = np.array(list(set(files_temp)))
    
    # if balance the sample (abandone some data to make it balanced)
    if balance_sample:
        index = np.where(files_temp[:,1]==0)[0][:np.sum(files_temp[:,1])]
        files_temp = np.vstack((files_temp[index,:],files_temp[files_temp[:,1]==1,:]))
        
    # Set training and validation index
    x_train_index, x_test_index, _, _ = train_test_split(files_temp[:,0], 
        files_temp[:,1], train_size=(1-split_rate), 
        test_size=split_rate,random_state=1, 
        stratify=files_temp[:,1])

    # get the training and validation set
    x_train,y_train,x_test,y_test = [],[],[],[]

    for i in filenames:
        if int(i[0][20:].split('_')[0]) in x_train_index:
            x_train.append(i)
            y_train.append(i[1])
        else:
            x_test.append(i)
            y_test.append(i[1])
    return x_train,y_train,x_test,y_test



""
def get_network(args):
    """ return given network
    """

    if args.net == 'vgg16':
        from models.vgg import vgg16_bn
        net = vgg16_bn()
    elif args.net == 'vgg13':
        from models.vgg import vgg13_bn
        net = vgg13_bn()
    elif args.net == 'vgg11':
        from models.vgg import vgg11_bn
        net = vgg11_bn()
    elif args.net == 'vgg19':
        from models.vgg import vgg19_bn
        net = vgg19_bn()
    elif args.net == 'densenet121':
        from models.densenet import densenet121
        net = densenet121()
    elif args.net == 'densenet161':
        from models.densenet import densenet161
        net = densenet161()
    elif args.net == 'densenet169':
        from models.densenet import densenet169
        net = densenet169()
    elif args.net == 'densenet201':
        from models.densenet import densenet201
        net = densenet201()
    elif args.net == 'googlenet':
        from models.googlenet import googlenet
        net = googlenet()
    elif args.net == 'inceptionv3':
        from models.inceptionv3 import inceptionv3
        net = inceptionv3()
    elif args.net == 'inceptionv4':
        from models.inceptionv4 import inceptionv4
        net = inceptionv4()
    elif args.net == 'inceptionresnetv2':
        from models.inceptionv4 import inception_resnet_v2
        net = inception_resnet_v2()
    elif args.net == 'xception':
        from models.xception import xception
        net = xception()
    elif args.net == 'resnet18':
        from models.resnet import resnet18
        net = resnet18()
    elif args.net == 'resnet34':
        from models.resnet import resnet34
        net = resnet34()
    elif args.net == 'resnet50':
        from models.resnet import resnet50
        net = resnet50()
    elif args.net == 'resnet101':
        from models.resnet import resnet101
        net = resnet101()
    elif args.net == 'resnet152':
        from models.resnet import resnet152
        net = resnet152()
    elif args.net == 'preactresnet18':
        from models.preactresnet import preactresnet18
        net = preactresnet18()
    elif args.net == 'preactresnet34':
        from models.preactresnet import preactresnet34
        net = preactresnet34()
    elif args.net == 'preactresnet50':
        from models.preactresnet import preactresnet50
        net = preactresnet50()
    elif args.net == 'preactresnet101':
        from models.preactresnet import preactresnet101
        net = preactresnet101()
    elif args.net == 'preactresnet152':
        from models.preactresnet import preactresnet152
        net = preactresnet152()
    elif args.net == 'resnext50':
        from models.resnext import resnext50
        net = resnext50()
    elif args.net == 'resnext101':
        from models.resnext import resnext101
        net = resnext101()
    elif args.net == 'resnext152':
        from models.resnext import resnext152
        net = resnext152()
    elif args.net == 'shufflenet':
        from models.shufflenet import shufflenet
        net = shufflenet()
    elif args.net == 'shufflenetv2':
        from models.shufflenetv2 import shufflenetv2
        net = shufflenetv2()
    elif args.net == 'squeezenet':
        from models.squeezenet import squeezenet
        net = squeezenet()
    elif args.net == 'mobilenet':
        from models.mobilenet import mobilenet
        net = mobilenet()
    elif args.net == 'mobilenetv2':
        from models.mobilenetv2 import mobilenetv2
        net = mobilenetv2()
    elif args.net == 'nasnet':
        from models.nasnet import nasnet
        net = nasnet()
    elif args.net == 'attention56':
        from models.attention import attention56
        net = attention56()
    elif args.net == 'attention92':
        from models.attention import attention92
        net = attention92()
    elif args.net == 'seresnet18':
        from models.senet import seresnet18
        net = seresnet18()
    elif args.net == 'seresnet34':
        from models.senet import seresnet34
        net = seresnet34()
    elif args.net == 'seresnet50':
        from models.senet import seresnet50
        net = seresnet50()
    elif args.net == 'seresnet101':
        from models.senet import seresnet101
        net = seresnet101()
    elif args.net == 'seresnet152':
        from models.senet import seresnet152
        net = seresnet152()
    elif args.net == 'wideresnet':
        from models.wideresidual import wideresnet
        net = wideresnet()
    elif args.net == 'stochasticdepth18':
        from models.stochasticdepth import stochastic_depth_resnet18
        net = stochastic_depth_resnet18()
    elif args.net == 'stochasticdepth34':
        from models.stochasticdepth import stochastic_depth_resnet34
        net = stochastic_depth_resnet34()
    elif args.net == 'stochasticdepth50':
        from models.stochasticdepth import stochastic_depth_resnet50
        net = stochastic_depth_resnet50()
    elif args.net == 'stochasticdepth101':
        from models.stochasticdepth import stochastic_depth_resnet101
        net = stochastic_depth_resnet101()
    #####ADDED######################################################################
    elif args.net == 'transfer_resnet50':
        from models.resnet import transfer_resnet50
        net = transfer_resnet50()
    elif args.net == 'transfer_resnet18':
        from models.resnet import transfer_resnet18
        net = transfer_resnet18()
    elif args.net == 'transfer_resnet101':
        from models.resnet import transfer_resnet101
        net = transfer_resnet101()
    ################################################################################
    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    if args.gpu: #use_gpu
        net = net.cuda()

    return net


def get_training_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cifar100 training dataset
        std: std of cifar100 training dataset
        path: path to cifar100 training python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: train_data_loader:torch dataloader object
    """

    transform_train = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    #cifar100_training = CIFAR100Train(path, transform=transform_train)
    cifar100_training = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    cifar100_training_loader = DataLoader(
        cifar100_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_training_loader

def get_test_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cifar100 test dataset
        std: std of cifar100 test dataset
        path: path to cifar100 test python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: cifar100_test_loader:torch dataloader object
    """

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    #cifar100_test = CIFAR100Test(path, transform=transform_test)
    cifar100_test = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    cifar100_test_loader = DataLoader(
        cifar100_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_test_loader

def compute_mean_std(cifar100_dataset):
    """compute the mean and std of cifar100 dataset
    Args:
        cifar100_training_dataset or cifar100_test_dataset
        witch derived from class torch.utils.data

    Returns:
        a tuple contains mean, std value of entire dataset
    """

    data_r = numpy.dstack([cifar100_dataset[i][1][:, :, 0] for i in range(len(cifar100_dataset))])
    data_g = numpy.dstack([cifar100_dataset[i][1][:, :, 1] for i in range(len(cifar100_dataset))])
    data_b = numpy.dstack([cifar100_dataset[i][1][:, :, 2] for i in range(len(cifar100_dataset))])
    mean = numpy.mean(data_r), numpy.mean(data_g), numpy.mean(data_b)
    std = numpy.std(data_r), numpy.std(data_g), numpy.std(data_b)

    return mean, std

class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


def most_recent_folder(net_weights, fmt):
    """
        return most recent created folder under net_weights
        if no none-empty folder were found, return empty folder
    """
    # get subfolders in net_weights
    folders = os.listdir(net_weights)

    # filter out empty folders
    folders = [f for f in folders if len(os.listdir(os.path.join(net_weights, f)))]
    if len(folders) == 0:
        return ''

    # sort folders by folder created time
    folders = sorted(folders, key=lambda f: datetime.datetime.strptime(f, fmt))
    return folders[-1]

def most_recent_weights(weights_folder):
    """
        return most recent created weights file
        if folder is empty return empty string
    """
    weight_files = os.listdir(weights_folder)
    if len(weights_folder) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'

    # sort files by epoch
    weight_files = sorted(weight_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))

    return weight_files[-1]

def last_epoch(weights_folder):
    weight_file = most_recent_weights(weights_folder)
    if not weight_file:
       raise Exception('no recent weights were found')
    resume_epoch = int(weight_file.split('-')[1])

    return resume_epoch

def best_acc_weights(weights_folder):
    """
        return the best acc .pth file in given folder, if no
        best acc weights file were found, return empty string
    """
    files = os.listdir(weights_folder)
    if len(files) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'
    best_files = [w for w in files if re.search(regex_str, w).groups()[2] == 'best']
    if len(best_files) == 0:
        return ''

    best_files = sorted(best_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))
    return best_files[-1]



###############################################################################################################
# model diagnosis
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
import pandas as pd
import math
from sklearn import metrics
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
def calculateModelScores(y_true, y_pred,y_prob, clf_name, dataset,verbal = True):
    # classification metrics
    scores = {}

    # print report
    target_names = ['class 0', 'class 1']
    text = metrics.classification_report(y_true, y_pred, target_names=target_names)
    
    #pdb.set_trace()
    #conf_mat = metrics.confusion_matrix(y_true, y_pred)
    conf_mat=pd.crosstab(y_true, y_pred,rownames=['label'],colnames=['pre'])
    if verbal:
        print(conf_mat)
        print(text)

    # accuracy
    scores['accuracy'] = metrics.accuracy_score(y_true, y_pred)

    # precision
    scores['precision'] = metrics.precision_score(y_true, y_pred)

    # recall
    scores['recall (senstitive)'] = metrics.recall_score(y_true, y_pred)
    
    scores['recall_neg (specificity)'] = metrics.recall_score(y_true==0, y_pred==0)

    # F1-score
    scores['f1_score'] = metrics.f1_score(y_true, y_pred)

    # ROC/AUC
    fpr, tpr, th = metrics.roc_curve(y_true, y_prob[:, 1])
    # print(fpr, tpr, th)
    scores['auc'] = metrics.auc(fpr, tpr)
    
    threshold=th[np.argmax(tpr - fpr)]
    scores['threshold']=threshold
    if verbal:
        print('figure ROC curve')
        figureROC(fpr, tpr, th, scores['auc'], dataset)

        # RadScore
        figureRadScore(y_true, y_prob, 'lr', dataset,threshold)

        # Decision Curve Analysis
        figureDCA(y_true, y_prob[:, 1], dataset)

    # calibration curve
    # brier_score_loss
    scores['brier_score'] = metrics.brier_score_loss(y_true, y_prob[:, 1], pos_label=y_true.max())
    return scores


def figureROC(fpr, tpr, th, auc, dataset):
    # calculate the best cut-off point by Youden index
    uindex = np.argmax(tpr - fpr)

    plt.figure()
    plt.plot(fpr, tpr, lw=1, label='ROC (area = %0.2f)' % auc)
    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
    plt.plot(fpr[uindex], tpr[uindex], 'r', markersize=8)
    plt.text(fpr[uindex], tpr[uindex], '%.3f(%.3f,%.3f)' % (th[uindex], fpr[uindex], tpr[uindex]), ha='center', va='bottom', fontsize=10)
    plt.title('ROC curve (' + dataset + ')')
    plt.legend(loc='lower right')
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.xlabel('1-Specificity')
    plt.ylabel('Sensitivity')
    plt.grid(True)
    plt.show()


def figureRadScore(y_true, y_prob, clf_name, dataset,threshold):
    if clf_name == 'lr':
        rad_score = np.log(y_prob[:, 1] / y_prob[:, 0])
        adjust=np.log(threshold/(1-threshold))
        rad_score_sort = np.sort(rad_score-adjust)
        y_true_sort = y_true[np.argsort(rad_score)]

        # figure rad score
        print("figure RadScore")
        plt.figure()
        class0 = np.where(y_true_sort == 0)[0]
        class1 = np.where(y_true_sort == 1)[0]
        
        #pdb.set_trace()

        print(rad_score_sort[class0])
        print(rad_score_sort[class1])
        plt.bar(np.array(class0).squeeze(), rad_score_sort[class0], width=0.6, color='magenta', label='0')
        plt.bar(np.array(class1).squeeze(), rad_score_sort[class1], width=0.6, color='cyan', label='1')
        plt.title('Rad-Score (' + dataset + ')')
        plt.xlabel('Bar Chart')
        plt.ylabel('Rad Score')
        #plt.grid(True)
        plt.xticks(range(len(np.argsort(rad_score))), np.argsort(rad_score))
        plt.legend(loc='lower right')
        plt.show()


def netBenefit(y_true, y_prob, threshold):
    y_pred = np.zeros(y_prob.shape)
    y_pred[np.where(y_prob > threshold)] = 1
    num = y_true.size
    tp_num = len(np.intersect1d(np.where(y_true == 1), np.where(y_pred == 1)))
    fp_num = len(np.intersect1d(np.where(y_true == 0), np.where(y_pred == 1)))

    tpr = tp_num / num
    fpr = fp_num / num

    NB = tpr - fpr * threshold / (1 - threshold)

    return NB


def figureDCA(y_true, y_prob, dataset):
    thresholds = np.array([i/100.0 for i in range(100)])
    y_prob_0 = np.zeros(y_prob.shape)
    y_prob_1 = np.ones(y_prob.shape)

    net_benefits = np.zeros(100)
    net_benefits_0 = np.zeros(100)  # None
    net_benefits_1 = np.zeros(100)  # All
    for i in range(100):
        th = thresholds[i]
        net_benefits[i] = netBenefit(y_true, y_prob, th)
        net_benefits_0[i] = netBenefit(y_true, y_prob_0, th)
        net_benefits_1[i] = netBenefit(y_true, y_prob_1, th)

    # figure DC
    print("figure decision curve")
    plt.figure()
    plt.plot(thresholds, net_benefits, color='blue', label='LR')
    plt.plot(thresholds, net_benefits_0, color='black', label='None')
    plt.plot(thresholds, net_benefits_1, color='red', label='All')
    plt.legend(loc='lower right')
    plt.title('DCA (' + dataset + ')')
    plt.xlabel('Threshold probability')
    plt.ylabel('Net benefit')
    plt.ylim(-0.5, 0.5)
    #plt.grid(True)
    plt.show()
""
def find_patient(filenames):
    # find number of patient
    files_temp = []
    for i in filenames:
#         print(i)
#         temp = int(i[20:].split('_')[0])
        temp = int(i.split('/')[-1].split('_')[0][7:])
        files_temp.append(temp)
    files_temp = np.array(list(set(files_temp)))
    return files_temp
def integrate_patient(result):# Garbage design
    # build dict to store prob
    result2 = {}
    label2 = {}
    for i in find_patient(result['name']):
        result2[i] = []

    for i in range(len(result['label'])):
#         temp = int(result['name'][i][20:].split('_')[0])
        temp = int(result['name'][i].split('/')[-1].split('_')[0][7:])
        result2[temp].append(result['prob'][i])
        label2[temp]=result['label'][i]
    # average prob
    for i in result2.keys():
        result2[i] = np.mean(result2[i],axis=0)
    # rearrage
    result3 = {}
    result3['label'],result3['pred'],result3['prob'],result3['name']=[],[],[],[]
    for i in result2.keys():
        try:
            result3['name'].append(i)
        except:
            pass
        result3['prob'].append(result2[i])
        result3['label'].append(label2[i])
    result3['prob'] = np.array(result3['prob'])
    _,result3['pred'] = torch.tensor(result3['prob']).max(1)
    result3['pred'] = result3['pred'].numpy()
    return result3

