#引入模块
#数据操作
import math
from os import write

import numpy as np
#读取写入数据
import pandas as pd
import os
import csv

from torch.xpu import device
#进度条
from tqdm import tqdm
#Pytorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset,random_split
#绘制图像
from tensorboardX import SummaryWriter


#设置随机种子
def same_seed(seed):
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False  #保证实验可重复
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

#划分数据集
def train_valid_split(data_set,valid_ratio,seed):
    valid_data_size=int(len(data_set)*valid_ratio)   #验证集
    train_data_size=len(data_set)-valid_data_size  #训练集
    train_data,valid_data=random_split(data_set,[train_data_size,valid_data_size],generator=torch.Generator().manual_seed(seed))
    return np.array(train_data),np.array(valid_data)

#选择特征
def select_feat(train_data,valid_data,test_data,select_all=True):
    y_train=train_data[:,-1]#全部的行，倒数第一个元素
    y_valid=valid_data[:,-1]

    raw_x_train=train_data[:,:-1]#除了最后一列
    raw_x_valid=valid_data[:,:-1]
    raw_x_test=test_data

    if select_all:
        feat_idx=list(range(raw_x_train.shape[1]))
    else:
        feat_idx=[0,1,2,3,4]

    return raw_x_train[:,feat_idx],raw_x_valid[:,feat_idx],raw_x_test[:,feat_idx],y_train,y_valid

#数据集
class COVID19Dataset(Dataset):
    def __init__(self,features,targets=None):
        if targets is None:
            self.targets=targets
        else:
            self.targets=torch.FloatTensor(targets)
        self.features=torch.FloatTensor(features)

    def __getitem__(self, idx):
        if self.targets is None:
            return self.features[idx]
        else:
            return self.features[idx], self.targets[idx]

    def __len__(self):
        return len(self.features)

#神经网络
class My_Model(nn.Module):
    def __init__(self,input_dim):
        super(My_Model, self).__init__()
        self.layers=nn.Sequential(
            nn.Linear(input_dim,16),
            nn.ReLU(),
            nn.Linear(16,8),
            nn.ReLU(),
            nn.Linear(8,1),
        )

    def forward(self,x):
        x=self.layers(x)
        x=x.squeeze(1)
        return x

#参数设置
device='cuda'if torch.cuda.is_available() else 'cpu'
config={
    'seed':5201314,
    'select_all':True,
    'valid_ratio':0.2,
    'n_epochs':3000,
    'batch_size':256,
    'learning_rate':1e-5,
    'early_stop':400,
    'save_path':'./models/model.ckpt'#模型存储路径
}

#训练过程
def trainer(train_loader,valid_loader,model,config,device):
    criterion=nn.MSELoss(reduction='mean')
    optimizer=torch.optim.SGD(model.parameters(),lr=config['learning_rate'],momentum=0.9)
    writer=SummaryWriter()
    if not os.path.exists('./models'):
        os.mkdir('./models')

    n_epochs=config['n_epochs']
    best_loss=math.inf
    step=0
    early_stop_count=0

    for epoch in range(n_epochs):
        model.train()
        loss_record=[]
        train_pbar=tqdm(train_loader,position=0,leave=True)

        #xunlianguocheng
        for x,y in train_pbar:
            optimizer.zero_grad()
            x,y=x.to(device),y.to(device)
            pred=model(x)
            loss=criterion(pred,y)
            loss.backward()
            optimizer.step()
            step+=1
            loss_record.append(loss.detach().item())


            train_pbar.set_description(f'Epoch[{epoch+1}/{n_epochs}]')
            train_pbar.set_postfix({'loss': loss.detach().item()})

        mean_train_loss=sum(loss_record)/len(loss_record)
        writer.add_scalar('Loss/train',mean_train_loss,step)

        #yanzheng
        model.eval()
        loss_record=[]

        for x,y in valid_loader:
            x,y=x.to(device),y.to(device)
            with torch.no_grad():
                pred=model(x)
                loss=criterion(pred,y)
            loss_record.append(loss.item())

        mean_valid_loss=sum(loss_record)/len(loss_record)
        print(f'Epoch[{epoch+1}/{n_epochs}]:Train loss: {mean_train_loss:.4f},Valid loss: {mean_valid_loss:.4f}')
        writer.add_scalar('Loss/valid',mean_valid_loss,step)

        if mean_valid_loss < best_loss:
            best_loss=mean_valid_loss
            torch.save(model.state_dict(),config['save_path'])
            print('Saving model with loss {:.3f}...'.format(best_loss))
            early_stop_count=0
        else:
            early_stop_count+=1

        if early_stop_count >= config['early_stop']:
            print('\n Model is not improving,so we halt the training sessiom.')
            return


"'准备工作'"
#设置随机种子
same_seed(config['seed'])
#读取数据
train_data=pd.read_csv('./covid.train.csv').values
test_data=pd.read_csv('./covid.test.csv').values
#划分数据集
train_data,valid_data=train_valid_split(train_data,config['valid_ratio'],config['seed'])
print(f"""train_data size:{train_data.shape},valid_data size:{valid_data.shape}),test_data size:{test_data.shape})""")
#选择特征
x_train,x_valid,x_test,y_train,y_valid=select_feat(train_data,valid_data,test_data,config['select_all'])
print(f'the number of features is: {x_train.shape[1]}')
#构造数据集
train_dataset=COVID19Dataset(x_train,y_train)
valid_dataset=COVID19Dataset(x_valid,y_valid)
test_dataset=COVID19Dataset(x_test)
#准备Dataloader
train_loader=DataLoader(train_dataset,batch_size=config['batch_size'],shuffle=True,pin_memory=True)
valid_loader=DataLoader(valid_dataset,batch_size=config['batch_size'],shuffle=True,pin_memory=True)
test_loader=DataLoader(test_dataset,batch_size=config['batch_size'],shuffle=False,pin_memory=True)

#开始训练
model=My_Model(input_dim=x_train.shape[1]).to(device)
trainer(train_loader,valid_loader,model,config,device)


#预测
def predict(test_loader,model,device):
    model.eval()
    preds=[]
    for x in tqdm(test_loader):
        x=x.to(device)
        with torch.no_grad():
            pred=model(x)
            preds.append(pred.detach().cpu())
    preds=torch.cat(preds,dim=0).numpy()
    return preds

def save_pred(pred,file):
    with open(file,'w') as fp:
        writer=csv.writer(fp)
        writer.writerows(['id','tested_positive'])
        for i,p in enumerate(pred):
            writer.writerow([i,p])

#预测并保存结果
model =My_Model(input_dim=x_train.shape[1]).to(device)
model.load_state_dict(torch.load(config['save_path']))
preds=predict(test_loader,model,device)
save_pred(preds,'pred.csv')