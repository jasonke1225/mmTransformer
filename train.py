# -*- coding: utf-8 -*-
import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from functools import partial

from config.Config import Config
# ============= Dataset =====================
from lib.dataset.collate import collate_single_cpu
from lib.dataset.dataset_for_argoverse import STFDataset as ArgoverseDataset
from lib.dataset.utils import transform_coord
# ============= Models ======================
from lib.models.mmTransformer import mmTrans
from lib.utils.evaluation_utils import compute_forecasting_metrics, FormatData

# from lib.utils.traj_nms import traj_nms
from lib.utils.utilities import load_checkpoint, load_model_class, save_checkpoint
import gc
import warnings
import math

def parse_args():

    parser = argparse.ArgumentParser(description='Evaluate the mmTransformer')
    parser.add_argument('config', help='config file path')
    # parser.add_argument('--model-name', type=str, default='demo')
    # parser.add_argument('--model-save-path', type=str, default='./models/')

    args = parser.parse_args()
    return args

class AutomaticWeightedLoss(torch.nn.Module):
    """automatically weighted multi-task loss
    log(sigma)=a(a是一个可学习的变量)
    torch.exp(-a)=torch.exp(-log(sigma))=torch.exp(log(sigma**-1))=1/sigma
    Params:
        num: int, the number of loss
        x: multi-task loss
    Examples:
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """
    def __init__(self, num=3):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True) # params is 變數 sigma
        self.params = torch.nn.Parameter(params) 

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += (loss / (self.params[i] ** 2) + torch.log(1 + self.params[i]))
            # +1避免了log 0的问题  log sigma部分对于整体loss的影响不大
        return loss_sum

def calculate_loss(min_fde_trj, gt_trj, C):
    '''
    vanilla training strategy: only using the proposal with the minimum final displacement error
    vanilla training strategy dont need confidence score
    '''
    # regression loss
    crit = torch.nn.HuberLoss().to(device)
    regression_loss = crit(min_fde_trj, gt_trj)

    # confidence loss
    # T_y = 1
    # lamda_s = 1
    # confidence_loss = torch.nn.functional.kl_div(T_y.softmax(dim=-1).log(), lamda_s.softmax(dim=-1)).to(device)

    # classification loss
    label = torch.ones(gt_trj.shape[0]).to(device)
    P = C # P is the sum of pred proposal in GT region，而沒用RTS時則相當於minFDE的pred proposal's sum
    indicator = 1 # 因為都是minFDE的pred proposal，可以看成是預測軌跡一定跟GT區域一樣，所以indicator一定是1
    classification_loss = -1 * torch.log(P)
    classification_loss = classification_loss.mean() # batch size內要取平均

    # loss_sum = awl(regression_loss, confidence_loss, classification_loss)
    loss_sum = awl(regression_loss, classification_loss)
    return loss_sum


def renormalized(out_trj, data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    names = data['NAME'] # name is file name
    togloble = data['NORM_CENTER']
    theta = torch.Tensor(data['THETA'])
    city_name = data['CITY_NAME']
    gt_trj = torch.tensor([]).to(device)
    pred_trj = torch.tensor([]).to(device)
    for i, name in enumerate(names):
        # Renormalized the predicted traj
        pred = transform_coord(out_trj[i], -theta[i])
        pred = pred + torch.tensor(togloble[i]).to(device)
        pred = pred.unsqueeze(0)
        pred_trj = torch.cat((pred_trj, pred), 0)

        gt = data['FUTURE'][i][0,:,:2].to(device)
        gt = gt.cumsum(axis=-2)
        gt = transform_coord(gt, -theta[i])
        gt = gt + torch.tensor(togloble[i]).to(device)
        gt = gt.unsqueeze(0)
        gt_trj = torch.cat((gt_trj, gt), 0)
    
    return pred_trj, gt_trj

def train(model, dataloader, optimizer, epoch):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()

    total_loss_score = 0
    min_fde = 0
    for j, data in enumerate(tqdm(dataloader)):
        for key in data.keys():
            if isinstance(data[key], torch.Tensor):
                if(data[key].device != device):
                    data[key] = data[key].to(device)

        out = model(data)
        # print('out', out.shape)

        out_score = out[1][:,0]
        out_trj = out[0][:,0]
        '''
        renormalized
        note: renormalized, fde算法有檢查過，是對的
        '''
        pred_trj, gt_trj = renormalized(out_trj, data)
        pred_trj = pred_trj.permute(1,0,2,3).to(device) #取target agent的未來軌跡，並轉成(6,batch num, 30, 2)
        fde = torch.linalg.norm(pred_trj[:,:,-1,:]-gt_trj[:,-1,:], axis=-1)
        fde = fde.permute(1,0) # (batch_num, 6) 
        D_s = torch.min(fde, dim=1)[0] # L2 of min_fde_endpoint and GT endpoint
        min_current_fde = torch.mean(D_s)
        min_fde += min_current_fde.item()

        min_fde_index = torch.argmin(fde, dim=1)
        min_fde_trj = torch.tensor([]).to(device)
        C = torch.tensor([]).to(device) # min_fde_score (W_y)
        
        pred_trj = pred_trj.permute(1,0,2,3).to(device) #轉回(batch num, 6, 30, 2)

        for i in range(out_trj.shape[0]):
            tmp_trj = pred_trj[i, min_fde_index[i].item()].unsqueeze(0).to(device)
            min_fde_trj = torch.cat((min_fde_trj, tmp_trj), 0).to(device)
            tmp_score = out_score[i][min_fde_index[i].item()].unsqueeze(0) # sum of out_score[i] is 1
            C = torch.cat((C, tmp_score),0)

        # weight losses
        loss_sum = calculate_loss(min_fde_trj, gt_trj, C)
        optimizer.zero_grad(set_to_none=True)
        loss_sum.backward()
        total_loss_score += float(loss_sum.item())

        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1) # gradient max norm is 0.1
        optimizer.step()

    total_loss_score /= len(dataloader)
    min_fde /= len(dataloader)
    
    return total_loss_score, min_fde

def val(model, dataloader, optimizer, epoch):
    """
    nearly same as train, except backward and opt
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    total_loss_score = 0
    min_fde = 0
    with torch.no_grad():
        for j, data in enumerate(tqdm(dataloader)):
            for key in data.keys():
                if isinstance(data[key], torch.Tensor):
                    if(data[key].device != device):
                        data[key] = data[key].to(device)

            out = model(data)

            out_score = out[1][:,0]
            out_trj = out[0][:,0]
            '''
            renormalized
            '''
            pred_trj, gt_trj = renormalized(out_trj, data)

            pred_trj = pred_trj.permute(1,0,2,3).to(device) #取target agent的未來軌跡，並轉成(6,batch num, 30, 2)
            fde = torch.sqrt((pred_trj[:,:,-1,0] - gt_trj[:,-1,0]) ** 2 + 
                        (pred_trj[:,:,-1,1] - gt_trj[:,-1,1]) ** 2)
            fde = fde.permute(1,0) # (batch_num, 6) 
            D_s = torch.min(fde, dim=1)[0] # L2 of min_fde_endpoint and GT endpoint
            min_current_fde = torch.mean(D_s)
            min_fde += min_current_fde.item()

            min_fde_index = torch.argmin(fde, dim=1)
            min_fde_trj = torch.tensor([]).to(device)
            C = torch.tensor([]).to(device) # min_fde_score (W_y)
            
            pred_trj = pred_trj.permute(1,0,2,3).to(device) #轉回(batch num, 6, 30, 2)
            for i in range(out_trj.shape[0]):
                tmp_trj = pred_trj[i, min_fde_index[i].item()].unsqueeze(0).to(device)
                min_fde_trj = torch.cat((min_fde_trj, tmp_trj), 0).to(device)
                tmp_score = out_score[i][min_fde_index[i].item()].unsqueeze(0)
                C = torch.cat((C, tmp_score),0)

            # weight losses
            loss_sum = calculate_loss(min_fde_trj, gt_trj, C)
            total_loss_score += float(loss_sum.item())

        total_loss_score /= len(dataloader)
        min_fde /= len(dataloader)

    return total_loss_score, min_fde

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.cuda.empty_cache()
    gc.collect()
    start_time = time.time()
    gpu_num = torch.cuda.device_count()
    print("gpu number:{}".format(gpu_num))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parse_args()
    cfg = Config.fromfile(args.config)

    # ================================== INIT DATASET ==========================================================
    train_cfg = cfg.get('train_dataset')
    train_dataset = ArgoverseDataset(train_cfg)
    train_dataloader = DataLoader(train_dataset,
                                shuffle=train_cfg["shuffle"],
                                batch_size=train_cfg["batch_size"],
                                num_workers=train_cfg["num_workers"],
                                collate_fn=collate_single_cpu)

    validation_cfg = cfg.get('val_dataset')
    val_dataset = ArgoverseDataset(validation_cfg)
    val_dataloader = DataLoader(val_dataset,
                                shuffle=validation_cfg["shuffle"],
                                batch_size=validation_cfg["batch_size"],
                                num_workers=validation_cfg["workers_per_gpu"],
                                collate_fn=collate_single_cpu)
                                
    # =================================== INIT MODEL ===========================================================
    model_cfg = cfg.get('model')
    stacked_transfomre = load_model_class(model_cfg['type'])
    print(model_cfg)
    model = mmTrans(stacked_transfomre, model_cfg)


    awl = AutomaticWeightedLoss(2)
    lr_rate = 1e-4 # origin is 0.001
    optimizer = torch.optim.AdamW([
                {'params':model.parameters(), 'lr':lr_rate, 'weight_decay':0.0001},
                {'params':awl.parameters(), 'lr':lr_rate, 'weight_decay':0.0001}])

    dir_path = os.path.dirname(os.path.realpath(__file__))
    dir_path += "/ckpt"
    all_file_name = os.listdir(dir_path)
    num = 0
    model = model.to(device)
    if(len(all_file_name)>0):
        for i in range(len(all_file_name)):
            tmp = all_file_name[i].split('.')
            tmp = tmp[0].split('_')
            if(num<int(tmp[1])):
                num = int(tmp[1])

        model_name =  dir_path + "/model_"+str(num)+".pt"
        model, optimizer, awl = load_checkpoint(model_name, model, optimizer, awl)
    
    optimizer.zero_grad(set_to_none=True) #set_to_none=True
    epochs = 100
    min_loss = 100
    min_index = 0
    for i in range(epochs):          
        train_loss, train_min_fde = train(model, train_dataloader, optimizer, i)
        val_loss, val_min_fde = val(model, val_dataloader, optimizer, i)
        print('[epoch %d] train loss: %.6f' %(i + 1, train_loss), "min_fde: %.6f" % train_min_fde)
        print('[epoch %d] val loss: %.6f' %(i + 1, val_loss), "min_fde: %.6f" % val_min_fde)
        if i % 5 == 4:
            save_checkpoint_dir = dir_path + "/model_"+ str(i+num+1)+".pt"
            save_checkpoint(save_checkpoint_dir, model, optimizer, awl, train_loss, val_loss)

    print('Train Process Finished!!')
