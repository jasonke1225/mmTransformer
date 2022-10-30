# -*- coding: utf-8 -*-
import argparse
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from functools import partial

from config.Config import Config
# ============= Dataset =====================
from lib.dataset.collate import collate_single_cpu
from lib.dataset.dataset_for_argoverse import STFDataset as ArgoverseDataset
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
    parser.add_argument('--model-name', type=str, default='demo')
    parser.add_argument('--model-save-path', type=str, default='./models/')

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
            loss_sum += (loss/(self.params[i] * self.params[i]) + torch.log(self.params[i]+1))
            # +1避免了log 0的问题  log sigma部分对于整体loss的影响不大
        return loss_sum

def train(model, dataloader, optimizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    

    total_loss_score = 0
    for j, data in enumerate(tqdm(dataloader)):
        for key in data.keys():
            if isinstance(data[key], torch.Tensor):
                if(data[key].device != device):
                    data[key] = data[key].to(device)

        out = model(data)
        gt = data['FUTURE']
        gt_trj = torch.tensor([])
        for i in range(len(gt)):
            tmp_trj = gt[i][0,:,:2].unsqueeze(0)
            gt_trj = torch.cat((gt_trj, tmp_trj), 0)

        gt_trj = gt_trj.to(device) #取每個batch的target agent的GT未來軌跡 (batch num, 30, 2)
        out_score = out[1][:,0]
        out_trj = out[0][:,0]
        pred_trj = out_trj.permute(1,0,2,3).to(device) #取target agent的未來軌跡，並轉成(6,batch num, 30, 2)

        pdist = torch.nn.PairwiseDistance(p=2) # L2 distance
        fde = pdist(pred_trj[:,:,-1,:], gt_trj[:,-1,:]) # 計算 L2 distance of 6個pred_trj和gt_trj的end point(找FDE), (6, batch_num)
        fde = fde.permute(1,0) # (batch_num, 6)
        D_s = torch.min(fde, dim=1)[0] # L2 of min_fde_endpoint and GT endpoint
        min_fde_index = torch.argmin(fde, dim=1)
        min_fde_trj = torch.tensor([]).to(device)
        C = torch.tensor([]).to(device) # min_fde_score (W_y)
        
        for i in range(out_trj.shape[0]):
            tmp_trj = out_trj[i, min_fde_index[i].item()].unsqueeze(0).to(device)
            min_fde_trj = torch.cat((min_fde_trj, tmp_trj), 0).to(device)
            tmp_score = out_score[i][min_fde_index[i].item()].unsqueeze(0)
            C = torch.cat((C, tmp_score),0)
        
        # regression loss
        crit = torch.nn.HuberLoss().to(device)
        regression_loss = crit(min_fde_trj, gt_trj)

        # confidence loss
        C_exp = torch.exp(C)
        T_y = C_exp / torch.sum(C_exp)
        D_s_exp = torch.exp(-1*D_s)
        lamda_s = D_s_exp / torch.sum(D_s_exp)
        confidence_loss = torch.nn.functional.kl_div(T_y.softmax(dim=-1).log(), lamda_s.softmax(dim=-1)).to(device)

        # classification loss
        P = C # P is the sum of pred proposal in GT region，而沒用RTS時則相當於minFDE的pred proposal's sum
        indicator = 1 # 因為都是minFDE的pred proposal，可以看成是預測軌跡一定跟GT區域一樣，所以indicator一定是1
        sum_P_log = torch.log(P).sum()
        classification_loss = -1 * sum_P_log

        # weight losses
        loss_sum = awl(regression_loss, confidence_loss, classification_loss)
        loss_sum.backward()
        total_loss_score += float(loss_sum.item())

        #if(j%2==0 or j==(len(dataloader)-1)):
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1) # gradient max norm is 0.1
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # if(j%50==0 or j==(len(dataloader)-1)):
        #     del out
        #     del gt_trj
        #     del pred_trj
        #     del loss_score
        #     gc.collect()
        #     torch.cuda.empty_cache()

    total_loss_score /= len(dataloader)

    return total_loss_score

def val(model, dataloader, optimizer):
    """
    nearly same as train, except backward and opt
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    total_loss_score = 0
    with torch.no_grad():
        for j, data in enumerate(tqdm(dataloader)):
            for key in data.keys():
                if isinstance(data[key], torch.Tensor):
                    if(data[key].device != device):
                        data[key] = data[key].to(device)

            out = model(data)
            gt = data['FUTURE']
            gt_trj = torch.tensor([])
            for i in range(len(gt)):
                tmp_trj = gt[i][0,:,:2].unsqueeze(0)
                gt_trj = torch.cat((gt_trj, tmp_trj), 0)

            gt_trj = gt_trj.to(device)
            out_score = out[1][:,0]
            out_trj = out[0][:,0]
            pred_trj = out_trj.permute(1,0,2,3).to(device)

            pdist = torch.nn.PairwiseDistance(p=2)
            fde = pdist(pred_trj[:,:,-1,:], gt_trj[:,-1,:])
            fde = fde.permute(1,0)
            D_s = torch.min(fde, dim=1)[0]
            min_fde_index = torch.argmin(fde, dim=1)
            min_fde_trj = torch.tensor([]).to(device)
            C = torch.tensor([]).to(device)
            
            for i in range(out_trj.shape[0]):
                tmp_trj = out_trj[i, min_fde_index[i].item()].unsqueeze(0).to(device)
                min_fde_trj = torch.cat((min_fde_trj, tmp_trj), 0).to(device)
                tmp_score = out_score[i][min_fde_index[i].item()].unsqueeze(0)
                C = torch.cat((C, tmp_score),0)
            
            # regression loss
            crit = torch.nn.HuberLoss().to(device)
            regression_loss = crit(min_fde_trj, gt_trj)

            # confidence loss
            C_exp = torch.exp(C)
            T_y = C_exp / torch.sum(C_exp)
            D_s_exp = torch.exp(-1*D_s)
            lamda_s = D_s_exp / torch.sum(D_s_exp)
            confidence_loss = torch.nn.functional.kl_div(T_y.softmax(dim=-1).log(), lamda_s.softmax(dim=-1)).to(device)

            # classification loss
            P = C
            indicator = 1
            sum_P_log = torch.log(P).sum()
            classification_loss = -1 * sum_P_log

            # weight losses
            loss_sum = awl(regression_loss, confidence_loss, classification_loss)
            total_loss_score += float(loss_sum.item())

        total_loss_score /= len(dataloader)

    return total_loss_score

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.cuda.empty_cache()
    gc.collect()
    start_time = time.time()
    gpu_num = torch.cuda.device_count()
    print("gpu number:{}".format(gpu_num))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

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
    # =================================== Metric Initial =======================================================
    format_results = FormatData()
    # evaluate = partial(compute_forecasting_metrics,
    #                    max_n_guesses=6,
    #                    horizon=30,
    #                    miss_threshold=2.0)
    # =================================== INIT MODEL ===========================================================
    model_cfg = cfg.get('model')
    stacked_transfomre = load_model_class(model_cfg['type'])
    model = mmTrans(stacked_transfomre, model_cfg)
    # find pt in directory
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dir_path += "/ckpt"
    all_file_name = os.listdir(dir_path)
    num = 0
    if(len(all_file_name)>0):
        for i in range(len(all_file_name)):
            tmp = all_file_name[i].split('.')
            tmp = tmp[0].split('_')
            if(num<int(tmp[1])):
                num = int(tmp[1])

        model_name =  dir_path + "/model_"+str(num)+".pt"
        model = load_checkpoint(model_name, model)

    model = model.to(device)
    awl = AutomaticWeightedLoss(3)
    lr_rate = 10 ** -3 # origin is 0.001
    optimizer = torch.optim.AdamW([
                {'params':model.parameters(), 'lr':lr_rate, 'weight_decay':0.0001},
                {'params':awl.parameters(), 'lr':lr_rate, 'weight_decay':0}])
    optimizer.zero_grad() #set_to_none=True
    epochs = 100
    min_loss = 100
    min_index = 0
    for i in range(epochs):
        train_loss = train(model, train_dataloader, optimizer)
        val_loss = val(model, val_dataloader, optimizer)
        print('[epoch %d] train loss: %.6f' %(i + 1, train_loss))
        print('[epoch %d] val loss: %.6f' %(i + 1, val_loss))
        if i % 5 == 4:
            save_checkpoint_dir = dir_path + "/model_"+ str(i+num+1)+".pt"
            save_checkpoint(save_checkpoint_dir, model, optimizer)

        if(train_loss<min_loss):
            min_loss = train_loss
            min_index = i

        if (i - min_index > 9):
            lr_rate *= 0.1
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr_rate, weight_decay=0.0001)
            print("lr_rate * 0.1 = ",lr_rate)

    print('Train Process Finished!!')
