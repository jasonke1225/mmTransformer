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

def parse_args():

    parser = argparse.ArgumentParser(description='Evaluate the mmTransformer')
    parser.add_argument('config', help='config file path')
    parser.add_argument('--model-name', type=str, default='demo')
    parser.add_argument('--model-save-path', type=str, default='./models/')

    args = parser.parse_args()
    return args

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
        gt_end_point = gt_trj[:,29,:].to(device)
        pred_trj = out[0][:,0].permute(1,0,2,3).to(device) #取target agent的未來軌跡，並轉成(6,batch num, 30, 2)
        
        crit = torch.nn.HuberLoss().to(device)
        regression_loss = crit(pred_trj, gt_trj)
        confidence_loss = 0
        for i in range(pred_trj.shape[0]):
            pred_end_point = pred_trj[i,:,29,:].to(device)
            confidence_loss += torch.nn.functional.kl_div(pred_end_point.softmax(dim=-1).log(), gt_end_point.softmax(dim=-1)).to(device)
        
        confidence_loss /= 6
        loss_score = regression_loss + confidence_loss
        #loss_score.requires_grad_(True)
        loss_score.backward()
        total_loss_score += float(loss_score.item())

        #if(j%2==0 or j==(len(dataloader)-1)):
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1) # gradient max norm is 0.1
        optimizer.step()
        optimizer.zero_grad(set_to_none=True) # set_to_none=True

        # if(j%50==0 or j==(len(dataloader)-1)):
        #     del out
        #     del gt_trj
        #     del pred_trj
        #     del loss_score
        #     gc.collect()
        #     torch.cuda.empty_cache()

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
    dir_path += "/ckpt_64_loss2"
    all_file_name = os.listdir(dir_path)
    num = 0
    if(len(all_file_name)>0):
        for i in range(len(all_file_name)):
            tmp = all_file_name[i].split('.')
            tmp = tmp[0].split('_')
            if(num<int(tmp[1])):
                num = int(tmp[1])

        model_name =  "ckpt_64_loss2/model_"+str(num)+".pt"
        model = load_checkpoint(model_name, model)

    model = model.to(device)
    lr_rate = 10 ** -4 # origin is 0.001
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_rate, weight_decay=0.0001)
    optimizer.zero_grad() #set_to_none=True
    epochs = 100
    min_loss = 100
    min_index = 0
    for i in range(epochs):
        running_loss = train(model, train_dataloader, optimizer)
        print('[epoch %d] loss: %.6f' %(i + 1, running_loss))
        if(running_loss<min_loss):
            min_loss = running_loss
            min_index = i
        if i % 5 == 4:
            save_checkpoint_dir = "ckpt_64_loss2/model_"+ str(i+num+1)+".pt"
            save_checkpoint(save_checkpoint_dir, model, optimizer)
        if (i - min_index > 9):
            lr_rate *= 0.1
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr_rate, weight_decay=0.0001)
            print("lr_rate * 0.1 = ",lr_rate)

    print('Train Process Finished!!')
