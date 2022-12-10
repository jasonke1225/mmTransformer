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
from lib.utils.utilities import load_checkpoint, load_model_class
import gc

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

def calculate_loss(min_fde_trj, gt_trj, C):
    '''
    vanilla training strategy: only using the proposal with the minimum final displacement error
    vanilla training strategy dont need confidence score
    '''
    # regression loss
    crit = torch.nn.HuberLoss().to(device)
    regression_loss = crit(min_fde_trj, gt_trj)
    
    # classification loss
    label = torch.ones(gt_trj.shape[0]).to(device)
    P = C # P is the sum of pred proposal in GT region，而沒用RTS時則相當於minFDE的pred proposal's sum
    indicator = 1 # 因為都是minFDE的pred proposal，可以看成是預測軌跡一定跟GT區域一樣，所以indicator一定是1
    classification_loss = -1 * torch.log(P)
    classification_loss = classification_loss.mean() # batch size內要取平均
    # classification_loss = F.cross_entropy(torch.log(P), label, reduction='mean').to(device)

    # loss_sum = awl(regression_loss, confidence_loss, classification_loss)
    # loss_sum = awl(regression_loss, classification_loss)
    loss_sum = regression_loss
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
        pred = transform_coord_tensor(out_trj[i], -theta[i])
        pred = pred + torch.tensor(togloble[i]).to(device)
        pred = pred.unsqueeze(0)
        pred_trj = torch.cat((pred_trj, pred), 0)

        gt = data['FUTURE'][i][0,:,:2].to(device)
        gt = gt.cumsum(axis=-2)
        gt = transform_coord_tensor(gt, -theta[i])
        gt = gt + torch.tensor(togloble[i]).to(device)
        gt = gt.unsqueeze(0)
        gt_trj = torch.cat((gt_trj, gt), 0)
    
    return pred_trj, gt_trj

def transform_coord_tensor(coords, angle):
    x = coords[..., 0]
    y = coords[..., 1]
    x_transform = torch.cos(angle)*x-torch.sin(angle)*y
    y_transform = torch.cos(angle)*y+torch.sin(angle)*x
    output_coords = torch.stack((x_transform, y_transform), axis=-1)

    return output_coords
    
if __name__ == "__main__":

    start_time = time.time()
    gpu_num = torch.cuda.device_count()
    print("gpu number:{}".format(gpu_num))

    args = parse_args()
    cfg = Config.fromfile(args.config)

    # ================================== INIT DATASET ==========================================================
    validation_cfg = cfg.get('train_dataset')
    # print("validation_cfg: ", validation_cfg)
    val_dataset = ArgoverseDataset(validation_cfg)
    val_dataloader = DataLoader(val_dataset,
                                shuffle=validation_cfg["shuffle"],
                                batch_size=validation_cfg["batch_size"],
                                num_workers=validation_cfg["workers_per_gpu"],
                                collate_fn=collate_single_cpu)
    # =================================== Metric Initial =======================================================
    format_results = FormatData()
    evaluate = partial(compute_forecasting_metrics,
                       max_n_guesses=6,
                       horizon=30,
                       miss_threshold=2.0)
    # =================================== INIT MODEL ===========================================================
    # pretrain_model name
    model_cfg = cfg.get('model')
    stacked_transfomre = load_model_class(model_cfg['type'])
    model = mmTrans(stacked_transfomre, model_cfg).cuda()
    model_name = os.path.join(args.model_save_path,
                              '{}.pt'.format(args.model_name))

    # train by my self's model name
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dir_path += "/ckpt_128_4worker"
    all_file_name = os.listdir(dir_path)
    num = 0
    if(len(all_file_name)>0):
        for i in range(len(all_file_name)):
            tmp = all_file_name[i].split('.')
            tmp = tmp[0].split('_')
            if(num<int(tmp[1])):
                num = int(tmp[1])

        state_name =  dir_path + "/model_"+str(num)+".pt"
    state = torch.load(state_name)
    model.load_state_dict(state['state_dict'])
    # awl_state = torch.load(state_name)
    # awl = AutomaticWeightedLoss(2)
    # awl.load_state_dict(state['awl'])
    print('Successfully Loaded model: {}'.format(state_name))
    print('Finished Initialization in {:.3f}s!!!'.format(
        time.time()-start_time))
    # ==================================== EVALUATION LOOP =====================================================
    model.eval()
    progress_bar = tqdm(val_dataloader)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_loss_score = 0
    with torch.no_grad():
        for j, data in enumerate(progress_bar):
            for key in data.keys():
                if isinstance(data[key], torch.Tensor):
                    data[key] = data[key].to(device)

            out = model(data)
            format_results(data, out) #會累加file的資訊進去
            # if(j==0):
            #     out_score = out[1][:,0]
            #     out_trj = out[0][:,0]
            #     pred_trj, gt_trj = renormalized(out_trj, data)
            #     pred_trj = pred_trj.permute(1,0,2,3).to(device)
                
            #     print("pred_trj.shape ",pred_trj.shape)
            #     fde = torch.sqrt((pred_trj[:,:,-1,0] - gt_trj[:,-1,0]) ** 2 + 
            #         (pred_trj[:,:,-1,1] - gt_trj[:,-1,1]) ** 2)
            #     fde = fde.permute(1,0) # (batch_num, 6) 
            #     D_s = torch.min(fde, dim=1)[0]
            #     min_current_fde = torch.mean(D_s)
            #     print("min_current_fde", min_current_fde) 
            #     #min_fde += min_current_fde.item()

            #     min_fde_index = torch.argmin(fde, dim=1)
            #     print("min_fde_index", min_fde_index)
            #     min_fde_trj = torch.tensor([]).to(device)
            #     C = torch.tensor([]).to(device) # min_fde_score (W_y)
            #     pred_trj = pred_trj.permute(1,0,2,3).to(device)
            #     print("pred_trj.shape ",pred_trj.shape)
            #     for i in range(out_trj.shape[0]):
            #         tmp_trj = pred_trj[i, min_fde_index[i].item()].unsqueeze(0).to(device)
            #         min_fde_trj = torch.cat((min_fde_trj, tmp_trj), 0).to(device)
            #         tmp_score = out_score[i][min_fde_index[i].item()].unsqueeze(0) # sum of out_score[i] is 1
            #         C = torch.cat((C, tmp_score),0)

            #     # weight losses
            #     # print("min_fde_trj",min_fde_trj)
            #     # print("gt_trj",gt_trj)
            #     loss_sum = calculate_loss(min_fde_trj, gt_trj, C)
            #     print("loss_sum", loss_sum)
            #     assert 0

        
    
    result = format_results.results
    final_metrics = compute_forecasting_metrics(result['forecasted_trajectories'],
                                        result['gt_trajectories'],result['city_names'],
                                        6, 30, 2.0, result['forecasted_probabilities'])
    
    print(final_metrics)
    # print(evaluate(**format_results.results))
    print('Validation Process Finished!!')
