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


def parse_args():

    parser = argparse.ArgumentParser(description='Evaluate the mmTransformer')
    parser.add_argument('config', help='config file path')
    parser.add_argument('--model-name', type=str, default='demo')
    parser.add_argument('--model-save-path', type=str, default='./models/')

    args = parser.parse_args()
    return args

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
    model_cfg = cfg.get('model')
    stacked_transfomre = load_model_class(model_cfg['type'])
    model = mmTrans(stacked_transfomre, model_cfg).cuda()
    # model_name = os.path.join(args.model_save_path,
    #                           '{}.pt'.format(args.model_name))

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

        model_name =  dir_path + "/model_"+str(num)+".pt"

    model = load_checkpoint(model_name, model)
    print('Successfully Loaded model: {}'.format(model_name))
    print('Finished Initialization in {:.3f}s!!!'.format(
        time.time()-start_time))
    # ==================================== EVALUATION LOOP =====================================================
    model.eval()
    progress_bar = tqdm(val_dataloader)
    print(len(val_dataloader))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        for j, data in enumerate(progress_bar):
            for key in data.keys():
                if isinstance(data[key], torch.Tensor):
                    data[key] = data[key].to(device)

            out = model(data)
            format_results(data, out)

            # gt = data['FUTURE']
            # gt_trj = torch.tensor([])
            # for i in range(len(gt)):
            #     tmp_trj = gt[i][0,:,:2].unsqueeze(0)
            #     gt_trj = torch.cat((gt_trj, tmp_trj), 0)

            # gt_end_point = gt_trj[:,gt_trj.shape[1]-1,:].to(device)
            # gt_trj = gt_trj.to(device)

            # pred_trj = out[0][:,0].permute(1,0,2,3).to(device)
            # pred_end_point = pred_trj[:,:,pred_trj.shape[2]-1,:].to(device)
            
            # kl_div_loss = torch.nn.functional.kl_div(pred_end_point.softmax(dim=-1).log(), gt_end_point.softmax(dim=-1)).to(device)
            # # print(kl_div_loss)

            # crit = torch.nn.HuberLoss().to(device)
            # loss_score = crit(pred_trj, gt_trj)
            # print(loss_score)
            # assert 0
            
    
    print(evaluate(**format_results.results))
    print('Validation Process Finished!!')
