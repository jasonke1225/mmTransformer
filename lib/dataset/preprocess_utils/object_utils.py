import os
from typing import Any, Dict, List

import numpy as np
import pandas as pd

def pad_track(
        track_df: pd.DataFrame,
        seq_timestamps: np.ndarray,
        base: int,
        track_len: int,
        raw_data_format: Dict[str, int],
) -> np.ndarray:
    """Pad incomplete tracks.
    Args:
        track_df (Dataframe): Dataframe of the track id's car (by lin)
        seq_timestamps (numpy array): All timestamps in the sequence
        base: base frame id (0 for observed trajectory, 20 for future trajectory)
        track_len (int): Length of whole trajectory (observed + future)
        raw_data_format (Dict): Format of the sequence
    Returns:
        padded_track_array (numpy array)
    """
    track_vals = track_df.values
    track_timestamps = track_df["TIMESTAMP"].values
    seq_timestamps = seq_timestamps[base:base+track_len]

    # start and index of the track in the sequence
    # 找每個track_id在全部時間(50個frame)中的start index和end index (by lin)
    start_idx = np.where(seq_timestamps == track_timestamps[0])[0][0]
    end_idx = np.where(seq_timestamps == track_timestamps[-1])[0][0]

    # Edge padding in front and rear, i.e., repeat the first and last coordinates
    # 只填充上下row， start_idx = 1, end_idx = 2, A = np.pad(a,((start_idx, end_idx), (0, 0)), "edge") (by lin)
    # ex: a = [[0,1,2],                  A = [[0,1,2],
    #          [3,4,5],          =>           [0,1,2],
    #          [6,7,8]]                       [3,4,5],
    #                                         [6,7,8],
    #                                         [6,7,8],
    #                                         [6,7,8]]
    # if self.PADDING_TYPE == "REPEAT"
    padded_track_array = np.pad(track_vals,
                                ((start_idx, track_len - end_idx - 1),
                                    (0, 0)), "edge")

    mask = np.ones((end_idx+1-start_idx))
    # let start_idx = 1, end_idx = 2, track_len = 20 (by lin)
    # final mask = [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] (by lin)
    mask = np.pad(mask, (start_idx, track_len - end_idx - 1), 'constant')
    if padded_track_array.shape[0] < track_len:
        # rare case, just ignore
        return None, None, False

    # Overwrite the timestamps in padded part
    # padded_track_array可視為強行將所有track_id的資料都pad成有20幀和30幀的資料 (by lin) 
    for i in range(padded_track_array.shape[0]):
        padded_track_array[i, 0] = seq_timestamps[i]
    assert mask.shape[0] == padded_track_array.shape[0]
    return padded_track_array, mask, True


# Important: We delete some invalid nearby agents according to our criterion!!! Refer to continue option
def get_nearby_moving_obj_feature_ls(agent_df, traj_df, obs_len, seq_ts, obj_radius, norm_center, raw_dataformat, fut_len=30):
    """
    args: obs_len = 20 (by lin)
          row_data_format {'TIMESTAMP': 0, 'TRACK_ID': 1, 'OBJECT_TYPE': 2, 'X': 3, 'Y': 4, 'CITY_NAME': 5} (by lin)
          seq_ts = np.unique(traj_df['TIMESTAMP'].values) (by lin)
          
    returns: obj_feature_ls, a list of list, (track, timestamp, mask, track_id, gt_track, gt_mask)
    
    # obj_feature_ls 為一個csv中的資料，其track_id符合以下規定: (by lin)
    # 1. 不是'AGENT'
    # 2. 在'AGENT'第20幀前有出現過
    # 3. 在'AGENT'第20幀有出現且與其距離在obj_radius(10)內
    # 4. mask的總值大於3 (出現的幀數小於20幀才會去pad，pad的幀數對應的mask位置為0)
    """
    obj_feature_ls = []
    query_x, query_y = agent_df[['X', 'Y']].values[obs_len-1]
    p0 = np.array([query_x, query_y])
    for track_id, remain_df in traj_df.groupby('TRACK_ID'):
        if remain_df['OBJECT_TYPE'].iloc[0] == 'AGENT':
            continue
        
        # hist_df 為用track_id分類好，且不是'AGENT'，並為'AGENT'20幀前有出現過的車子所組成的dataframe (by lin)
        hist_df = remain_df[remain_df['TIMESTAMP'] <=
                            agent_df['TIMESTAMP'].values[obs_len-1]]
        if(len(hist_df) == 0):
            continue
        # pad hist
        xys, ts = None, None
        # 如果目前track_id在'AGENT'第20幀前出現的幀數少於20，將資料pad成20幀的資料 (by lin)
        if len(hist_df) < obs_len:
            paded_nd, mask, flag = pad_track(
                hist_df, seq_ts, 0, obs_len, raw_dataformat)
            if flag == False:
                continue
            xys = np.array(paded_nd[:, 3:5], dtype=np.float64)
            ts = np.array(paded_nd[:, 0], dtype=np.float64)
        else:
            xys = hist_df[['X', 'Y']].values
            ts = hist_df["TIMESTAMP"].values
            mask = np.ones((obs_len))

        p1 = xys[-1]
        if mask[-1] == 0 or np.linalg.norm(p0 - p1) > obj_radius:
            continue
        if(sum(mask) <= 3):
            continue

        fut_df = remain_df[remain_df['TIMESTAMP'] >
                           agent_df['TIMESTAMP'].values[obs_len-1]]
        # pad future
        gt_xys = None
        if len(fut_df) == 0:
            gt_xys = np.zeros((fut_len, 2))+norm_center
            gt_mask = np.zeros((fut_len))
        elif len(fut_df) < fut_len:
            paded_nd, gt_mask, flag = pad_track(
                fut_df, seq_ts, obs_len, fut_len, raw_dataformat)
            if flag == False:
                continue
            gt_xys = np.array(paded_nd[:, 3:5], dtype=np.float64)
        else:
            gt_xys = fut_df[['X', 'Y']].values
            gt_mask = np.ones((fut_len))

        xys -= norm_center  # normalize to last observed timestamp point of agent
        gt_xys -= norm_center

        assert xys.shape[0] == obs_len
        assert mask.shape[0] == obs_len
        assert gt_xys.shape[0] == fut_len
        assert gt_mask.shape[0] == fut_len
        obj_feature_ls.append(
            [xys, ts, mask, track_id, gt_xys, gt_mask])
    
    return obj_feature_ls
