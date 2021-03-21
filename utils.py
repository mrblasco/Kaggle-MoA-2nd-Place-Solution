import pandas as pd
import numpy as np

import json
import logging
import os
import shutil
import torch
import random

def seed_everything(seed = 42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def select_ns_targets(train_features, train_targets_scored, train_targets_nonscored, q_n_cut = 0.9): 
  """
  Select non-scored targets based on correlation with scored targets' features.
  """
  nonctr_id = train_features.loc[train_features['cp_type']!= 'ctl_vehicle', 'sig_id'].tolist()
  tmp_con1 = [i in nonctr_id for i in train_targets_scored['sig_id']]
  mat_cor = pd.DataFrame(np.corrcoef(train_targets_scored.drop('sig_id', axis = 1)[tmp_con1].T
                        , train_targets_nonscored.drop('sig_id', axis = 1)[tmp_con1].T))
  mat_cor2 = mat_cor.iloc[(train_targets_scored.shape[1] - 1):, 0:train_targets_scored.shape[1]-1]

  target_nonsc_cols = train_targets_nonscored.drop('sig_id', axis = 1).columns.values.tolist()
  target_cols = train_targets_scored.drop('sig_id', axis = 1).columns.values.tolist()

  mat_cor2.index = target_nonsc_cols
  mat_cor2.columns = target_cols
  mat_cor2 = mat_cor2.dropna()
  mat_cor2_max = mat_cor2.abs().max(axis = 1)
  out = mat_cor2_max[mat_cor2_max > np.quantile(mat_cor2_max, q_n_cut)].index.tolist()
  return out

def qnorm(train_f, test_f, feat_dic):
  """
  Quantile normalization 
  """
  import numpy as np
  # train = gene
  q2 = train_f[feat_dic['gene']].apply(np.quantile, axis = 1, q = 0.25).copy()
  q7 = train_f[feat_dic['gene']].apply(np.quantile, axis = 1, q = 0.75).copy()
  qmean = (q2+q7)/2
  train_f[feat_dic['gene']] = (train_f[feat_dic['gene']].T - qmean.values).T
  
  # test = gene
  q2 = test_f[feat_dic['gene']].apply(np.quantile, axis = 1, q = 0.25).copy()
  q7 = test_f[feat_dic['gene']].apply(np.quantile, axis = 1, q = 0.75).copy()
  qmean = (q2+q7)/2
  test_f[feat_dic['gene']] = (test_f[feat_dic['gene']].T - qmean.values).T

  # train = cell 
  q2 = train_f[feat_dic['cell']].apply(np.quantile, axis = 1, q = 0.25).copy()
  q7 = train_f[feat_dic['cell']].apply(np.quantile, axis = 1, q = 0.72).copy()
  qmean = (q2+q7)/2
  train_f[feat_dic['cell']] = (train_f[feat_dic['cell']].T - qmean.values).T
  qmean2 = train_f[feat_dic['cell']].abs().apply(np.quantile, axis = 1, q = 0.75).copy()+4
  train_f[feat_dic['cell']] = (train_f[feat_dic['cell']].T / qmean2.values).T.copy()

  # test = cell 
  q2 = test_f[feat_dic['cell']].apply(np.quantile, axis = 1, q = 0.25).copy()
  q7 = test_f[feat_dic['cell']].apply(np.quantile, axis = 1, q = 0.72).copy()
  qmean = (q2+q7)/2
  test_f[feat_dic['cell']] = (test_f[feat_dic['cell']].T - qmean.values).T
  qmean2 = test_f[feat_dic['cell']].abs().apply(np.quantile, axis = 1, q = 0.75).copy()+4
  test_f[feat_dic['cell']] = (test_f[feat_dic['cell']].T / qmean2.values).T.copy()
  return train_f, test_f

def norm_fit(df_1, saveM = True, sc_name = 'zsco'):   
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, Normalizer, QuantileTransformer, PowerTransformer
    ss_1_dic = {'zsco':StandardScaler(), 
                'mima':MinMaxScaler(), 
                'maxb':MaxAbsScaler(), 
                'robu':RobustScaler(), 
                'norm':Normalizer(), 
                'quan':QuantileTransformer(n_quantiles = 100, random_state = 0, output_distribution = "normal"), 
                'powe':PowerTransformer()}
    ss_1 = ss_1_dic[sc_name]
    df_2 = pd.DataFrame(ss_1.fit_transform(df_1), index = df_1.index, columns = df_1.columns)
    if saveM == False:
        return(df_2)
    else:
        return(df_2, ss_1)

def norm_tra(df_1, ss_x):
    df_2 = pd.DataFrame(ss_x.transform(df_1), index = df_1.index, columns = df_1.columns)
    return(df_2)

def g_table(list1):
    table_dic = {}
    for i in list1:
        if i not in table_dic.keys():
            table_dic[i] = 1
        else:
            table_dic[i] += 1
    return(table_dic)


def pca_pre(tr, va, te, n_comp, feat_raw, feat_new):
    from sklearn.decomposition import PCA
    pca = PCA(n_components = n_comp, random_state = 42)
    tr2 = pd.DataFrame(pca.fit_transform(tr[feat_raw]), columns = feat_new)
    va2 = pd.DataFrame(pca.transform(va[feat_raw]), columns = feat_new)
    te2 = pd.DataFrame(pca.transform(te[feat_raw]), columns = feat_new)
    return(tr2, va2, te2)


# function import from cs230-github repo 
class Params():
    """Class that loads hyperparameters from a json file.

    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__

def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)

