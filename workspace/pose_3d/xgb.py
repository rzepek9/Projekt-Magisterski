from pathlib import Path

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from xgboost import XGBClassifier

from workspace.pose_3d.dataset import KeypointsMatrix

def data_loading(training_set, test_set):
    x_train, y_train = [], []
    for idx in range(len(training_set)):
        x, y = training_set[idx]
        x_train.append(x.flatten())
        y_train.append(y.flatten())

    x_train, y_train = torch.stack(x_train), torch.stack(y_train)

    x_test, y_test = [], []
    for idx in range(len(test_set)):
        x, y = test_set[idx]
        x_test.append(x.flatten())
        y_test.append(y.flatten())

    x_test, y_test = torch.stack(x_test), torch.stack(y_test)
    
    return x_train, y_train, x_test, y_test

# define the model
def model_train(x_train, y_train, x_test, y_test, n_estimators, eta, max_depth, subsample, colsample_bytree):
    
    model = XGBClassifier(n_estimators=n_estimators,
                          eta=eta,
                          max_depth=max_depth,
                          subsample = subsample,
                          colsample_bytree=colsample_bytree,
                          gamma = 1,
                          early_stopping_rounds=10,
                          random_state=1,
                          three_method='hist',
                          device = 'cuda:3'
                         )

    # define the datasets to evaluate each iteration
    evalset = [(x_train, y_train), (x_test,y_test)]

    model.fit(x_train, y_train, eval_metric='logloss', eval_set=evalset)

    # evaluate performance
    yhat = model.predict(x_test)
    score = accuracy_score(y_test, yhat)
    print('Accuracy: %.3f' % score)
    results = model.evals_result()
    return score, results

def parameters_searching(name, eta, camera=None, cordinates=False, height=False, min_max=False, cg=False, vf=False, select=False):
    training_set = KeypointsMatrix('/home/s175668/raid/Praca-Magisterska/dataset/folds/clean/003_train.csv',
                                   camera=camera, cordinates_move=cordinates, height_normalize=height, signal_normalize_0_1=min_max,
                                   cg=cg, vf=vf, choose_points=select)
    
    test_set = KeypointsMatrix('/home/s175668/raid/Praca-Magisterska/dataset/folds/clean/003_val.csv',
                                camera=camera, cordinates_move=cordinates, height_normalize=height, signal_normalize_0_1=min_max,
                                cg=cg, vf=vf, choose_points=select)
    
    x_train, y_train, x_test, y_test = data_loading(training_set, test_set)
    
    eta = eta
    n_estimators = 4000
    best_score = 0
    best_result = None
    best_hyperparameters = []
    
    for depth in [3, 4, 7, 8, 9, 10, 11, 12, 14, 15]:
        for subsample in [0.3, 0.5, 0.7, 0.9]:
            for tree in [0.3, 0.5, 0.7, 0.9]:
                score, results = model_train(x_train, y_train, x_test, y_test, n_estimators=n_estimators, eta=eta, max_depth=depth, subsample=subsample, colsample_bytree=tree)
                if score > best_score:
                    best_result = results
                    best_score = score
                    best_hyperparameters.append([name, score*100, n_estimators, eta, depth, subsample, tree])
    
    df_001_side_cords_move_height_001 = pd.DataFrame(best_hyperparameters, columns=['run', 'score', 'runs', 'eta', 'max_depth', 'subsample', 'tree'])
    df_001_side_cords_move_height_001.to_csv(str(out_root), index=False, mode='a', header=False)
    

if __name__ == '__main__':
    out_root = Path('/home/s175668/raid/Praca-Magisterska/experiments/clean/xgboost_parameters_003.csv')
    
    if not out_root.exists():
        out_root.parent.mkdir(parents=True, exist_ok=True)
        df_empty = pd.DataFrame(columns=['run', 'score', 'runs', 'eta', 'max_depth', 'subsample', 'tree'])
        df_empty.to_csv(out_root, index=False)
    
    parameters_searching('003_cg_side_all_kpts', 0.01, camera='side', cg=True)
    
    parameters_searching('003_cg_all_kpts', 0.01, camera=None, cg=True)
    
    parameters_searching('003_cg_side_selected_kpts', 0.01, camera='side', cg=True, select=True)
    
    parameters_searching('003_cg_selected_kpts', 0.01, cg=True, select=True)
    
    parameters_searching('003_cg_selected_kpts_min_max', 0.01, cg=True, select=True, min_max=True)
    
    parameters_searching('003_cg_side_selected_kpts_min_max', 0.01, camera='side', cg=True, select=True, min_max=True)
    
    parameters_searching('003_vf_side_all_kpts', 0.01, camera='side', vf=True)
    
    parameters_searching('003_vf_all_kpts', 0.01, camera=None, vf=True)

    parameters_searching('003_vf_selected_kpts', 0.01, vf=True, select=True)
    
    parameters_searching('003_vf_selected_kpts', 0.01, camera='side', vf=True, select=True)
    
    parameters_searching('003_vf_selected_kpts_min_max', 0.01, vf=True, select=True, min_max=True)

    parameters_searching('003_vf_side_selected_kpts_min_max', 0.01, camera='side', vf=True, select=True, min_max=True)
    
    parameters_searching('003_vf_cg_side_selected_kpts_min_max', 0.01, camera='side', vf=True, cg=True, select=True, min_max=True)
    
    parameters_searching('003_vf_cg_selected_kpts_min_max', 0.01, vf=True, cg=True, select=True, min_max=True)
    
    parameters_searching('003_vf_cg_selected_kpts', 0.01, vf=True, cg=True, select=True)

    parameters_searching('003_vf_cg_side_selected_kpts', 0.01, camera='side', vf=True, cg=True, select=True)
    
    parameters_searching('003_vf_cg_side', 0.01, camera='side', vf=True, cg=True)
    
    parameters_searching('003_vf_cg', 0.01, vf=True, cg=True)