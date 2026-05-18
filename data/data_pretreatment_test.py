# 数据预处理
import sys
import os

# 获取当前脚本所在的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 将该目录加入系统路径
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import scipy.io as sio
import torch
import mylib as ml

# 以 Corel5k 为例
data_path_corel5k = r'D:\hzh\py_work\AIMNet\data\raw\corel5k\corel5k_six_view.mat'
data_path_espgame = r'D:\hzh\py_work\AIMNet\data\raw\espgame\espgame_six_view.mat'
data_path_iaprtc12 = r'D:\hzh\py_work\AIMNet\data\raw\iaprtc12\iaprtc12_six_view.mat'
data_path_mirflickr = r'D:\hzh\py_work\AIMNet\data\raw\mirflickr\mirflickr_six_view.mat'
data_path_pascal07 = r'D:\hzh\py_work\AIMNet\data\raw\pascal07\pascal07_six_view.mat'


# X_corel5k, Y_corel5k, n_sample_corel5k, views_corel5k, n_classes_corel5k = ml.load_and_preprocess(data_path_corel5k)
# X_espgame, Y_espgame, n_sample_espgame, views_espgame, n_classes_espgame = ml.load_and_preprocess(data_path_espgame)
# X_iaprtc12, Y_iaprtc12, n_sample_iaprtc12, views_iaprtc12, n_classes_iaprtc12 = ml.load_and_preprocess(data_path_iaprtc12)
# X_mirflickr, Y_mirflickr, n_sample_mirflickr, views_mirflickr, n_classes_mirflickr = ml.load_and_preprocess(data_path_mirflickr)
# X_pascal07, Y_pascal07, n_sample_pascal07, views_pascal07, n_classes_pascal07 = ml.load_and_preprocess(data_path_pascal07)


# corel5k_X_mask, corel5k_Y_mask = ml.create_masks(n_sample_corel5k, views_corel5k, n_classes_corel5k)
# espgame_X_mask, espgame_Y_mask = ml.create_masks(n_sample_espgame, views_espgame, n_classes_espgame)
# iaprtc12_X_mask, iaprtc12_Y_mask = ml.create_masks(n_sample_iaprtc12, views_iaprtc12, n_classes_iaprtc12)
# mirflickr_X_mask, mirflickr_Y_mask = ml.create_masks(n_sample_mirflickr, views_mirflickr, n_classes_mirflickr)
# pascal07_X_mask, pascal07_Y_mask = ml.create_masks(n_sample_pascal07, views_pascal07, n_classes_pascal07)



# processed_data_corel5k = {
#     'X': X_corel5k,
#     'Y': Y_corel5k,
#     'X_mask': corel5k_X_mask,
#     'Y_mask': corel5k_Y_mask,
#     'n_sample': n_sample_corel5k,
#     'views': views_corel5k,
#     'n_classes': n_classes_corel5k
# }

# processed_data_espgame = {
#     'X': X_espgame,
#     'Y': Y_espgame,
#     'X_mask': espgame_X_mask,
#     'Y_mask': espgame_Y_mask,
#     'n_sample': n_sample_espgame,
#     'views': views_espgame,
#     'n_classes': n_classes_espgame
# }

# processed_data_iaprtc12 = {
#     'X': X_iaprtc12,
#     'Y': Y_iaprtc12,
#     'X_mask': iaprtc12_X_mask,
#     'Y_mask': iaprtc12_Y_mask,
#     'n_sample': n_sample_iaprtc12,
#     'views': views_iaprtc12,
#     'n_classes': n_classes_iaprtc12
# }

# processed_data_mirflickr = {
#     'X': X_mirflickr,
#     'Y': Y_mirflickr,
#     'X_mask': mirflickr_X_mask,
#     'Y_mask': mirflickr_Y_mask,
#     'n_sample': n_sample_mirflickr,
#     'views': views_mirflickr,
#     'n_classes': n_classes_mirflickr
# }

# processed_data_pascal07 = {
#     'X': X_pascal07,
#     'Y': Y_pascal07,
#     'X_mask': pascal07_X_mask,
#     'Y_mask': pascal07_Y_mask,
#     'n_sample': n_sample_pascal07,
#     'views': views_pascal07,
#     'n_classes': n_classes_pascal07
# }

processed_data_corel5k = ml.data_processing(data_path_corel5k,train_ratio=0.7,missing_rate=0.5)
processed_data_espgame = ml.data_processing(data_path_espgame,train_ratio=0.7,missing_rate=0.5)
processed_data_iaprtc12 = ml.data_processing(data_path_iaprtc12,train_ratio=0.7,missing_rate=0.5)
processed_data_mirflickr = ml.data_processing(data_path_mirflickr,train_ratio=0.7,missing_rate=0.5)
processed_data_pascal07 = ml.data_processing(data_path_pascal07,train_ratio=0.7,missing_rate=0.5)

torch.save(processed_data_corel5k, r'D:\hzh\py_work\AIMNet\data\processed\train_and_test_corel5k_03test_rate_05missing_rate.pt')
torch.save(processed_data_espgame, r'D:\hzh\py_work\AIMNet\data\processed\train_and_test_espgame_03test_rate_05missing_rate.pt')
torch.save(processed_data_iaprtc12, r'D:\hzh\py_work\AIMNet\data\processed\train_and_test_iaprtc12_03test_rate_05missing_rate.pt')
torch.save(processed_data_mirflickr, r'D:\hzh\py_work\AIMNet\data\processed\train_and_test_mirflickr_03test_rate_05missing_rate.pt')
torch.save(processed_data_pascal07, r'D:\hzh\py_work\AIMNet\data\processed\train_and_test_pascal07_03test_rate_05missing_rate.pt')