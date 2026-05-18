import scipy.io as sio
import torch
import mylib as ml

data_path_corel5k_pt = 'data/processed/corel5k_double_missing.pt'
data_path_espgame_pt = 'data/processed/espgame_double_missing.pt'
data_path_iaprtc12_pt = 'data/processed/iaprtc12_double_missing.pt'
data_path_mirflickr_pt = 'data/processed/mirflickr_double_missing.pt'
data_path_pascal07_pt = 'data/processed/pascal07_double_missing.pt'

train_corel5k, test_corel5k = ml.split_and_finalize_data(data_path_corel5k_pt)
train_espgame, test_espgame = ml.split_and_finalize_data(data_path_espgame_pt)
train_iaprtc12, test_iaprtc12 = ml.split_and_finalize_data(data_path_iaprtc12_pt)
train_mirflickr, test_mirflickr = ml.split_and_finalize_data(data_path_mirflickr_pt)
train_pascal07, test_pascal07 = ml.split_and_finalize_data(data_path_pascal07_pt)

train_and_test_corel5k_03test_rate_05missing_rate = {
    'train': train_corel5k,
    'test': test_corel5k
}

train_and_test_espgame_03test_rate_05missing_rate = {
    'train': train_espgame,
    'test': test_espgame
}

train_and_test_iaprtc12_03test_rate_05missing_rate = {
    'train': train_iaprtc12,
    'test': test_iaprtc12
}

train_and_test_mirflickr_03test_rate_05missing_rate = {
    'train': train_mirflickr,
    'test': test_mirflickr
}

train_and_test_pascal07_03test_rate_05missing_rate = {
    'train': train_pascal07,
    'test': test_pascal07
}

torch.save(train_and_test_corel5k_03test_rate_05missing_rate, 'data/processed/train_and_test_corel5k_03test_rate_05missing_rate.pt')
torch.save(train_and_test_espgame_03test_rate_05missing_rate, 'data/processed/train_and_test_espgame_03test_rate_05missing_rate.pt')
torch.save(train_and_test_iaprtc12_03test_rate_05missing_rate, 'data/processed/train_and_test_iaprtc12_03test_rate_05missing_rate.pt')
torch.save(train_and_test_mirflickr_03test_rate_05missing_rate, 'data/processed/train_and_test_mirflickr_03test_rate_05missing_rate.pt')
torch.save(train_and_test_pascal07_03test_rate_05missing_rate, 'data/processed/train_and_test_pascal07_03test_rate_05missing_rate.pt')

