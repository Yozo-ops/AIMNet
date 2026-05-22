import models  
import torch
from sklearn.metrics import average_precision_score, hamming_loss, roc_auc_score


# # 模拟测试
# device = torch.device('cuda' if torch.cuda.is_available() else print('cpu'))
# model = models.AIMNet(view_dims=[100, 200, 150], n_classes=10).to(device)

# # 伪造 32 个样本的输入
# mock_x = [torch.randn(32, 100).to(device), torch.randn(32, 200).to(device), torch.randn(32, 150).to(device)]
# mock_W = (torch.rand(32, 3) > 0.5).float().to(device) # 必须是 float 类型
# mock_edge = torch.randint(0, 10, (2, 30)).to(device)

# # 运行前向传播
# pred = model(mock_x, mock_edge, mock_W)
# print("正向传播成功！输出形状为:", pred.shape) # 期望输出: torch.Size([32, 10])

if __name__ == "__main__":

    data = torch.load("data/processed/train_and_test_corel5k_03test_rate_05missing_rate.pt")
    train_data = data['train']
    val_data = data['test']
    
    average_precision = average_precision_score(train_data['Y'][train_data['G'] == 1], train_data['Y'][train_data['G'] == 1])
    AUC = roc_auc_score(train_data['Y'], train_data['Y'])
    print("训练集平均准确率:", average_precision)
    print("训练集AUC:", AUC)

    # def features_num_extract(data):
    #     view_dims = []
    #     for i in range(len(data['X'])):
    #         view_dims.append(data['X'][i].shape[1])
    #     return view_dims
    
    # print(features_num_extract(train_data))
    # print(train_data['X'][0].shape)
