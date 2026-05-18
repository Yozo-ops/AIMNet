import models  
import torch


# 模拟测试
device = torch.device('cuda' if torch.cuda.is_available() else print('cpu'))
model = models.AIMNet(view_dims=[100, 200, 150], n_classes=10).to(device)

# 伪造 32 个样本的输入
mock_x = [torch.randn(32, 100).to(device), torch.randn(32, 200).to(device), torch.randn(32, 150).to(device)]
mock_W = (torch.rand(32, 3) > 0.5).float().to(device) # 必须是 float 类型
mock_edge = torch.randint(0, 10, (2, 30)).to(device)

# 运行前向传播
pred = model(mock_x, mock_edge, mock_W)
print("正向传播成功！输出形状为:", pred.shape) # 期望输出: torch.Size([32, 10])