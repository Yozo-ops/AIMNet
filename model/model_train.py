import torch
import torch.optim as optim
import numpy as np
from sklearn.metrics import average_precision_score, hamming_loss, roc_auc_score
import matplotlib.pyplot as plt
import models
import os 


def compute_metrics(y_true, y_pred, G):
    """
    根据论文要求计算评估指标 (只在已知/有效的测试标签上评估，或在全量完整测试集上评估)
    y_true: [n_samples, n_classes]
    y_pred: [n_samples, n_classes] (模型的概率输出)
    G: [n_samples, n_classes] (已知标签掩码，如果是全量完整测试集则全为1)
    """
    # 转为 numpy 方便计算
    y_true_np = y_true.cpu().detach().numpy()
    y_pred_np = y_pred.cpu().detach().numpy()
    G_np = G.cpu().detach().numpy()
    
    # 论文指标 1: 1 - Hamming Loss
    # 注意：标准多标签需将预测概率二值化 (以0.5为界)
    y_pred_bin = (y_pred_np >= 0.5).astype(int)
    # 只计算已知标签处的 Hamming Loss
    actual_elements = np.sum(G_np)
    if actual_elements == 0:
        hl = 0.0
    else:
        hl = np.sum((y_true_np != y_pred_bin) * G_np) / actual_elements
    one_minus_hl = 1.0 - hl
    
    # 论文指标 2: Average Precision (AP)
    # 分别计算每个样本或每个类别的 AP，这里采用宏观/微观或过滤未知后的标准 sklearn AP
    # 为了简化且不失准确性，我们在计算有效位置的总体 AP
    try:
        # 过滤掉完全没有正样本或全为负样本的无效位置（sklearn 限制）
        ap = average_precision_score(y_true_np[G_np == 1], y_pred_np[G_np == 1])
    except ValueError:
        ap = 0.0
        
    # 论文指标 3: AUC
    try:
        auc = roc_auc_score(y_true_np[G_np == 1], y_pred_np[G_np == 1])
    except ValueError:
        auc = 0.5

    return {
        "1-HL": one_minus_hl,
        "AP": ap,
        "AUC": auc
    }

def plot_training_history(history, save_path="training_curves.png"):
    """
    根据记录的训练历史绘制并保存图像
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(14, 5))
    
    # ---- 第一张子图：Loss 曲线 ----
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss', color='blue', linewidth=2)
    plt.plot(epochs, history['val_loss'], label='Validation Loss', color='red', linestyle='--', linewidth=2)
    plt.title('Training and Validation Loss', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # ---- 第二张子图：性能指标曲线 (AP & AUC) ----
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_ap'], label='Train AP', color='green', linewidth=1.5)
    plt.plot(epochs, history['val_ap'], label='Validation AP', color='orange', linewidth=2)
    plt.plot(epochs, history['val_auc'], label='Validation AUC', color='purple', linestyle='-.', linewidth=2)
    plt.title('Performance Metrics (AP & AUC)', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.ylim(0, 1.05) # 指标通常在 0-1 之间
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight') # 保存为高清晰度图片
    plt.close()
    print(f"--> 训练性能曲线已保存至: {save_path}")

def train_aimnet(model, train_data, val_data, epochs=200, lr=0.001, weight_decay=1e-5, device="cuda"):
    """
    AIMNet 的完整训练函数
    train_data / val_data 应该是一个字典，包含:
        - 'x_list': 列表 [X_1, X_2, ... X_m]
        - 'edge_index': 标签图的边索引
        - 'W': 视图缺失矩阵
        - 'Y': 标签矩阵
        - 'G': 标签缺失矩阵
    """
    model = model.to(device)
    
    # 论文提到整个框架仅含一个损失函数，联合更新所有参数 weight_decay=weight_decay权重衰减策略，就是在损失函数后添加L2 正则项
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # 学习率衰减策略（非必需，但有助于深层网络稳定收敛）
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
    
    best_ap = 0.0

    history = {
        'train_loss': [], 'val_loss': [],
        'train_ap': [], 'val_ap': [],
        'train_auc': [], 'val_auc': []
    }

    print("开始训练 AIMNet 模型...")
    print("-" * 50)
    
    # 将训练数据和验证数据搬运到对应设备 (CPU/GPU)
    def to_device(data_dict):
        result = {
            'x_list': [x.to(device) for x in data_dict['X']],
            'W': data_dict['W'].to(device),
            'Y': data_dict['Y'].to(device),
            'G': data_dict['G'].to(device),
            'edge_index': data_dict['C'].to(device)
        }
        return result
        
    train_inputs = to_device(train_data)
    val_inputs = to_device(val_data)
    
    for epoch in range(1, epochs + 1):
        # ================= 训练阶段 =================
        model.train()
        optimizer.zero_grad()
        
        # 前向传播
        fused_P = model(train_inputs['x_list'], train_inputs['edge_index'], train_inputs['W'])
        
        # 计算带遮罩的损失 (公式 12)
        loss = model.compute_loss(fused_P, train_inputs['Y'], train_inputs['G'])
        
        # 反向传播与参数更新
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) #梯度裁剪
        optimizer.step()
        
        # ================= 验证阶段 =================
        model.eval()
        with torch.no_grad():
            val_P = model(val_inputs['x_list'], val_inputs['edge_index'], val_inputs['W'])
            val_loss = model.compute_loss(val_P, val_inputs['Y'], val_inputs['G'])
            
            # 计算当前 Epoch 的性能指标
            train_metrics = compute_metrics(train_inputs['Y'], fused_P, train_inputs['G'])
            val_metrics = compute_metrics(val_inputs['Y'], val_P, val_inputs['G'])
            
        # 调整学习率
        # scheduler.step(val_metrics["AP"])
        
        history['train_loss'].append(loss.item())
        history['val_loss'].append(val_loss.item())
        history['train_ap'].append(train_metrics['AP'])
        history['val_ap'].append(val_metrics['AP'])
        history['train_auc'].append(train_metrics['AUC'])
        history['val_auc'].append(val_metrics['AUC'])

        # 打印日志
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch [{epoch:03d}/{epochs}] | Train Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f}")
            print(f"  [Train] AP: {train_metrics['AP']:.4f}, 1-HL: {train_metrics['1-HL']:.4f}, AUC: {train_metrics['AUC']:.4f}")
            print(f"  [Val]   AP: {val_metrics['AP']:.4f}, 1-HL: {val_metrics['1-HL']:.4f}, AUC: {val_metrics['AUC']:.4f}")
            print("-" * 50)
            
        # 保存最佳模型权重
        if val_metrics["AP"] > best_ap:
            best_ap = val_metrics["AP"]
            torch.save(model.state_dict(), "best_aimnet_model.pt")
            print(f"--> 检测到更好的验证集 AP: {best_ap:.4f}, 模型权重已保存。")
            print("-" * 50)

        if epoch % 20 == 0 or epoch == epochs:
            plot_training_history(history, save_path="training_curves.png")
            
    print("训练完成！最佳验证集 AP:", best_ap)
    return model

def features_num_extract(data):
    view_dims = []
    for i in range(len(data['X'])):
        view_dims.append(data['X'][i].shape[1])
    return view_dims

def class_num_extract(data):
    return data['Y'].shape[1]



if __name__ == "__main__":

    data = torch.load("data/processed/train_and_test_corel5k_03test_rate_05missing_rate.pt")
    train_data = data['train']
    val_data = data['test']

    view_dims = features_num_extract(train_data)
    n_classes = class_num_extract(train_data)

    # 2. 实例化模型
    model = models.AIMNet(view_dims=view_dims, n_classes=n_classes, d_e=128, tau=1.0)
    
    # 3. 运行训练 (如果没有 GPU，可以将 device 改为 "cpu")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trained_model = train_aimnet(model, train_data, val_data, epochs=100, lr=0.00001, device=device)