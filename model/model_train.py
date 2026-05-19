import torch
import torch.optim as optim
import numpy as np
from sklearn.metrics import average_precision_score, hamming_loss, roc_auc_score

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
    
    # 论文提到整个框架仅含一个损失函数，联合更新所有参数
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # 学习率衰减策略（非必需，但有助于深层网络稳定收敛）
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, verbose=True)
    
    best_ap = 0.0
    print("开始训练 AIMNet 模型...")
    print("-" * 50)
    
    # 将训练数据和验证数据搬运到对应设备 (CPU/GPU)
    def to_device(data_dict):
        result = {
            'x_list': [x.to(device) for x in data_dict['X']],
            'W': data_dict['W'].to(device),
            'Y': data_dict['Y'].to(device),
            'G': data_dict['G'].to(device)
        }
        if 'C' in data_dict:
            result['edge_index'] = data_dict['C'].to(device)
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
        scheduler.step(val_metrics["AP"])
        
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
            
    print("训练完成！最佳验证集 AP:", best_ap)
    return model