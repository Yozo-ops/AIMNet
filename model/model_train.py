import torch
import torch.optim as optim
import numpy as np
from sklearn.metrics import average_precision_score, hamming_loss, roc_auc_score
import matplotlib.pyplot as plt
import models
import os 
import pandas as pd
import random


def set_all_seeds(seed=42):
    """
    固定所有的随机种子，确保实验的完全可复现性
    """
    # 1. 固定 Python 内置随机种子
    random.seed(seed)
    
    # 2. 固定 Numpy 随机种子
    np.random.seed(seed)
    
    # 3. 固定 PyTorch 随机种子 (CPU)
    torch.manual_seed(seed)
    
    # 4. 固定 PyTorch 随机种子 (GPU)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        # torch.cuda.manual_seed_all(seed)
    # 5. 强制底层 cuDNN 使用确定性算法 (极其关键)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def compute_metrics(y_true, y_pred, G):
    y_true_np = y_true.cpu().detach().numpy()
    y_pred_np = y_pred.cpu().detach().numpy()
    G_np = G.cpu().detach().numpy()
    
    num_samples, n_classes = y_true_np.shape
    
    # 1. 1-Hamming Loss
    y_pred_bin = (y_pred_np >= 0.5).astype(int)
    actual_elements = np.sum(G_np)
    hl = np.sum((y_true_np != y_pred_bin) * G_np) / actual_elements if actual_elements > 0 else 0.0
    one_minus_hl = 1.0 - hl
    
    # 为了防止模型将未知的标签（G=0）排在前面，我们将未知预测的分数设为极小值
    masked_y_pred = np.where(G_np == 1, y_pred_np, -np.inf)
    
    # 2. 1-One Error (1-OE)
    # One Error 评估的是：模型给出的最高分预测标签，是否是一个错误的标签。
    top_idx = np.argmax(masked_y_pred, axis=1)
    oe = np.mean(y_true_np[np.arange(num_samples), top_idx] == 0)
    one_minus_oe = 1.0 - oe
    
    # 3. 1-Coverage (1-COV)
    # 排序：分数越高的标签 rank 值越小 (排第1名则 rank=1)
    sorted_indices = np.argsort(-masked_y_pred, axis=1)
    ranks = np.empty_like(sorted_indices)
    np.put_along_axis(ranks, sorted_indices, np.arange(1, n_classes + 1), axis=1)
    
    true_pos_mask = (y_true_np == 1) & (G_np == 1)
    valid_samples = np.sum(true_pos_mask, axis=1) > 0 # 只计算至少有一个正标签的样本
    
    # 找出每个样本中，排得最后的那一个真实正标签的 rank
    cov_per_sample = np.max(np.where(true_pos_mask, ranks, 0), axis=1) - 1
    cov = np.mean(cov_per_sample[valid_samples]) / n_classes if np.sum(valid_samples) > 0 else 0.0
    one_minus_cov = 1.0 - cov
    
    # 4. Sample-wise AP & Ranking Loss (RL)
    # 数学上，对于单个样本，(1 - AUC) 完全等价于 Ranking Loss (错误排序对的比例)
    ap_list = []
    rl_list = []
    
    for i in range(num_samples):
        valid = G_np[i] == 1
        y_t = y_true_np[i, valid]
        y_p = y_pred_np[i, valid]
        
        if np.sum(y_t) > 0:  
            ap_list.append(average_precision_score(y_t, y_p))
            
        if 0 < np.sum(y_t) < len(y_t): # 只有同时存在正类和负类才能算排序
            rl_list.append(1.0 - roc_auc_score(y_t, y_p))
            
    ap = np.mean(ap_list) if len(ap_list) > 0 else 0.0
    rl = np.mean(rl_list) if len(rl_list) > 0 else 0.0
    one_minus_rl = 1.0 - rl
    
    # 5. Macro AUC
    auc_list = []
    for i in range(n_classes):
        valid_idx = (G_np[:, i] == 1)
        y_t = y_true_np[valid_idx, i]
        y_p = y_pred_np[valid_idx, i]
        if len(np.unique(y_t)) == 2:
            auc_list.append(roc_auc_score(y_t, y_p))
    auc = np.mean(auc_list) if len(auc_list) > 0 else 0.5

    return {
        "1-HL": one_minus_hl,
        "AP": ap,
        "AUC": auc,
        "1-OE": one_minus_oe,
        "1-COV": one_minus_cov,
        "1-RL": one_minus_rl
    }

def plot_training_history(history, save_path="training_curves.png"):
    """
    根据记录的训练历史绘制并保存图像
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(18, 5))
    
    # ---- 第一张子图：Loss 曲线 ----
    plt.subplot(1, 3, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss', color='blue', linewidth=2)
    plt.plot(epochs, history['val_loss'], label='Validation Loss', color='red', linestyle='--', linewidth=2)
    plt.title('Training and Validation Loss', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # ---- 第二张子图：核心排序指标 (AP & AUC) ----
    plt.subplot(1, 3, 2)
    plt.plot(epochs, history['val_ap'], label='Val AP', color='orange', linewidth=2)
    plt.plot(epochs, history['val_auc'], label='Val AUC', color='purple', linestyle='-.', linewidth=2)
    plt.title('Core Metrics (Val AP & AUC)', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.ylim(0, 1.05) 
    plt.legend()
    plt.grid(True, alpha=0.3)

    # ---- 第三张子图：多标签辅助指标 ----
    plt.subplot(1, 3, 3)
    plt.plot(epochs, history['val_1_hl'], label='Val 1-HL', color='green')
    plt.plot(epochs, history['val_1_rl'], label='Val 1-RL', color='brown')
    plt.plot(epochs, history['val_1_oe'], label='Val 1-OE', color='pink')
    plt.plot(epochs, history['val_1_cov'], label='Val 1-COV', color='gray')
    plt.title('Other Multilabel Metrics (Val)', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.ylim(0, 1.05) 
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight') 
    plt.close()

def train_aimnet(model, train_data, val_data, epochs=200, lr=0.001, weight_decay=1e-5, batch_size=512, device="cuda"):
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    
    best_ap = 0.0
    history = {
        'train_loss': [], 'val_loss': [],
        'train_ap': [], 'val_ap': [],
        'train_auc': [], 'val_auc': [],
        'train_1_hl': [], 'val_1_hl': [],
        'train_1_rl': [], 'val_1_rl': [],
        'train_1_oe': [], 'val_1_oe': [],
        'train_1_cov': [], 'val_1_cov': []
    }

    print(f"开始训练 AIMNet 模型 (Batch Size: {batch_size})...")
    print("-" * 50)
    
    # 辅助函数：将全量数据搬运到 GPU
    def to_device(data_dict):
        return {
            'x_list': [x.to(device) for x in data_dict['X']],
            'W': data_dict['W'].to(device),
            'Y': data_dict['Y'].to(device),
            'G': data_dict['G'].to(device),
            'edge_index': data_dict['C'].to(device) # 注意：标签相关性图不用切分，它是全局的
        }
        
    train_inputs = to_device(train_data)
    val_inputs = to_device(val_data)
    n_train_samples = train_inputs['Y'].shape[0]
    
    for epoch in range(1, epochs + 1):
        # ================= 训练阶段 (Mini-Batch) =================
        model.train()
        epoch_train_loss = 0.0
        
        # 1. 打乱训练集索引
        indices = torch.randperm(n_train_samples, device=device)
        
        train_P_list = [] # 用于收集每个 batch 的预测概率，用于计算全局 AP
        train_Y_list = []
        train_G_list = []
        
        # 2. 遍历每个 Batch
        for start_idx in range(0, n_train_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_train_samples)
            batch_idx = indices[start_idx:end_idx]
            
            # 切割当前 batch 的数据
            batch_x_list = [x[batch_idx] for x in train_inputs['x_list']]
            batch_W = train_inputs['W'][batch_idx]
            batch_Y = train_inputs['Y'][batch_idx]
            batch_G = train_inputs['G'][batch_idx]
            
            optimizer.zero_grad()
            
            # 前向传播 (注意 edge_index 是标签图，不需要切分)
            fused_logits = model(batch_x_list, train_inputs['edge_index'], batch_W)
            loss = model.compute_loss(fused_logits, batch_Y, batch_G)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_train_loss += loss.item() * len(batch_idx) # 累加损失
            
            # 收集结果用于计算指标
            train_P_list.append(torch.sigmoid(fused_logits).detach())
            train_Y_list.append(batch_Y)
            train_G_list.append(batch_G)
            
        # 计算该 epoch 的平均训练 loss
        avg_train_loss = epoch_train_loss / n_train_samples
        
        # 拼接所有的 batch 结果计算全局训练指标
        all_train_P = torch.cat(train_P_list, dim=0)
        all_train_Y = torch.cat(train_Y_list, dim=0)
        all_train_G = torch.cat(train_G_list, dim=0)
        train_metrics = compute_metrics(all_train_Y, all_train_P, all_train_G)
        
        # ================= 验证阶段 (为了防止验证集也爆显存，同样用 Batch，但不需要打乱) =================
        model.eval()
        epoch_val_loss = 0.0
        val_P_list = []
        n_val_samples = val_inputs['Y'].shape[0]
        
        with torch.no_grad():
            for start_idx in range(0, n_val_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_val_samples)
                
                batch_x_list = [x[start_idx:end_idx] for x in val_inputs['x_list']]
                batch_W = val_inputs['W'][start_idx:end_idx]
                batch_Y = val_inputs['Y'][start_idx:end_idx]
                batch_G = val_inputs['G'][start_idx:end_idx]
                
                val_logits = model(batch_x_list, val_inputs['edge_index'], batch_W)
                val_loss = model.compute_loss(val_logits, batch_Y, batch_G)
                
                epoch_val_loss += val_loss.item() * (end_idx - start_idx)
                val_P_list.append(torch.sigmoid(val_logits))
                
        avg_val_loss = epoch_val_loss / n_val_samples
        all_val_P = torch.cat(val_P_list, dim=0)
        val_metrics = compute_metrics(val_inputs['Y'], all_val_P, val_inputs['G'])
            
        # ================= 日志与保存 =================
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_ap'].append(train_metrics['AP'])
        history['val_ap'].append(val_metrics['AP'])
        history['train_auc'].append(train_metrics['AUC'])
        history['val_auc'].append(val_metrics['AUC'])
        history['train_1_hl'].append(train_metrics['1-HL'])
        history['val_1_hl'].append(val_metrics['1-HL'])
        history['train_1_rl'].append(train_metrics['1-RL'])
        history['val_1_rl'].append(val_metrics['1-RL'])
        history['train_1_oe'].append(train_metrics['1-OE'])
        history['val_1_oe'].append(val_metrics['1-OE'])
        history['train_1_cov'].append(train_metrics['1-COV'])
        history['val_1_cov'].append(val_metrics['1-COV'])

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch [{epoch:03d}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
            print(f"  [Train] AP: {train_metrics['AP']:.4f} | AUC: {train_metrics['AUC']:.4f} | 1-HL: {train_metrics['1-HL']:.4f}")
            print(f"          1-RL: {train_metrics['1-RL']:.4f} | 1-OE: {train_metrics['1-OE']:.4f} | 1-COV: {train_metrics['1-COV']:.4f}")
            print(f"  [Val]   AP: {val_metrics['AP']:.4f} | AUC: {val_metrics['AUC']:.4f} | 1-HL: {val_metrics['1-HL']:.4f}")
            print(f"          1-RL: {val_metrics['1-RL']:.4f} | 1-OE: {val_metrics['1-OE']:.4f} | 1-COV: {val_metrics['1-COV']:.4f}")
            print("-" * 60)
            
        if val_metrics["AP"] > best_ap:
            best_ap = val_metrics["AP"]
            torch.save(model.state_dict(), "best_aimnet_model.pt")

        if epoch % 20 == 0 or epoch == epochs:
            plot_training_history(history, save_path="training_curves_5.27.png")
            # 【修改】：将指标历史导出为标准的电子表格文件
            # pd.DataFrame(history).to_excel("training_history.xlsx", index=False)
            print(f"--> 训练性能图表与详细历史数据 (.xlsx) 已保存")
            
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
    set_all_seeds(42)

    data = torch.load("data/processed/train_and_test_espgame_03test_rate_05missing_rate.pt")
    train_data = data['train']
    val_data = data['test']

    view_dims = features_num_extract(train_data)
    n_classes = class_num_extract(train_data)

    # 2. 实例化模型
    model = models.AIMNet(view_dims=view_dims, n_classes=n_classes, d_e=512, tau=0.2)
    
    # 3. 运行训练 (如果没有 GPU，可以将 device 改为 "cpu")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trained_model = train_aimnet(model, train_data, val_data, epochs=100, lr=0.1, device=device) 