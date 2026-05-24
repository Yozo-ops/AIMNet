import torch
import torch.optim as optim
import numpy as np
from sklearn.metrics import average_precision_score, hamming_loss, roc_auc_score
import matplotlib.pyplot as plt
import models
import os 


def compute_metrics(y_true, y_pred, G):
    y_true_np = y_true.cpu().detach().numpy()
    y_pred_np = y_pred.cpu().detach().numpy()
    G_np = G.cpu().detach().numpy()
    
    # 1. Hamming Loss
    y_pred_bin = (y_pred_np >= 0.5).astype(int)
    actual_elements = np.sum(G_np)
    hl = np.sum((y_true_np != y_pred_bin) * G_np) / actual_elements if actual_elements > 0 else 0.0
    one_minus_hl = 1.0 - hl
    
    # 2. 计算 Macro AP 和 Macro AUC
    ap_list = []
    auc_list = []
    n_classes = y_true_np.shape[1]
    
    for i in range(n_classes):
        # 仅提取当前类别下，G=1 (已知) 的样本
        valid_idx = (G_np[:, i] == 1)
        y_t = y_true_np[valid_idx, i]
        y_p = y_pred_np[valid_idx, i]
        
        # sklearn 限制：只有该类别同时存在正样本和负样本时，计算指标才有意义
        if len(np.unique(y_t)) == 2:
            ap_list.append(average_precision_score(y_t, y_p))
            auc_list.append(roc_auc_score(y_t, y_p))
            
    ap = np.mean(ap_list) if len(ap_list) > 0 else 0.0
    auc = np.mean(auc_list) if len(auc_list) > 0 else 0.5

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

def train_aimnet(model, train_data, val_data, epochs=200, lr=0.001, weight_decay=1e-5, batch_size=512, device="cuda"):
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    
    best_ap = 0.0
    history = {'train_loss': [], 'val_loss': [], 'train_ap': [], 'val_ap': [], 'train_auc': [], 'val_auc': []}

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

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch [{epoch:03d}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
            print(f"  [Train] AP: {train_metrics['AP']:.4f}, 1-HL: {train_metrics['1-HL']:.4f}, AUC: {train_metrics['AUC']:.4f}")
            print(f"  [Val]   AP: {val_metrics['AP']:.4f}, 1-HL: {val_metrics['1-HL']:.4f}, AUC: {val_metrics['AUC']:.4f}")
            print("-" * 50)
            
        if val_metrics["AP"] > best_ap:
            best_ap = val_metrics["AP"]
            torch.save(model.state_dict(), "best_aimnet_model.pt")

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

    data = torch.load("data/processed/train_and_test_espgame_03test_rate_05missing_rate.pt")
    train_data = data['train']
    val_data = data['test']

    view_dims = features_num_extract(train_data)
    n_classes = class_num_extract(train_data)

    # 2. 实例化模型
    model = models.AIMNet(view_dims=view_dims, n_classes=n_classes, d_e=512, tau=0.2)
    
    # 3. 运行训练 (如果没有 GPU，可以将 device 改为 "cpu")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trained_model = train_aimnet(model, train_data, val_data, epochs=100, lr=1, device=device) 