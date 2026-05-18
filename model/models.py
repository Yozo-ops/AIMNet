import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class AIMNet(nn.Module):
    def __init__(self, view_dims, n_classes, d_e=512, tau=1.0):
        super(AIMNet, self).__init__()
        self.tau = tau
        self.m_views = len(view_dims)
        
        # 1. 特征分支：各视图独立的 MLP
        self.feature_extractors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_v, d_e),
                nn.ReLU(),
                nn.Linear(d_e, d_e)
            ) for d_v in view_dims
        ])
        
        # 2. 标签分支：初始嵌入和 K-head GAT
        self.label_embed = nn.Parameter(torch.Tensor(n_classes, n_classes))
        nn.init.xavier_uniform_(self.label_embed)
        self.gat = GATConv(n_classes, d_e, heads=4, concat=False)
        
        # 3. 分类器：映射交互后的嵌入到 1 维 Logit (对应公式 8 后续)
        self.classifier = nn.Linear(d_e, 1)

    def attention_induced_imputation(self, z_list, W):
        A_list = []
        for v in range(self.m_views):
            z_norm = F.normalize(z_list[v], p=2, dim=1)
            sim = torch.matmul(z_norm, z_norm.T) / self.tau
            A_v = torch.exp(sim)
            
            mask_2d = torch.outer(W[:, v], W[:, v])
            A_v = A_v * mask_2d
            A_list.append(A_v)

        A_stack = torch.stack(A_list, dim=0)
        A_bar, _ = torch.max(A_stack, dim=0)

        sum_A_bar = torch.sum(A_bar, dim=1, keepdim=True) + 1e-8
        A_bar_normalized = A_bar / sum_A_bar

        z_hat_list = []
        for v in range(self.m_views):
            w_v = W[:, v].unsqueeze(1)
            z_available = z_list[v] * w_v 
            z_bar_v = torch.matmul(A_bar_normalized, z_available)
            z_hat_v = z_list[v] * w_v + z_bar_v * (1.0 - w_v)
            z_hat_list.append(z_hat_v)
            
        return z_hat_list, A_bar

    def forward(self, x_list, edge_index, W):
        """
        Args:
            x_list: 各视图特征列表，每个元素为 [n_sample, d_v]
            edge_index: 标签图边索引 [2, num_edges]
            W: 视图缺失指示矩阵 [n_sample, m_views]
        Returns:
            fused_P: 最终的多视图融合预测概率 [n_sample, n_classes]
        """
        # ---- 第一、二阶段：提取标签与实例基础表征 ----
        label_features = F.leaky_relu(self.gat(self.label_embed, edge_index)) # [n_classes, d_e]
        z_list = [self.feature_extractors[v](x_list[v]) for v in range(self.m_views)]
        
        # ---- 第三阶段：注意力诱导特征填补 ----
        z_hat_list, A_bar = self.attention_induced_imputation(z_list, W)
        
        # ---- 第四阶段：特征交互与分类预测 (公式 8) ----
        P_list = []
        activated_L = torch.sigmoid(label_features).unsqueeze(0) # [1, n_classes, d_e]
        
        for v in range(self.m_views):
            z_hat_v = z_hat_list[v].unsqueeze(1) # [n_sample, 1, d_e]
            B_v = activated_L * z_hat_v          # 广播相乘得到 [n_sample, n_classes, d_e]
            P_v = self.classifier(B_v).squeeze(2) # [n_sample, n_classes]
            P_list.append(P_v)
            
        P_stack = torch.stack(P_list, dim=2) # [n_sample, n_classes, m_views]
        
        # ---- 第四阶段：计算原始注意力置信度 (公式 9 & 10) ----
        unshrunk_A = self.tau * torch.log(A_bar + 1e-8) # 还原未缩放的关联度 [n_sample, n_sample]
        Q_list = []
        for v in range(self.m_views):
            mask_j = W[:, v].unsqueeze(0) # [1, n_sample]
            # 过滤掉不可用的邻居，将其赋为极小值值，避免干扰 max 的提取
            masked_unshrunk_A = unshrunk_A * mask_j + (1.0 - mask_j) * -1e9
            q_v, _ = torch.max(masked_unshrunk_A, dim=1) # [n_sample]
            Q_list.append(q_v)
            
        Q = torch.stack(Q_list, dim=1) # [n_sample, m_views]
        Q_prime = (1.0 - W) * Q + W    # 真实存在的实例置信度固化为 1
        
        # ---- 第四阶段：多视图后期动态融合 (公式 11) ----
        Q_prime_expanded = Q_prime.unsqueeze(1) # [n_sample, 1, m_views]
        # 对各个视图的预测进行置信度加权平均
        fused_logits = torch.sum(P_stack * Q_prime_expanded, dim=2) / (torch.sum(Q_prime, dim=1, keepdim=True) + 1e-8)
        fused_P = torch.sigmoid(fused_logits) # [n_sample, n_classes]
        
        return fused_P

    def compute_loss(self, fused_P, Y, G):
        """
        对应公式 (12): Masked Binary Cross Entropy Loss
        Args:
            fused_P: 模型输出的融合预测概率 [n_sample, n_classes]
            Y: 真实标签矩阵 [n_sample, n_classes]
            G: 标签缺失指示矩阵 [n_sample, n_classes] (1代表已知，0代表缺失)
        """
        eps = 1e-8
        # 计算每个位置完整的交叉熵
        loss_matrix = -(Y * torch.log(fused_P + eps) + (1.0 - Y) * torch.log(1.0 - fused_P + eps))
        
        # 使用掩码矩阵 G 过滤掉未知标签的损失
        masked_loss = loss_matrix * G
        
        # 仅对真实存在的标签计算平均损失
        loss = torch.sum(masked_loss) / (torch.sum(G) + 1e-8)
        return loss