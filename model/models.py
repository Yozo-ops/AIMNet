import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class AIMNet(nn.Module):
    def __init__(self, view_dims, n_classes, heads=4, d_e=512, tau=1.0):
        super(AIMNet, self).__init__()
        self.tau = tau
        self.m_views = len(view_dims)
        self.heads = heads

        # 1. 特征分支：各视图独立的 MLP
        self.feature_extractors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_v, d_e),
                nn.LeakyReLU(0.1),
                # nn.Linear(d_e, d_e),
                # nn.LayerNorm(d_e),
                nn.Dropout()
            ) for d_v in view_dims
        ])
        
        
        # 2. 标签分支：初始嵌入和 K-head GAT
        # 采用独热编码而不进行初始化，区分每一个标签；由于各个标签嵌入正交，GAT没有办法计算注意力？
        self.label_embed = nn.Parameter(torch.eye(n_classes, n_classes) + torch.normal(0, 0.1, size=(n_classes, n_classes)))
        # 正态分布初始化？
        nn.init.normal_(self.label_embed)
        self.dropout1 = nn.Dropout()
        self.dropout2 = nn.Dropout()
        self.gat1 = GATConv(n_classes, d_e, heads=heads, dropout=0.5, concat=True)
        self.gat2 = GATConv(d_e*heads, d_e, dropout=0.5, concat=False)
        



        # 3. 分类器：映射交互后的嵌入到 1 维 Logit (对应公式 8 后续)
        self.classifier = nn.Conv1d(n_classes, n_classes, d_e, groups=n_classes)

        self.parameters_reset()

    def parameters_reset(self):
        # 统一的参数初始化
        for m in self.feature_extractors.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=0.1, nonlinearity='leaky_relu')
                nn.init.uniform_(m.bias, 0, 0.1)
                
        for m in [self.gat1, self.gat2]:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    # def _init_gat_weights(self, module):
    #     if isinstance(module, nn.Linear):
    #         nn.init.xavier_uniform_(module.weight)
    #         if module.bias is not None:
    #             nn.init.zeros_(module.bias)

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

        A_bar.fill_diagonal_(0.0)

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
        out = self.dropout1(self.label_embed)
        out = self.gat1(out, edge_index)
        out = self.dropout2(out)
        out = self.gat2(out, edge_index)
        label_features = F.elu(out) # [n_classes, d_e]
        
        z_list = [self.feature_extractors[v](x_list[v]) for v in range(self.m_views)]
        z_hat_list, A_bar = self.attention_induced_imputation(z_list, W)
        
        P_list = []
        activated_L = torch.sigmoid(label_features).unsqueeze(0) 
        
        for v in range(self.m_views):
            z_hat_v = z_hat_list[v].unsqueeze(1) 
            B_v = activated_L * z_hat_v          # [n_sample, n_classes, d_e]
            
            # 【配合分类器的修改】直接传入 B_v，利用 Conv1d 独立处理每个类别
            P_v = self.classifier(B_v).squeeze(2) # [n_sample, n_classes]
            P_list.append(P_v)
            
        P_stack = torch.stack(P_list, dim=2) 
        
        unshrunk_A = self.tau * torch.log(A_bar + 1e-8) 
        Q_list = []
        for v in range(self.m_views):
            mask_j = W[:, v].unsqueeze(0) 
            masked_unshrunk_A = unshrunk_A * mask_j + (1.0 - mask_j) * -1e9
            q_v, _ = torch.max(masked_unshrunk_A, dim=1) 
            Q_list.append(q_v)
            
        Q = torch.stack(Q_list, dim=1) 
        Q = torch.clamp(Q, min=0.0, max=1.0)
        Q_prime = (1.0 - W) * Q + W    
        
        Q_prime_expanded = Q_prime.unsqueeze(1) 
        fused_logits = torch.sum(P_stack * Q_prime_expanded, dim=2) / (torch.sum(Q_prime, dim=1, keepdim=True) + 1e-8)
        
        return fused_logits

    def compute_loss(self, fused_logits, Y, G):
        loss_matrix = F.binary_cross_entropy_with_logits(fused_logits, Y, reduction='none')
        masked_loss = loss_matrix * G
        loss = torch.sum(masked_loss) / (torch.sum(G) + 1e-8)
        return loss