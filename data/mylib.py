import scipy.io as sio
import torch
import numpy as np

def load_and_preprocess(file_path):
    # 加载 .mat 文件
    data = sio.loadmat(file_path)
    
    # 1. 提取标签 (假设键名为 'Y')
    Y = torch.from_numpy(data['label']).float()
    n_samples, n_classes = Y.shape
    
    # 2. 提取特征 (处理 (1, 6) 嵌套结构)
    X_raw = data['X'] # 这里的 X 形状为 (1, 6)
    X_list = []
    n_views = X_raw.shape[1]
    
    for v in range(n_views):
        # 提取第 v 个视图：索引为 [0, v]
        view_data = X_raw[0, v]
        # 转换为 FloatTensor 方便后续神经网络计算
        X_list.append(torch.from_numpy(view_data).float())
        
    return X_list, Y, n_samples, n_views, n_classes

def create_masks(n_samples, n_views, n_classes, missing_rate=0.5):
    # 初始化视图掩码 W (n, m) [cite: 77]
    W = torch.ones(n_samples, n_views)
    for v in range(n_views):
        perm = torch.randperm(n_samples)
        num_missing = int(n_samples * missing_rate)
        W[perm[:num_missing], v] = 0
    
    # 强制约束：每个样本至少有一个视图为 1
    for i in range(n_samples):
        if W[i].sum() == 0:
            W[i, torch.randint(0, n_views, (1,))] = 1
            
    # 2. 标签掩码 G (n, c) [cite: 77]
    # 随机掩盖 50% 的正负标签生成部分多标签数据 [cite: 243]
    G = (torch.rand(n_samples, n_classes) > missing_rate).float()
    
    return W, G

def compute_C(Y, G):
    # 只根据可见标签计算相关性 [cite: 218]
    available_Y = Y * G
    n, c = available_Y.shape
    C = torch.zeros(c, c)
    
    for i in range(c):
        occurrence_i = available_Y[:, i].sum()
        if occurrence_i > 0:
            for j in range(c):
                if i == j: continue # 对角线设为 0 
                co_occurrence = (available_Y[:, i] * available_Y[:, j]).sum()
                C[i, j] = co_occurrence / occurrence_i
    return C

def split_and_finalize_data(data, train_ratio=0.7):
    # 加载初步处理的数据 (包含 X_list, Y, W)
    X_list = data['X']
    Y = data['Y']
    W = data['W']
    G = data['G']

    n_samples = Y.shape[0]
    n_train = int(n_samples * train_ratio)
    
    # 1. 随机打乱索引
    indices = torch.randperm(n_samples)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]
    
    # 2. 拆分特征、标签和视图掩码
    # 特征是列表，需要循环处理
    train_X = [view[train_idx] for view in X_list]
    test_X = [view[test_idx] for view in X_list]
    
    train_Y = Y[train_idx]
    test_Y = Y[test_idx]
    
    train_W = W[train_idx]
    test_W = W[test_idx]
    
    train_G = G[train_idx]
    test_G = G[test_idx]
    
    # 4. 在训练集上计算标签相关性矩阵 C [cite: 198]
    C = compute_C(train_Y, train_G)
    C = convert_C_to_edge_index(C)

    C_test = compute_C(test_Y, test_G)
    C_test = convert_C_to_edge_index(C_test)

    return {
        'train': {'X': train_X, 'Y': train_Y, 'W': train_W, 'G': train_G, 'C': C, },
        'test': {'X': test_X, 'Y': test_Y, 'W': test_W, 'G': test_G, 'C': C_test}
    }


def convert_C_to_edge_index(C, threshold=0.0):
    """
    将标签相关性矩阵 C 转换为 PyG 的 edge_index 格式。
    
    Args:
        C: 标签相关性矩阵 [c, c]
        threshold: 判定边存在的阈值，论文中暗示 C_ij > 0 即可
        
    Returns:
        edge_index: PyG 格式的边索引张量 [2, num_edges], dtype=torch.long
    """
    # 找到所有大于阈值的矩阵元素坐标 (行, 列)
    # 这等价于寻找所有存在的边
    edges = torch.nonzero(C > threshold, as_tuple=False)
    
    # torch.nonzero 返回的是 [num_edges, 2]
    # PyG 要求的是 [2, num_edges]，且第一行是 source node，第二行是 target node
    # 因此需要进行转置 (transpose)
    edge_index = edges.t().contiguous()
    
    return edge_index

def data_processing(file_path, train_ratio=0.7, missing_rate=0.5):
    '''
    处理所有数据，集成上面所有函数功能。
    '''
    X_list, Y, n_samples, n_views, n_classes = load_and_preprocess(file_path)
    W, G = create_masks(n_samples, n_views, n_classes, missing_rate)
    data = {'X': X_list, 'Y': Y, 'W': W, 'G': G}
    data = split_and_finalize_data(data, train_ratio)
    return data

if __name__ == "__main__":
    data = torch.load("data/processed/train_and_test_corel5k_03test_rate_05missing_rate.pt")
    print(data['test'].keys())