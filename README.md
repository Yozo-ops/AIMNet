# Attention-Induced Embedding Imputation for Incomplete Multi-View Partial

## 5.23 先前忘记写工作日志了

## 5.24 初步确认可能是由于数据集标签为极不均衡导致过拟合与AP值提升不明显

## 5.25 一一对照源码进行修改
    - 调整标签嵌入矩阵为独热编码，源码使用正态分布初始化。复现不进行初始化。
    - 调整优化器Adma为SGD
    - 调整隐藏层为512
    - 调整分类器为一维卷积
    训练集上，模型结果得到改善；测试集改善不强。
### 调整评估标准AP从mAP更改为Sample-wise AP，与原文对齐。