import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any
import warnings
warnings.filterwarnings('ignore')

class CrossNetwork(nn.Module):
    """Cross Network for DCN-v2"""
    
    def __init__(self, input_dim: int, num_layers: int = 3, low_rank: int = 32):
        super(CrossNetwork, self).__init__()
        self.num_layers = num_layers
        self.input_dim = input_dim
        
        # 使用低秩分解来减少参数数量
        self.w_layers = nn.ModuleList([
            nn.Linear(input_dim, low_rank, bias=False) for _ in range(num_layers)
        ])
        self.v_layers = nn.ModuleList([
            nn.Linear(low_rank, input_dim, bias=False) for _ in range(num_layers)
        ])
        self.bias_layers = nn.ModuleList([
            nn.Linear(input_dim, 1, bias=False) for _ in range(num_layers)
        ])
        
        # 初始化参数
        self._init_weights()
    
    def _init_weights(self):
        for layer in self.w_layers:
            nn.init.xavier_uniform_(layer.weight)
        for layer in self.v_layers:
            nn.init.xavier_uniform_(layer.weight)
        for layer in self.bias_layers:
            nn.init.zeros_(layer.weight)
    
    def forward(self, x):
        x0 = x  # 保存原始输入
        xl = x  # 当前层输入
        
        for i in range(self.num_layers):
            # 计算交叉项: x_0 * (W @ x_l + b)
            xl_w = self.w_layers[i](xl)  # [batch_size, low_rank]
            xl_v = self.v_layers[i](xl_w)  # [batch_size, input_dim]
            xl_bias = self.bias_layers[i](xl)  # [batch_size, 1]
            
            # 广播bias
            xl_bias = xl_bias.expand_as(xl_v)
            
            # 计算新的xl
            xl = x0 * (xl_v + xl_bias) + xl
        
        return xl

class DeepNetwork(nn.Module):
    """Deep Network for DCN-v2"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], dropout_rate: float = 0.3):
        super(DeepNetwork, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        self.network = nn.Sequential(*layers)
        
        # 初始化参数
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        return self.network(x)

class DCNv2(nn.Module):
    """Deep Cross Network v2 for Flight Ranking"""
    
    def __init__(self, 
                 input_dim: int,
                 embedding_dims: Dict[str, int] = None,
                 cross_layers: int = 3,
                 cross_low_rank: int = 32,
                 deep_layers: List[int] = [512, 256, 128],
                 dropout_rate: float = 0.3,
                 use_bn: bool = True):
        super(DCNv2, self).__init__()
        
        self.input_dim = input_dim
        self.embedding_dims = embedding_dims or {}
        self.use_bn = use_bn
        
        # 输入层批归一化
        if use_bn:
            self.input_bn = nn.BatchNorm1d(input_dim)
        
        # Cross Network
        self.cross_network = CrossNetwork(input_dim, cross_layers, cross_low_rank)
        
        # Deep Network
        self.deep_network = DeepNetwork(input_dim, deep_layers, dropout_rate)
        
        # 最终输出层
        final_input_dim = input_dim + deep_layers[-1]  # cross + deep
        self.final_layer = nn.Sequential(
            nn.Linear(final_input_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1)
        )
        
        # 初始化最终层
        self._init_final_layer()
    
    def _init_final_layer(self):
        for module in self.final_layer:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        # 输入批归一化
        if self.use_bn:
            x = self.input_bn(x)
        
        # Cross Network
        cross_output = self.cross_network(x)
        
        # Deep Network
        deep_output = self.deep_network(x)
        
        # 拼接cross和deep的输出
        final_input = torch.cat([cross_output, deep_output], dim=1)
        
        # 最终输出
        output = self.final_layer(final_input)
        
        return output

class DCNv2Ranker(nn.Module):
    """DCN-v2 Ranker for Flight Ranking with Ranking Loss"""
    
    def __init__(self, 
                 input_dim: int,
                 embedding_dims: Dict[str, int] = None,
                 cross_layers: int = 3,
                 cross_low_rank: int = 32,
                 deep_layers: List[int] = [512, 256, 128],
                 dropout_rate: float = 0.3,
                 use_bn: bool = True):
        super(DCNv2Ranker, self).__init__()
        
        self.dcn = DCNv2(
            input_dim=input_dim,
            embedding_dims=embedding_dims,
            cross_layers=cross_layers,
            cross_low_rank=cross_low_rank,
            deep_layers=deep_layers,
            dropout_rate=dropout_rate,
            use_bn=use_bn
        )
    
    def forward(self, x):
        scores = self.dcn(x)
        return scores.squeeze(-1)  # [batch_size]
    
    def predict_ranking(self, x):
        """预测排名分数"""
        self.eval()
        with torch.no_grad():
            scores = self.forward(x)
        return scores

class RankingLoss(nn.Module):
    """Ranking Loss for Flight Ranking"""
    
    def __init__(self, margin: float = 1.0, reduction: str = 'mean'):
        super(RankingLoss, self).__init__()
        self.margin = margin
        self.reduction = reduction
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
    
    def forward(self, scores, labels, groups=None):
        """
        Args:
            scores: [batch_size] 预测分数
            labels: [batch_size] 真实标签 (0 或 1)
            groups: [batch_size] 组别标识 (ranker_id)
        """
        # 基础二分类损失
        bce_loss = self.bce_loss(scores, labels.float())
        
        if groups is not None:
            # 添加组内排序损失
            ranking_loss = self._compute_ranking_loss(scores, labels, groups)
            total_loss = bce_loss + ranking_loss
        else:
            total_loss = bce_loss
        
        if self.reduction == 'mean':
            return total_loss.mean()
        elif self.reduction == 'sum':
            return total_loss.sum()
        else:
            return total_loss
    
    def _compute_ranking_loss(self, scores, labels, groups):
        """计算组内排序损失"""
        ranking_losses = []
        
        # 对每个组计算排序损失
        unique_groups = torch.unique(groups)
        
        for group_id in unique_groups:
            group_mask = (groups == group_id)
            group_scores = scores[group_mask]
            group_labels = labels[group_mask]
            
            # 如果组内只有一个样本，跳过
            if group_scores.size(0) <= 1:
                continue
            
            # 计算正负样本对的损失
            pos_mask = group_labels == 1
            neg_mask = group_labels == 0
            
            if pos_mask.sum() > 0 and neg_mask.sum() > 0:
                pos_scores = group_scores[pos_mask]
                neg_scores = group_scores[neg_mask]
                
                # 向量化计算所有正负样本对的损失
                # pos_scores: [n_pos, 1], neg_scores: [1, n_neg]
                # 广播计算: [n_pos, n_neg]
                pos_expanded = pos_scores.unsqueeze(1)  # [n_pos, 1]
                neg_expanded = neg_scores.unsqueeze(0)  # [1, n_neg]
                
                # 计算所有正负样本对的损失
                pairwise_losses = torch.clamp(
                    self.margin - (pos_expanded - neg_expanded), min=0.0
                )
                
                # 将所有损失展平并添加到列表中
                ranking_losses.append(pairwise_losses.flatten())
        
        if ranking_losses:
            # 将所有损失张量拼接并求平均
            return torch.cat(ranking_losses).mean()
        else:
            return torch.tensor(0.0, device=scores.device)

class DCNv2Trainer:
    """DCN-v2 训练器"""
    
    def __init__(self, model, optimizer, criterion, device='cuda'):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.model.to(device)
    
    def train_epoch(self, train_loader, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        num_batches = len(train_loader)
        
        for batch_idx, batch in enumerate(train_loader):
            # 获取数据
            features = batch['features'].to(self.device)
            labels = batch['labels'].to(self.device)
            groups = batch.get('groups', None)
            
            if groups is not None:
                groups = groups.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(features)
            loss = self.criterion(outputs, labels, groups)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}/{num_batches}, Loss: {loss.item():.4f}')
        
        return total_loss / num_batches
    
    def evaluate(self, val_loader):
        """评估模型"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                features = batch['features'].to(self.device)
                labels = batch['labels'].to(self.device)
                groups = batch.get('groups', None)
                
                if groups is not None:
                    groups = groups.to(self.device)
                
                outputs = self.model(features)
                loss = self.criterion(outputs, labels, groups)
                
                total_loss += loss.item()
                
                # 收集预测和真实标签
                predictions = torch.sigmoid(outputs).cpu().numpy()
                all_predictions.extend(predictions)
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        
        # 计算AUC
        try:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(all_labels, all_predictions)
        except ImportError:
            print("sklearn未安装，跳过AUC计算")
            auc = 0.0
        
        return avg_loss, auc
    
    def save_model(self, path):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
    
    def load_model(self, path):
        """加载模型"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

if __name__ == "__main__":
    # 测试DCN-v2模型
    print("测试DCN-v2模型...")
    
    # 创建模型
    input_dim = 100  # 假设有100个特征
    model = DCNv2Ranker(
        input_dim=input_dim,
        cross_layers=3,
        cross_low_rank=32,
        deep_layers=[512, 256, 128],
        dropout_rate=0.3
    )
    
    # 测试前向传播
    batch_size = 32
    test_input = torch.randn(batch_size, input_dim)
    
    with torch.no_grad():
        output = model(test_input)
        print(f"模型输出形状: {output.shape}")
        print(f"输出样例: {output[:5]}")
    
    # 测试损失函数
    criterion = RankingLoss()
    labels = torch.randint(0, 2, (batch_size,))
    groups = torch.randint(0, 10, (batch_size,))
    
    loss = criterion(output, labels, groups)
    print(f"损失值: {loss.item()}")
    
    print("DCN-v2模型测试完成！") 