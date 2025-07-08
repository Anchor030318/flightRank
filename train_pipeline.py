import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score
import pickle
import joblib
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# 导入我们的模块
from data_preprocessing import DataPreprocessor
from feature_engineering import FeatureEngineer
from dcn_model import DCNv2Ranker, RankingLoss, DCNv2Trainer

class FlightDataset(Dataset):
    """航班数据集"""
    
    def __init__(self, features, labels, groups=None):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
        self.groups = torch.LongTensor(groups) if groups is not None else None
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        item = {
            'features': self.features[idx],
            'labels': self.labels[idx]
        }
        if self.groups is not None:
            item['groups'] = self.groups[idx]
        return item

class FlightRankingPipeline:
    """航班排序训练管道"""
    
    def __init__(self, config: Dict[str, Any], log_file: str = None):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 初始化组件
        self.preprocessor = DataPreprocessor()
        self.feature_engineer = FeatureEngineer(log_file=log_file)
        self.scaler = StandardScaler()
        self.model = None
        self.trainer = None
        
        # 训练历史
        self.train_history = {
            'train_loss': [],
            'val_loss': [],
            'val_auc': []
        }
    
    def load_data(self, train_path: str, test_path: str = None):
        """加载数据"""
        print("加载数据...")
        
        # 加载训练数据
        self.train_data = pd.read_parquet(train_path)
        print(f"训练数据形状: {self.train_data.shape}")
        
        # 加载测试数据（如果提供）
        if test_path:
            self.test_data = pd.read_parquet(test_path)
            print(f"测试数据形状: {self.test_data.shape}")
        else:
            self.test_data = None
    
    def preprocess_data(self):
        """预处理数据"""
        print("开始数据预处理...")
        
        # 预处理训练数据
        self.train_processed = self.preprocessor.fit_transform(self.train_data)
        
        # 预处理测试数据
        if self.test_data is not None:
            self.test_processed = self.preprocessor.transform(self.test_data)
        
        print("数据预处理完成")
    
    def feature_engineering(self):
        """特征工程"""
        print("开始特征工程...")
        
        # 训练数据特征工程
        self.train_featured = self.feature_engineer.fit_transform(self.train_processed)
        
        # 测试数据特征工程
        if hasattr(self, 'test_processed'):
            self.test_featured = self.feature_engineer.transform(self.test_processed)
        
        print("特征工程完成")
    
    def prepare_features(self):
        """准备最终特征"""
        print("准备训练特征...")
        
        # 获取特征列
        feature_columns = [col for col in self.train_featured.columns 
                          if col not in ['Id', 'selected']]
        
        # 训练数据
        X_train = self.train_featured[feature_columns].values
        y_train = self.train_featured['selected'].values
        
        # 处理分组信息
        if 'ranker_id' in self.train_featured.columns:
            groups_train = self.train_featured['ranker_id'].values
            # 将ranker_id转换为数字标识
            unique_groups = np.unique(groups_train)
            group_mapping = {group: idx for idx, group in enumerate(unique_groups)}
            groups_train = np.array([group_mapping[g] for g in groups_train])
        else:
            groups_train = None
        
        # 分割训练和验证数据
        if groups_train is not None:
            X_train, X_val, y_train, y_val, groups_train, groups_val = train_test_split(
                X_train, y_train, groups_train, test_size=0.2, random_state=42, stratify=y_train
            )
        else:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
            )
            groups_val = None
        
        # 特征标准化
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)
        
        # 存储特征信息
        self.feature_columns = feature_columns
        self.input_dim = X_train.shape[1]
        
        # 创建数据集
        self.train_dataset = FlightDataset(X_train, y_train, groups_train)
        self.val_dataset = FlightDataset(X_val, y_val, groups_val)
        
        print(f"训练集大小: {len(self.train_dataset)}")
        print(f"验证集大小: {len(self.val_dataset)}")
        print(f"特征维度: {self.input_dim}")
    
    def create_model(self):
        """创建模型"""
        print("创建DCN-v2模型...")
        
        model_config = self.config.get('model', {})
        
        self.model = DCNv2Ranker(
            input_dim=self.input_dim,
            cross_layers=model_config.get('cross_layers', 3),
            cross_low_rank=model_config.get('cross_low_rank', 32),
            deep_layers=model_config.get('deep_layers', [512, 256, 128]),
            dropout_rate=model_config.get('dropout_rate', 0.3),
            use_bn=model_config.get('use_bn', True)
        )
        
        # 打印模型信息
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"模型总参数: {total_params:,}")
        print(f"可训练参数: {trainable_params:,}")
    
    def setup_training(self):
        """设置训练"""
        print("设置训练参数...")
        
        train_config = self.config.get('training', {})
        
        # 优化器
        optimizer_name = train_config.get('optimizer', 'adam')
        lr = train_config.get('learning_rate', 0.001)
        
        if optimizer_name.lower() == 'adam':
            optimizer = torch.optim.Adam(
                self.model.parameters(), 
                lr=lr,
                weight_decay=train_config.get('weight_decay', 1e-5)
            )
        elif optimizer_name.lower() == 'adamw':
            optimizer = torch.optim.AdamW(
                self.model.parameters(), 
                lr=lr,
                weight_decay=train_config.get('weight_decay', 1e-5)
            )
        else:
            raise ValueError(f"不支持的优化器: {optimizer_name}")
        
        # 损失函数
        criterion = RankingLoss(
            margin=train_config.get('margin', 1.0),
            reduction='mean'
        )
        
        # 训练器
        self.trainer = DCNv2Trainer(
            model=self.model,
            optimizer=optimizer,
            criterion=criterion,
            device=self.device
        )
        
        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            patience=train_config.get('patience', 5),
            factor=0.5,
            verbose=True
        )
    
    def train_model(self):
        """训练模型"""
        print("开始训练模型...")
        
        train_config = self.config.get('training', {})
        
        # 创建数据加载器
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=train_config.get('batch_size', 256),
            shuffle=True,
            num_workers=train_config.get('num_workers', 4)
        )
        
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=train_config.get('batch_size', 256),
            shuffle=False,
            num_workers=train_config.get('num_workers', 4)
        )
        
        # 训练参数
        num_epochs = train_config.get('num_epochs', 50)
        early_stopping_patience = train_config.get('early_stopping_patience', 10)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # 训练一个epoch
            train_loss = self.trainer.train_epoch(train_loader, epoch)
            
            # 验证
            val_loss, val_auc = self.trainer.evaluate(val_loader)
            
            # 学习率调度
            self.scheduler.step(val_loss)
            
            # 保存训练历史
            self.train_history['train_loss'].append(train_loss)
            self.train_history['val_loss'].append(val_loss)
            self.train_history['val_auc'].append(val_auc)
            
            print(f'Epoch {epoch+1}/{num_epochs}:')
            print(f'  Train Loss: {train_loss:.4f}')
            print(f'  Val Loss: {val_loss:.4f}')
            print(f'  Val AUC: {val_auc:.4f}')
            
            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # 保存最佳模型
                self.save_model('best_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f'早停在epoch {epoch+1}')
                    break
        
        # 加载最佳模型
        self.load_model('best_model.pth')
        print("训练完成")
    
    def save_model(self, path: str):
        """保存模型"""
        self.trainer.save_model(path)
    
    def load_model(self, path: str):
        """加载模型"""
        self.trainer.load_model(path)
    
    def save_pipeline(self, path: str):
        """保存完整管道"""
        print(f"保存管道到: {path}")
        
        pipeline_data = {
            'config': self.config,
            'feature_columns': self.feature_columns,
            'input_dim': self.input_dim,
            'train_history': self.train_history
        }
        
        # 保存预处理器
        joblib.dump(self.preprocessor, f'{path}_preprocessor.pkl')
        joblib.dump(self.feature_engineer, f'{path}_feature_engineer.pkl')
        joblib.dump(self.scaler, f'{path}_scaler.pkl')
        
        # 保存管道数据
        with open(f'{path}_pipeline.pkl', 'wb') as f:
            pickle.dump(pipeline_data, f)
        
        print("管道保存完成")
    
    def load_pipeline(self, path: str):
        """加载完整管道"""
        print(f"加载管道从: {path}")
        
        # 加载预处理器
        self.preprocessor = joblib.load(f'{path}_preprocessor.pkl')
        self.feature_engineer = joblib.load(f'{path}_feature_engineer.pkl')
        self.scaler = joblib.load(f'{path}_scaler.pkl')
        
        # 加载管道数据
        with open(f'{path}_pipeline.pkl', 'rb') as f:
            pipeline_data = pickle.load(f)
        
        self.config = pipeline_data['config']
        self.feature_columns = pipeline_data['feature_columns']
        self.input_dim = pipeline_data['input_dim']
        self.train_history = pipeline_data['train_history']
        
        # 重建模型
        self.create_model()
        self.setup_training()
        
        print("管道加载完成")
    
    def plot_training_history(self):
        """绘制训练历史"""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib未安装，无法绘制训练历史图")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # 损失曲线
        ax1.plot(self.train_history['train_loss'], label='Train Loss')
        ax1.plot(self.train_history['val_loss'], label='Val Loss')
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # AUC曲线
        ax2.plot(self.train_history['val_auc'], label='Val AUC')
        ax2.set_title('Validation AUC')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('AUC')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.show()
    
    def run_full_pipeline(self):
        """运行完整管道"""
        print("开始运行完整训练管道...")
        
        # 数据预处理
        self.preprocess_data()
        
        # 特征工程
        self.feature_engineering()
        
        # 准备特征
        self.prepare_features()
        
        # 创建模型
        self.create_model()
        
        # 设置训练
        self.setup_training()
        
        # 训练模型
        self.train_model()
        
        # 保存管道
        self.save_pipeline('flight_ranking_pipeline')
        
        print("完整管道运行完成!")

def main():
    """主函数"""
    # 配置参数
    config = {
        'model': {
            'cross_layers': 3,
            'cross_low_rank': 32,
            'deep_layers': [512, 256, 128],
            'dropout_rate': 0.3,
            'use_bn': True
        },
        'training': {
            'batch_size': 256,
            'num_epochs': 50,
            'learning_rate': 0.001,
            'optimizer': 'adam',
            'weight_decay': 1e-5,
            'margin': 1.0,
            'patience': 5,
            'early_stopping_patience': 10,
            'num_workers': 4
        }
    }
    
    # 创建管道
    pipeline = FlightRankingPipeline(config)
    
    # 加载数据
    pipeline.load_data('train.parquet', 'test.parquet')
    
    # 运行完整管道
    pipeline.run_full_pipeline()
    
    # 绘制训练历史
    try:
        pipeline.plot_training_history()
    except ImportError:
        print("matplotlib未安装，跳过绘图")

if __name__ == "__main__":
    main() 