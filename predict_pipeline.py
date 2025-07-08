import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Any
import joblib
import pickle
import warnings
warnings.filterwarnings('ignore')

# 导入训练管道中的组件
from train_pipeline import FlightDataset, FlightRankingPipeline

class FlightPredictor:
    """航班排序预测器"""
    
    def __init__(self, pipeline_path: str):
        """
        初始化预测器
        
        Args:
            pipeline_path: 训练管道的保存路径
        """
        self.pipeline_path = pipeline_path
        self.pipeline = FlightRankingPipeline({})
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 加载训练好的管道
        self.load_pipeline()
    
    def load_pipeline(self):
        """加载训练好的管道"""
        print("加载训练好的管道...")
        
        # 加载管道
        self.pipeline.load_pipeline(self.pipeline_path)
        
        # 加载最佳模型
        self.pipeline.load_model('best_model.pth')
        
        print("管道加载完成")
    
    def preprocess_test_data(self, test_data: pd.DataFrame) -> np.ndarray:
        """
        预处理测试数据
        
        Args:
            test_data: 测试数据DataFrame
            
        Returns:
            预处理后的特征数组
        """
        print("预处理测试数据...")
        
        # 数据预处理
        test_processed = self.pipeline.preprocessor.transform(test_data)
        
        # 特征工程
        test_featured = self.pipeline.feature_engineer.transform(test_processed)
        
        # 获取特征
        X_test = test_featured[self.pipeline.feature_columns].values
        
        # 标准化
        X_test = self.pipeline.scaler.transform(X_test)
        
        print(f"测试数据预处理完成: {X_test.shape}")
        return X_test
    
    def predict_scores(self, X_test: np.ndarray, batch_size: int = 1024) -> np.ndarray:
        """
        预测分数
        
        Args:
            X_test: 测试数据特征
            batch_size: 批次大小
            
        Returns:
            预测分数数组
        """
        print("预测分数...")
        
        # 创建数据集
        test_dataset = FlightDataset(X_test, np.zeros(len(X_test)))
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # 预测
        self.pipeline.model.eval()
        predictions = []
        
        with torch.no_grad():
            for batch in test_loader:
                features = batch['features'].to(self.device)
                outputs = self.pipeline.model(features)
                scores = torch.sigmoid(outputs).cpu().numpy()
                predictions.extend(scores)
        
        predictions = np.array(predictions)
        print(f"预测完成: {predictions.shape}")
        
        return predictions
    
    def predict_ranking(self, test_data: pd.DataFrame, batch_size: int = 1024) -> pd.DataFrame:
        """
        预测排名
        
        Args:
            test_data: 测试数据
            batch_size: 批次大小
            
        Returns:
            包含预测分数和排名的DataFrame
        """
        print("开始预测排名...")
        
        # 预处理数据
        X_test = self.preprocess_test_data(test_data)
        
        # 预测分数
        scores = self.predict_scores(X_test, batch_size)
        
        # 创建结果DataFrame
        result_df = test_data[['Id', 'ranker_id']].copy()
        result_df['predicted_score'] = scores
        
        # 基于ranker_id分组进行排名
        result_df['rank'] = result_df.groupby('ranker_id')['predicted_score'].rank(
            method='first', ascending=False
        ).astype(int)
        
        print(f"排名预测完成: {result_df.shape}")
        return result_df
    
    def batch_predict(self, test_data: pd.DataFrame, batch_size: int = 10000) -> pd.DataFrame:
        """
        批量预测（适用于大数据集）
        
        Args:
            test_data: 测试数据
            batch_size: 批次大小
            
        Returns:
            预测结果DataFrame
        """
        print(f"开始批量预测，数据量: {len(test_data)}")
        
        results = []
        num_batches = len(test_data) // batch_size + (1 if len(test_data) % batch_size != 0 else 0)
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(test_data))
            
            print(f"处理批次 {i+1}/{num_batches}: {start_idx}-{end_idx}")
            
            batch_data = test_data.iloc[start_idx:end_idx]
            batch_result = self.predict_ranking(batch_data)
            results.append(batch_result)
        
        # 合并结果
        final_result = pd.concat(results, ignore_index=True)
        print(f"批量预测完成: {final_result.shape}")
        
        return final_result

class RankingPostProcessor:
    """排名后处理器"""
    
    def __init__(self):
        pass
    
    def ensure_unique_ranks(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        确保同一组内的排名唯一
        
        Args:
            df: 包含排名的DataFrame
            
        Returns:
            处理后的DataFrame
        """
        print("确保排名唯一性...")
        
        def fix_group_ranks(group):
            """修复组内排名"""
            # 按预测分数排序
            group = group.sort_values('predicted_score', ascending=False)
            
            # 重新分配排名
            group['rank'] = range(1, len(group) + 1)
            
            return group
        
        # 按组修复排名
        df_fixed = df.groupby('ranker_id').apply(fix_group_ranks).reset_index(drop=True)
        
        # 验证排名唯一性
        self.validate_rankings(df_fixed)
        
        print("排名唯一性检查完成")
        return df_fixed
    
    def validate_rankings(self, df: pd.DataFrame):
        """
        验证排名的正确性
        
        Args:
            df: 包含排名的DataFrame
        """
        print("验证排名正确性...")
        
        # 检查每个组的排名是否唯一
        for ranker_id, group in df.groupby('ranker_id'):
            ranks = group['rank'].values
            unique_ranks = np.unique(ranks)
            
            if len(ranks) != len(unique_ranks):
                print(f"警告: 组 {ranker_id} 存在重复排名!")
                print(f"排名: {sorted(ranks)}")
                print(f"唯一排名: {sorted(unique_ranks)}")
            
            # 检查排名是否连续
            expected_ranks = set(range(1, len(ranks) + 1))
            actual_ranks = set(ranks)
            
            if expected_ranks != actual_ranks:
                print(f"警告: 组 {ranker_id} 排名不连续!")
                print(f"期望排名: {sorted(expected_ranks)}")
                print(f"实际排名: {sorted(actual_ranks)}")
        
        # 整体统计
        total_groups = df['ranker_id'].nunique()
        print(f"总组数: {total_groups}")
        print(f"总记录数: {len(df)}")
        print(f"平均每组记录数: {len(df) / total_groups:.2f}")
    
    def create_submission_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        创建提交格式的数据
        
        Args:
            df: 包含排名的DataFrame
            
        Returns:
            提交格式的DataFrame
        """
        print("创建提交格式...")
        
        # 创建提交格式
        submission = df[['Id', 'rank']].copy()
        submission = submission.sort_values('Id')
        
        print(f"提交格式创建完成: {submission.shape}")
        return submission

def main():
    """主函数"""
    # 配置参数
    pipeline_path = 'flight_ranking_pipeline'
    test_data_path = 'test.parquet'
    
    print("开始预测流程...")
    
    # 创建预测器
    predictor = FlightPredictor(pipeline_path)
    
    # 加载测试数据
    print("加载测试数据...")
    test_data = pd.read_parquet(test_data_path)
    print(f"测试数据形状: {test_data.shape}")
    
    # 预测排名
    if len(test_data) > 100000:
        # 大数据集使用批量预测
        results = predictor.batch_predict(test_data, batch_size=50000)
    else:
        # 小数据集直接预测
        results = predictor.predict_ranking(test_data)
    
    # 后处理
    post_processor = RankingPostProcessor()
    
    # 确保排名唯一
    results = post_processor.ensure_unique_ranks(results)
    
    # 创建提交格式
    submission = post_processor.create_submission_format(results)
    
    # 保存结果
    results.to_csv('prediction_results.csv', index=False)
    submission.to_csv('submission.csv', index=False)
    
    print("预测完成！")
    print(f"详细结果保存到: prediction_results.csv")
    print(f"提交文件保存到: submission.csv")
    
    # 打印统计信息
    print("\n=== 预测统计 ===")
    print(f"总记录数: {len(results)}")
    print(f"总组数: {results['ranker_id'].nunique()}")
    print(f"平均分数: {results['predicted_score'].mean():.4f}")
    print(f"分数标准差: {results['predicted_score'].std():.4f}")
    print(f"最高分数: {results['predicted_score'].max():.4f}")
    print(f"最低分数: {results['predicted_score'].min():.4f}")
    
    # 查看每组的排名分布
    print("\n=== 排名分布 ===")
    rank_dist = results.groupby('ranker_id')['rank'].agg(['count', 'min', 'max'])
    print(f"组大小统计:")
    print(rank_dist['count'].describe())

if __name__ == "__main__":
    main() 