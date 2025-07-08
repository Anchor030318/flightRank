import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class SubmissionGenerator:
    """提交文件生成器"""
    
    def __init__(self):
        self.submission_format = None
        self.validation_errors = []
    
    def load_predictions(self, predictions_path: str) -> pd.DataFrame:
        """
        加载预测结果
        
        Args:
            predictions_path: 预测结果文件路径
            
        Returns:
            预测结果DataFrame
        """
        print(f"加载预测结果: {predictions_path}")
        
        if predictions_path.endswith('.csv'):
            df = pd.read_csv(predictions_path)
        elif predictions_path.endswith('.parquet'):
            df = pd.read_parquet(predictions_path)
        else:
            raise ValueError("不支持的文件格式，请使用CSV或Parquet格式")
        
        print(f"预测结果形状: {df.shape}")
        return df
    
    def validate_predictions(self, df: pd.DataFrame) -> bool:
        """
        验证预测结果的正确性
        
        Args:
            df: 预测结果DataFrame
            
        Returns:
            是否验证通过
        """
        print("验证预测结果...")
        self.validation_errors = []
        
        # 检查必要列
        required_columns = ['Id', 'ranker_id', 'predicted_score', 'rank']
        for col in required_columns:
            if col not in df.columns:
                self.validation_errors.append(f"缺少必要列: {col}")
        
        if self.validation_errors:
            return False
        
        # 检查数据类型
        if not pd.api.types.is_numeric_dtype(df['Id']):
            self.validation_errors.append("Id列必须是数值类型")
        
        if not pd.api.types.is_numeric_dtype(df['predicted_score']):
            self.validation_errors.append("predicted_score列必须是数值类型")
        
        if not pd.api.types.is_integer_dtype(df['rank']):
            self.validation_errors.append("rank列必须是整数类型")
        
        # 检查缺失值
        if df['Id'].isna().any():
            self.validation_errors.append("Id列存在缺失值")
        
        if df['ranker_id'].isna().any():
            self.validation_errors.append("ranker_id列存在缺失值")
        
        if df['rank'].isna().any():
            self.validation_errors.append("rank列存在缺失值")
        
        # 检查排名的有效性
        self._validate_rankings(df)
        
        if self.validation_errors:
            print("验证失败，错误信息:")
            for error in self.validation_errors:
                print(f"  - {error}")
            return False
        
        print("验证通过")
        return True
    
    def _validate_rankings(self, df: pd.DataFrame):
        """
        验证排名的正确性
        
        Args:
            df: 包含排名的DataFrame
        """
        print("验证排名正确性...")
        
        # 按组检查排名
        for ranker_id, group in df.groupby('ranker_id'):
            ranks = sorted(group['rank'].values)
            group_size = len(group)
            expected_ranks = list(range(1, group_size + 1))
            
            # 检查排名是否从1开始且连续
            if ranks != expected_ranks:
                self.validation_errors.append(
                    f"组 {ranker_id} 的排名不正确: 期望 {expected_ranks}, 实际 {ranks}"
                )
            
            # 检查排名是否唯一
            unique_ranks = len(set(group['rank'].values))
            if unique_ranks != group_size:
                self.validation_errors.append(
                    f"组 {ranker_id} 存在重复排名: {group_size} 个记录但只有 {unique_ranks} 个唯一排名"
                )
    
    def fix_rankings(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        修复排名问题
        
        Args:
            df: 包含排名的DataFrame
            
        Returns:
            修复后的DataFrame
        """
        print("修复排名问题...")
        
        def fix_group_ranking(group):
            """修复单个组的排名"""
            # 按预测分数降序排序
            group = group.sort_values('predicted_score', ascending=False)
            
            # 重新分配排名
            group['rank'] = range(1, len(group) + 1)
            
            return group
        
        # 按组修复排名
        df_fixed = df.groupby('ranker_id').apply(fix_group_ranking).reset_index(drop=True)
        
        print("排名修复完成")
        return df_fixed
    
    def create_submission_file(self, df: pd.DataFrame, output_path: str = 'submission.csv') -> pd.DataFrame:
        """
        创建提交文件
        
        Args:
            df: 包含排名的DataFrame
            output_path: 输出文件路径
            
        Returns:
            提交格式的DataFrame
        """
        print("创建提交文件...")
        
        # 创建提交格式
        submission = df[['Id', 'rank']].copy()
        
        # 按Id排序
        submission = submission.sort_values('Id')
        
        # 重置索引
        submission = submission.reset_index(drop=True)
        
        # 保存文件
        submission.to_csv(output_path, index=False)
        
        print(f"提交文件已保存: {output_path}")
        print(f"提交文件形状: {submission.shape}")
        
        return submission
    
    def generate_submission_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        生成提交报告
        
        Args:
            df: 包含排名的DataFrame
            
        Returns:
            提交报告字典
        """
        print("生成提交报告...")
        
        report = {
            'total_records': len(df),
            'total_groups': df['ranker_id'].nunique(),
            'avg_group_size': len(df) / df['ranker_id'].nunique(),
            'score_statistics': {
                'mean': df['predicted_score'].mean(),
                'std': df['predicted_score'].std(),
                'min': df['predicted_score'].min(),
                'max': df['predicted_score'].max(),
                'median': df['predicted_score'].median()
            },
            'rank_distribution': {},
            'group_size_distribution': {}
        }
        
        # 排名分布
        rank_counts = df['rank'].value_counts().sort_index()
        report['rank_distribution'] = {
            'rank_1_count': rank_counts.get(1, 0),
            'rank_2_count': rank_counts.get(2, 0),
            'rank_3_count': rank_counts.get(3, 0),
            'max_rank': df['rank'].max(),
            'min_rank': df['rank'].min()
        }
        
        # 组大小分布
        group_sizes = df.groupby('ranker_id').size()
        report['group_size_distribution'] = {
            'min_group_size': group_sizes.min(),
            'max_group_size': group_sizes.max(),
            'avg_group_size': group_sizes.mean(),
            'group_size_counts': group_sizes.value_counts().to_dict()
        }
        
        return report
    
    def print_submission_report(self, report: Dict[str, Any]):
        """
        打印提交报告
        
        Args:
            report: 提交报告字典
        """
        print("\n" + "="*50)
        print("提交文件报告")
        print("="*50)
        
        print(f"总记录数: {report['total_records']:,}")
        print(f"总组数: {report['total_groups']:,}")
        print(f"平均组大小: {report['avg_group_size']:.2f}")
        
        print("\n--- 分数统计 ---")
        stats = report['score_statistics']
        print(f"平均分数: {stats['mean']:.4f}")
        print(f"标准差: {stats['std']:.4f}")
        print(f"最小值: {stats['min']:.4f}")
        print(f"最大值: {stats['max']:.4f}")
        print(f"中位数: {stats['median']:.4f}")
        
        print("\n--- 排名分布 ---")
        rank_dist = report['rank_distribution']
        print(f"排名1的数量: {rank_dist['rank_1_count']:,}")
        print(f"排名2的数量: {rank_dist['rank_2_count']:,}")
        print(f"排名3的数量: {rank_dist['rank_3_count']:,}")
        print(f"最大排名: {rank_dist['max_rank']}")
        print(f"最小排名: {rank_dist['min_rank']}")
        
        print("\n--- 组大小分布 ---")
        group_dist = report['group_size_distribution']
        print(f"最小组大小: {group_dist['min_group_size']}")
        print(f"最大组大小: {group_dist['max_group_size']}")
        print(f"平均组大小: {group_dist['avg_group_size']:.2f}")
        
        print("\n组大小计数:")
        for size, count in sorted(group_dist['group_size_counts'].items()):
            print(f"  {size}个航班的组: {count:,} 个")
        
        print("="*50)
    
    def run_full_generation(self, predictions_path: str, output_path: str = 'submission.csv') -> pd.DataFrame:
        """
        运行完整的提交文件生成流程
        
        Args:
            predictions_path: 预测结果文件路径
            output_path: 输出文件路径
            
        Returns:
            提交格式的DataFrame
        """
        print("开始提交文件生成流程...")
        
        # 加载预测结果
        df = self.load_predictions(predictions_path)
        
        # 验证预测结果
        if not self.validate_predictions(df):
            print("预测结果验证失败，尝试修复...")
            df = self.fix_rankings(df)
            
            # 再次验证
            if not self.validate_predictions(df):
                raise ValueError("无法修复预测结果中的错误")
        
        # 创建提交文件
        submission = self.create_submission_file(df, output_path)
        
        # 生成报告
        report = self.generate_submission_report(df)
        self.print_submission_report(report)
        
        # 保存详细结果
        detailed_output = output_path.replace('.csv', '_detailed.csv')
        df.to_csv(detailed_output, index=False)
        print(f"详细结果已保存: {detailed_output}")
        
        print("提交文件生成完成！")
        return submission

class SubmissionValidator:
    """提交文件验证器"""
    
    def __init__(self):
        pass
    
    def validate_submission_format(self, submission_path: str) -> bool:
        """
        验证提交文件格式
        
        Args:
            submission_path: 提交文件路径
            
        Returns:
            是否验证通过
        """
        print(f"验证提交文件格式: {submission_path}")
        
        try:
            # 读取提交文件
            df = pd.read_csv(submission_path)
            
            # 检查列名
            if list(df.columns) != ['Id', 'rank']:
                print("错误: 提交文件必须包含且仅包含 'Id' 和 'rank' 两列")
                return False
            
            # 检查数据类型
            if not pd.api.types.is_integer_dtype(df['Id']):
                print("错误: Id列必须是整数类型")
                return False
            
            if not pd.api.types.is_integer_dtype(df['rank']):
                print("错误: rank列必须是整数类型")
                return False
            
            # 检查缺失值
            if df.isna().any().any():
                print("错误: 提交文件不能包含缺失值")
                return False
            
            # 检查排名范围
            if df['rank'].min() < 1:
                print("错误: 排名必须从1开始")
                return False
            
            print("提交文件格式验证通过")
            return True
            
        except Exception as e:
            print(f"验证过程中出错: {e}")
            return False
    
    def validate_with_test_data(self, submission_path: str, test_data_path: str) -> bool:
        """
        与测试数据进行交叉验证
        
        Args:
            submission_path: 提交文件路径
            test_data_path: 测试数据路径
            
        Returns:
            是否验证通过
        """
        print("与测试数据进行交叉验证...")
        
        try:
            # 读取文件
            submission = pd.read_csv(submission_path)
            test_data = pd.read_parquet(test_data_path)
            
            # 检查Id是否匹配
            test_ids = set(test_data['Id'])
            submission_ids = set(submission['Id'])
            
            if test_ids != submission_ids:
                missing_ids = test_ids - submission_ids
                extra_ids = submission_ids - test_ids
                
                if missing_ids:
                    print(f"错误: 缺少 {len(missing_ids)} 个Id")
                
                if extra_ids:
                    print(f"错误: 多出 {len(extra_ids)} 个Id")
                
                return False
            
            # 检查每个组的排名
            test_groups = test_data.groupby('ranker_id')['Id'].apply(set).to_dict()
            submission_dict = submission.set_index('Id')['rank'].to_dict()
            
            for ranker_id, group_ids in test_groups.items():
                group_ranks = [submission_dict[id_] for id_ in group_ids]
                expected_ranks = set(range(1, len(group_ids) + 1))
                actual_ranks = set(group_ranks)
                
                if expected_ranks != actual_ranks:
                    print(f"错误: 组 {ranker_id} 的排名不正确")
                    return False
            
            print("交叉验证通过")
            return True
            
        except Exception as e:
            print(f"交叉验证过程中出错: {e}")
            return False

def main():
    """主函数"""
    print("启动提交文件生成器...")
    
    # 配置参数
    predictions_path = 'prediction_results.csv'
    output_path = 'submission.csv'
    test_data_path = 'test.parquet'
    
    # 创建生成器
    generator = SubmissionGenerator()
    
    # 运行完整流程
    try:
        submission = generator.run_full_generation(predictions_path, output_path)
        
        # 验证提交文件
        validator = SubmissionValidator()
        
        # 格式验证
        if validator.validate_submission_format(output_path):
            print("✓ 提交文件格式验证通过")
        else:
            print("✗ 提交文件格式验证失败")
            return
        
        # 与测试数据交叉验证
        if validator.validate_with_test_data(output_path, test_data_path):
            print("✓ 与测试数据交叉验证通过")
        else:
            print("✗ 与测试数据交叉验证失败")
            return
        
        print("\n🎉 提交文件生成并验证成功！")
        print(f"最终提交文件: {output_path}")
        
    except Exception as e:
        print(f"生成过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 