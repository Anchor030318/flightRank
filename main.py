#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
航班排序DCN-v2模型 - 主入口文件

这是一个完整的航班排序解决方案，使用DCN-v2（Deep Cross Network v2）模型。

使用方法:
    python main.py --mode train    # 训练模式
    python main.py --mode predict  # 预测模式
    python main.py --mode all      # 完整流程（训练+预测）

作者: Assistant
日期: 2024
"""

import argparse
import os
import sys
import time
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from train_pipeline import FlightRankingPipeline
from predict_pipeline import FlightPredictor
from submission_generator import SubmissionGenerator, SubmissionValidator

def check_files():
    """检查必要文件是否存在"""
    required_files = {
        'data/train.parquet': '训练数据',
        'data/test.parquet': '测试数据'
    }
    
    missing_files = []
    for file_path, description in required_files.items():
        if not os.path.exists(file_path):
            missing_files.append(f"{file_path} ({description})")
    
    if missing_files:
        print("错误: 缺少以下必要文件:")
        for file in missing_files:
            print(f"  - {file}")
        print("\n请确保训练和测试数据文件在当前目录下。")
        return False
    
    return True

def train_model():
    """训练模型"""
    print("="*60)
    print("开始训练DCN-v2航班排序模型")
    print("="*60)
    
    start_time = time.time()
    
    # 配置参数
    config = {
        'model': {
            'cross_layers': 3,              # 交叉网络层数
            'cross_low_rank': 32,           # 低秩分解维度
            'deep_layers': [512, 256, 128], # 深度网络层数
            'dropout_rate': 0.3,            # Dropout比例
            'use_bn': True                  # 是否使用批归一化
        },
        'training': {
            'batch_size': 256,              # 批次大小
            'num_epochs': 50,               # 训练轮数
            'learning_rate': 0.001,         # 学习率
            'optimizer': 'adam',            # 优化器
            'weight_decay': 1e-5,           # 权重衰减
            'margin': 1.0,                  # 排序损失的边际
            'patience': 5,                  # 学习率调度耐心
            'early_stopping_patience': 10, # 早停耐心
            'num_workers': 4                # 数据加载进程数
        }
    }
    
    try:
        # 创建训练管道，添加日志文件
        pipeline = FlightRankingPipeline(config, log_file='train_pipeline.log')
        
        # 加载数据
        pipeline.load_data('data/train.parquet', 'data/test.parquet')
        
        # 运行完整训练流程
        pipeline.run_full_pipeline()
        
        training_time = time.time() - start_time
        print(f"\n✅ 训练完成! 耗时: {training_time/60:.2f} 分钟")
        
        # 绘制训练历史（如果可能）
        try:
            pipeline.plot_training_history()
            print("📊 训练历史图已保存为 training_history.png")
        except ImportError:
            print("⚠️  matplotlib未安装，跳过训练历史绘图")
        
        return True
        
    except Exception as e:
        print(f"❌ 训练过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return False

def predict_and_generate_submission():
    """预测并生成提交文件"""
    print("="*60)
    print("开始预测并生成提交文件")
    print("="*60)
    
    start_time = time.time()
    
    try:
        # 检查是否存在训练好的模型
        pipeline_path = 'flight_ranking_pipeline'
        required_model_files = [
            f'{pipeline_path}_preprocessor.pkl',
            f'{pipeline_path}_feature_engineer.pkl',
            f'{pipeline_path}_scaler.pkl',
            f'{pipeline_path}_pipeline.pkl',
            'best_model.pth'
        ]
        
        missing_model_files = [f for f in required_model_files if not os.path.exists(f)]
        if missing_model_files:
            print("❌ 未找到训练好的模型文件，请先运行训练:")
            for f in missing_model_files:
                print(f"  - {f}")
            return False
        
        # 创建预测器
        print("📂 加载训练好的模型...")
        predictor = FlightPredictor(pipeline_path)
        
        # 加载测试数据
        print("📊 加载测试数据...")
        import pandas as pd
        test_data = pd.read_parquet('data/test.parquet')
        print(f"测试数据形状: {test_data.shape}")
        
        # 预测
        print("🔮 开始预测...")
        if len(test_data) > 100000:
            # 大数据集使用批量预测
            results = predictor.batch_predict(test_data, batch_size=50000)
        else:
            # 小数据集直接预测
            results = predictor.predict_ranking(test_data)
        
        # 保存预测结果
        results.to_csv('prediction_results.csv', index=False)
        print("💾 预测结果已保存到: prediction_results.csv")
        
        # 生成提交文件
        print("📝 生成提交文件...")
        generator = SubmissionGenerator()
        submission = generator.run_full_generation('prediction_results.csv', 'submission.csv')
        
        # 验证提交文件
        print("🔍 验证提交文件...")
        validator = SubmissionValidator()
        
        format_ok = validator.validate_submission_format('submission.csv')
        cross_validation_ok = validator.validate_with_test_data('submission.csv', 'data/test.parquet')
        
        if format_ok and cross_validation_ok:
            prediction_time = time.time() - start_time
            print(f"\n✅ 预测和提交文件生成完成! 耗时: {prediction_time/60:.2f} 分钟")
            print("📄 最终提交文件: submission.csv")
            return True
        else:
            print("❌ 提交文件验证失败")
            return False
            
    except Exception as e:
        print(f"❌ 预测过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='航班排序DCN-v2模型')
    parser.add_argument('--mode', choices=['train', 'predict', 'all'], 
                       default='all', help='运行模式: train(训练), predict(预测), all(完整流程)')
    
    args = parser.parse_args()
    
    print("🛫 航班排序DCN-v2模型")
    print("="*60)
    
    # 检查必要文件
    if not check_files():
        sys.exit(1)
    
    success = True
    
    if args.mode in ['train', 'all']:
        success = train_model()
        if not success:
            sys.exit(1)
    
    if args.mode in ['predict', 'all']:
        success = predict_and_generate_submission()
        if not success:
            sys.exit(1)
    
    if success:
        print("\n" + "="*60)
        print("🎉 任务完成!")
        print("="*60)
        
        if args.mode in ['predict', 'all']:
            print("📁 生成的文件:")
            print("  - prediction_results.csv (详细预测结果)")
            print("  - submission.csv (最终提交文件)")
            print("  - submission_detailed.csv (详细提交结果)")
            
        if args.mode in ['train', 'all']:
            print("📁 模型文件:")
            print("  - best_model.pth (最佳模型)")
            print("  - flight_ranking_pipeline_*.pkl (预处理器)")
            
        print("\n💡 使用建议:")
        print("  1. 检查training_history.png了解训练过程")
        print("  2. 查看submission.csv确认提交格式")
        print("  3. 可以调整config参数重新训练以提高性能")

if __name__ == "__main__":
    main() 