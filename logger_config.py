#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
日志配置模块

为整个航班排序项目提供统一的日志配置和管理功能。

使用方法:
    from logger_config import get_logger
    logger = get_logger(__name__)
    logger.info("这是一个信息日志")

作者: Assistant
日期: 2024
"""

import logging
import os
from datetime import datetime
from pathlib import Path

# 日志目录
LOG_DIR = Path('logs')
LOG_DIR.mkdir(exist_ok=True)

# 日志格式
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# 日志级别映射
LOG_LEVELS = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL
}

def setup_logger(name, log_file=None, level=logging.INFO, console=True):
    """
    设置日志器
    
    Args:
        name: 日志器名称
        log_file: 日志文件路径（可选）
        level: 日志级别
        console: 是否输出到控制台
    
    Returns:
        配置好的日志器
    """
    # 创建格式化器
    formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)
    
    # 创建日志器
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 清除已有的处理器
    logger.handlers.clear()
    
    # 控制台处理器
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # 文件处理器
    if log_file:
        # 确保日志目录存在
        log_path = LOG_DIR / log_file
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def get_logger(name, log_file=None, level='INFO', console=True):
    """
    获取日志器（推荐使用）
    
    Args:
        name: 日志器名称，通常使用 __name__
        log_file: 日志文件名（相对于logs目录）
        level: 日志级别（字符串）
        console: 是否输出到控制台
    
    Returns:
        配置好的日志器
    """
    log_level = LOG_LEVELS.get(level.upper(), logging.INFO)
    return setup_logger(name, log_file, log_level, console)

def get_module_logger(module_name, enable_file_logging=True):
    """
    为模块获取标准日志器
    
    Args:
        module_name: 模块名称
        enable_file_logging: 是否启用文件日志
    
    Returns:
        配置好的日志器
    """
    # 生成日志文件名
    timestamp = datetime.now().strftime('%Y%m%d')
    log_file = f'{module_name}_{timestamp}.log' if enable_file_logging else None
    
    return get_logger(module_name, log_file, level='INFO')

def configure_project_logging(level='INFO'):
    """
    配置整个项目的日志系统
    
    Args:
        level: 全局日志级别
    """
    # 设置根日志器
    root_logger = logging.getLogger()
    root_logger.setLevel(LOG_LEVELS.get(level.upper(), logging.INFO))
    
    # 清除默认处理器
    root_logger.handlers.clear()
    
    # 创建项目主日志文件
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    main_log_file = f'flight_rank_main_{timestamp}.log'
    
    # 设置主日志器
    main_logger = get_logger('FlightRank', main_log_file, level)
    main_logger.info("="*60)
    main_logger.info("航班排序项目日志系统启动")
    main_logger.info(f"日志级别: {level}")
    main_logger.info(f"日志目录: {LOG_DIR.absolute()}")
    main_logger.info("="*60)
    
    return main_logger

def cleanup_old_logs(days_to_keep=7):
    """
    清理旧的日志文件
    
    Args:
        days_to_keep: 保留的天数
    """
    if not LOG_DIR.exists():
        return
    
    cutoff_time = datetime.now().timestamp() - (days_to_keep * 24 * 3600)
    
    for log_file in LOG_DIR.glob('*.log'):
        if log_file.stat().st_mtime < cutoff_time:
            try:
                log_file.unlink()
                print(f"删除旧日志文件: {log_file.name}")
            except OSError as e:
                print(f"删除日志文件失败 {log_file.name}: {e}")

# 为主要模块提供快捷方式
def get_preprocessor_logger():
    """数据预处理模块日志器"""
    return get_module_logger('data_preprocessing')

def get_feature_engineer_logger():
    """特征工程模块日志器"""
    return get_module_logger('feature_engineering')

def get_model_logger():
    """模型训练模块日志器"""
    return get_module_logger('dcn_model')

def get_pipeline_logger():
    """训练管道日志器"""
    return get_module_logger('train_pipeline')

def get_predictor_logger():
    """预测模块日志器"""
    return get_module_logger('predict_pipeline')

def get_submission_logger():
    """提交文件生成日志器"""
    return get_module_logger('submission_generator')

if __name__ == "__main__":
    # 测试日志配置
    print("测试日志配置...")
    
    # 配置项目日志
    main_logger = configure_project_logging('INFO')
    
    # 测试各模块日志器
    loggers = [
        get_preprocessor_logger(),
        get_feature_engineer_logger(),
        get_model_logger(),
        get_pipeline_logger(),
        get_predictor_logger(),
        get_submission_logger()
    ]
    
    for logger in loggers:
        logger.info(f"测试日志器: {logger.name}")
        logger.debug("这是调试信息")
        logger.warning("这是警告信息")
    
    # 清理旧日志
    cleanup_old_logs(days_to_keep=7)
    
    print("日志配置测试完成！")
    print(f"日志文件保存在: {LOG_DIR.absolute()}") 