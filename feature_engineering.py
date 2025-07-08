import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 使用统一的日志配置
from logger_config import get_logger

class FeatureEngineer:
    def __init__(self, log_file=None):
        self.group_stats = {}
        self.price_stats = {}
        # 使用统一的日志配置系统
        self.logger = get_logger('FeatureEngineer', log_file or 'feature_engineering.log')
        
    def create_flight_complexity_features(self, df):
        """创建航班复杂度特征"""
        self.logger.info("创建航班复杂度特征...")
        
        # 总航段数
        df['total_segments'] = df['legs0_segment_count'] + df['legs1_segment_count']
        
        # 转机次数
        df['total_transfers'] = df['total_segments'] - 2
        df['total_transfers'] = df['total_transfers'].clip(lower=0)
        
        # 是否直飞
        df['is_direct_flight'] = (df['total_segments'] == 2).astype(int)
        
        # 是否单程
        df['is_one_way'] = (df['legs1_segment_count'] == 0).astype(int)
        
        self.logger.debug(f"航班复杂度特征创建完成，新增特征: total_segments, total_transfers, is_direct_flight, is_one_way")
        return df
    
    def create_time_features(self, df):
        """创建时间相关特征"""
        self.logger.info("创建时间相关特征...")
        
        # 计算总旅行时间
        if 'legs0_duration_minutes' in df.columns and 'legs1_duration_minutes' in df.columns:
            df['total_travel_time'] = df['legs0_duration_minutes'] + df['legs1_duration_minutes']
        
        # 出发时间偏好特征
        if 'legs0_departureAt_hour' in df.columns:
            # 是否早班机（6-10点）
            df['is_early_morning'] = ((df['legs0_departureAt_hour'] >= 6) & 
                                     (df['legs0_departureAt_hour'] <= 10)).astype(int)
            
            # 是否晚班机（20-24点）
            df['is_late_night'] = ((df['legs0_departureAt_hour'] >= 20) & 
                                  (df['legs0_departureAt_hour'] <= 24)).astype(int)
            
            # 是否黄金时段（10-18点）
            df['is_golden_time'] = ((df['legs0_departureAt_hour'] >= 10) & 
                                   (df['legs0_departureAt_hour'] <= 18)).astype(int)
        
        # 是否周末出发
        if 'legs0_departureAt_day_of_week' in df.columns:
            df['is_weekend_departure'] = (df['legs0_departureAt_day_of_week'] >= 5).astype(int)
        
        # 预订提前天数
        if 'requestDate_day' in df.columns and 'legs0_departureAt_day' in df.columns:
            df['booking_advance_days'] = df['legs0_departureAt_day'] - df['requestDate_day']
            df['booking_advance_days'] = df['booking_advance_days'].clip(lower=0)
        
        self.logger.debug("时间相关特征创建完成")
        return df
    
    def create_price_features(self, df):
        """创建价格相关特征"""
        self.logger.info("创建价格相关特征...")
        
        # 每小时价格
        if 'totalPrice' in df.columns and 'total_travel_time' in df.columns:
            df['price_per_hour'] = df['totalPrice'] / (df['total_travel_time'] + 1e-8)
        
        # 价格分级
        if 'totalPrice' in df.columns:
            df['price_category'] = pd.cut(df['totalPrice'], 
                                        bins=[0, 10000, 30000, 50000, 100000, np.inf],
                                        labels=[0, 1, 2, 3, 4]).astype(int)
        
        # 税费比例分级
        if 'tax_ratio' in df.columns:
            df['tax_category'] = pd.cut(df['tax_ratio'], 
                                      bins=[0, 0.1, 0.2, 0.3, np.inf],
                                      labels=[0, 1, 2, 3]).astype(int)
        
        self.logger.debug("价格相关特征创建完成")
        return df
    
    def create_carrier_features(self, df):
        """创建航空公司特征"""
        self.logger.info("创建航空公司特征...")
        
        # 主要航空公司
        main_carriers = []
        for leg in ['legs0', 'legs1']:
            for seg in ['segments0', 'segments1', 'segments2', 'segments3']:
                col = f'{leg}_{seg}_marketingCarrier_code'
                if col in df.columns:
                    main_carriers.append(col)
        
        if main_carriers:
            # 航空公司数量
            carrier_cols = [col for col in main_carriers if col in df.columns]
            if carrier_cols:
                # 统计不同航空公司数量
                carrier_df = df[carrier_cols].fillna(-1)
                df['unique_carriers'] = carrier_df.nunique(axis=1)
                
                # 是否代码共享
                df['is_codeshare'] = (df['unique_carriers'] > 1).astype(int)
        
        self.logger.debug("航空公司特征创建完成")
        return df
    
    def create_aircraft_features(self, df):
        """创建飞机型号特征"""
        self.logger.info("创建飞机型号特征...")
        
        # 飞机型号特征
        aircraft_cols = []
        for leg in ['legs0', 'legs1']:
            for seg in ['segments0', 'segments1', 'segments2', 'segments3']:
                col = f'{leg}_{seg}_aircraft_code'
                if col in df.columns:
                    aircraft_cols.append(col)
        
        if aircraft_cols:
            # 统计不同飞机型号数量
            aircraft_df = df[aircraft_cols].fillna(-1)
            df['unique_aircraft'] = aircraft_df.nunique(axis=1)
            
            # 是否更换机型
            df['aircraft_change'] = (df['unique_aircraft'] > 1).astype(int)
        
        self.logger.debug("飞机型号特征创建完成")
        return df
    
    def create_seat_features_group_wise(self, df):
        """创建组内座位特征比较"""
        self.logger.info("创建组内座位特征比较...")
        
        # 座位相关特征
        seat_cols = []
        for leg in ['legs0', 'legs1']:
            for seg in ['segments0', 'segments1', 'segments2', 'segments3']:
                col = f'{leg}_{seg}_seatsAvailable'
                if col in df.columns:
                    seat_cols.append(col)
        
        if seat_cols and 'ranker_id' in df.columns:
            # 计算每个航班的总可用座位数
            df['total_seats_available'] = df[seat_cols].sum(axis=1)
            df['min_seats_available'] = df[seat_cols].min(axis=1)
            df['avg_seats_available'] = df[seat_cols].mean(axis=1)
            
            # 组内座位排名特征
            df['seats_rank_in_group'] = df.groupby('ranker_id')['total_seats_available'].rank(
                method='dense', ascending=False
            )
            
            # 相对于组内平均值的座位数
            group_avg_seats = df.groupby('ranker_id')['total_seats_available'].transform('mean')
            df['seats_vs_group_avg'] = df['total_seats_available'] / (group_avg_seats + 1e-8)
            
            # 是否是组内座位最多的航班
            group_max_seats = df.groupby('ranker_id')['total_seats_available'].transform('max')
            df['is_max_seats_in_group'] = (df['total_seats_available'] == group_max_seats).astype(int)
            
        self.logger.debug("组内座位特征比较创建完成")
        return df
    
    def create_baggage_features_group_wise(self, df):
        """创建组内行李特征比较"""
        self.logger.info("创建组内行李特征比较...")
        
        # 行李额度特征
        baggage_cols = []
        for leg in ['legs0', 'legs1']:
            for seg in ['segments0', 'segments1', 'segments2', 'segments3']:
                col = f'{leg}_{seg}_baggageAllowance_quantity'
                if col in df.columns:
                    baggage_cols.append(col)
        
        if baggage_cols and 'ranker_id' in df.columns:
            # 计算总行李额度
            df['total_baggage_allowance'] = df[baggage_cols].sum(axis=1)
            df['min_baggage_allowance'] = df[baggage_cols].min(axis=1)
            df['has_free_baggage'] = (df['total_baggage_allowance'] > 0).astype(int)
            
            # 组内行李排名特征
            df['baggage_rank_in_group'] = df.groupby('ranker_id')['total_baggage_allowance'].rank(
                method='dense', ascending=False
            )
            
            # 相对于组内平均值的行李额度
            group_avg_baggage = df.groupby('ranker_id')['total_baggage_allowance'].transform('mean')
            df['baggage_vs_group_avg'] = df['total_baggage_allowance'] / (group_avg_baggage + 1e-8)
            
            # 是否是组内行李额度最高的航班
            group_max_baggage = df.groupby('ranker_id')['total_baggage_allowance'].transform('max')
            df['is_max_baggage_in_group'] = (df['total_baggage_allowance'] == group_max_baggage).astype(int)
        
        self.logger.debug("组内行李特征比较创建完成")
        return df
    
    def create_cabin_features_group_wise(self, df):
        """创建组内舱位特征比较"""
        self.logger.info("创建组内舱位特征比较...")
        
        # 舱位等级特征
        cabin_cols = []
        for leg in ['legs0', 'legs1']:
            for seg in ['segments0', 'segments1', 'segments2', 'segments3']:
                col = f'{leg}_{seg}_cabinClass'
                if col in df.columns:
                    cabin_cols.append(col)
        
        if cabin_cols and 'ranker_id' in df.columns:
            # 计算舱位等级
            df['avg_cabin_class'] = df[cabin_cols].mean(axis=1)
            df['max_cabin_class'] = df[cabin_cols].max(axis=1)
            df['mixed_cabin'] = (df[cabin_cols].nunique(axis=1) > 1).astype(int)
            
            # 组内舱位排名特征（舱位等级越高越好，通常数字越大等级越高）
            df['cabin_rank_in_group'] = df.groupby('ranker_id')['avg_cabin_class'].rank(
                method='dense', ascending=False
            )
            
            # 相对于组内平均舱位等级
            group_avg_cabin = df.groupby('ranker_id')['avg_cabin_class'].transform('mean')
            df['cabin_vs_group_avg'] = df['avg_cabin_class'] / (group_avg_cabin + 1e-8)
            
            # 是否是组内舱位等级最高的航班
            group_max_cabin = df.groupby('ranker_id')['avg_cabin_class'].transform('max')
            df['is_max_cabin_in_group'] = (df['avg_cabin_class'] == group_max_cabin).astype(int)
        
        self.logger.debug("组内舱位特征比较创建完成")
        return df
    
    def create_route_features(self, df):
        """创建航线特征"""
        self.logger.info("创建航线特征...")
        
        # 航线复杂度
        if 'searchRoute' in df.columns:
            # 航线长度（字符数）
            df['route_length'] = df['searchRoute'].str.len()
            
            # 是否往返
            df['is_roundtrip'] = df['searchRoute'].str.contains('/', na=False).astype(int)
            
            # 航线中的城市数量
            df['cities_count'] = df['searchRoute'].str.count('[A-Z]{3}')
        
        self.logger.debug("航线特征创建完成")
        return df
    
    def create_user_features(self, df):
        """创建用户特征"""
        self.logger.info("创建用户特征...")
        
        # 用户类型特征
        user_features = []
        
        # VIP用户
        if 'isVip' in df.columns:
            user_features.append('isVip')
        
        # 3D访问权限
        if 'isAccess3D' in df.columns:
            user_features.append('isAccess3D')
        
        # 自助预订
        if 'bySelf' in df.columns:
            user_features.append('bySelf')
        
        # 创建用户档案评分
        if user_features:
            df['user_profile_score'] = df[user_features].sum(axis=1)
        
        # 常旅客等级
        if 'frequentFlyer' in df.columns:
            df['is_frequent_flyer'] = (df['frequentFlyer'] != 0).astype(int)
        
        self.logger.debug("用户特征创建完成")
        return df
    
    def create_group_features(self, df, fit_mode=True):
        """创建群组特征"""
        self.logger.info("创建群组特征...")
        
        # 基于ranker_id的群组特征
        if 'ranker_id' in df.columns and 'totalPrice' in df.columns:
            if fit_mode:
                # 计算群组统计
                group_stats = df.groupby('ranker_id')['totalPrice'].agg(['mean', 'std', 'count']).reset_index()
                group_stats.columns = ['ranker_id', 'group_price_mean', 'group_price_std', 'group_size']
                self.group_stats = group_stats
                self.logger.info(f"计算了 {len(group_stats)} 个群组的价格统计")
            else:
                # 使用已有统计
                group_stats = self.group_stats
                self.logger.info("使用已有的群组价格统计")
            
            # 合并群组特征
            df = df.merge(group_stats, on='ranker_id', how='left')
            
            # 填充缺失值
            df['group_price_mean'] = df['group_price_mean'].fillna(df['totalPrice'].mean())
            df['group_price_std'] = df['group_price_std'].fillna(df['totalPrice'].std())
            df['group_size'] = df['group_size'].fillna(1)
            
            # 创建相对价格特征
            df['price_vs_group_mean'] = df['totalPrice'] / (df['group_price_mean'] + 1e-8)
            df['price_rank_in_group'] = df.groupby('ranker_id')['totalPrice'].rank(pct=True)
        
        self.logger.debug("群组特征创建完成")
        return df
    
    def create_interaction_features(self, df):
        """创建交互特征"""
        self.logger.info("创建交互特征...")
        
        # 价格和时间的交互
        if 'totalPrice' in df.columns and 'total_travel_time' in df.columns:
            df['price_time_interaction'] = df['totalPrice'] * df['total_travel_time']
        
        # 用户类型和价格的交互
        if 'isVip' in df.columns and 'totalPrice' in df.columns:
            df['vip_price_interaction'] = df['isVip'] * df['totalPrice']
        
        # 转机次数和价格的交互
        if 'total_transfers' in df.columns and 'totalPrice' in df.columns:
            df['transfer_price_interaction'] = df['total_transfers'] * df['totalPrice']
        
        # 组内排名特征的交互
        if 'price_rank_in_group' in df.columns and 'cabin_rank_in_group' in df.columns:
            df['price_cabin_rank_interaction'] = df['price_rank_in_group'] * df['cabin_rank_in_group']
        
        self.logger.debug("交互特征创建完成")
        return df
    
    def fit_transform(self, df):
        """训练时的特征工程"""
        self.logger.info("="*50)
        self.logger.info("开始特征工程（训练模式）...")
        self.logger.info(f"输入数据形状: {df.shape}")
        
        # 复制数据
        df = df.copy()
        
        # 创建各类特征
        df = self.create_flight_complexity_features(df)
        df = self.create_time_features(df)
        df = self.create_price_features(df)
        df = self.create_carrier_features(df)
        df = self.create_aircraft_features(df)
        df = self.create_seat_features_group_wise(df)  # 改为组内比较
        df = self.create_baggage_features_group_wise(df)  # 改为组内比较
        df = self.create_cabin_features_group_wise(df)  # 改为组内比较
        df = self.create_route_features(df)
        df = self.create_user_features(df)
        df = self.create_group_features(df, fit_mode=True)
        df = self.create_interaction_features(df)
        
        self.logger.info(f"特征工程完成，最终特征数: {df.shape[1]}")
        self.logger.info(f"新增特征数: {df.shape[1] - len(['Id', 'selected']) if 'selected' in df.columns else df.shape[1] - 1}")
        self.logger.info("="*50)
        return df
    
    def transform(self, df):
        """预测时的特征工程"""
        self.logger.info("="*50)
        self.logger.info("开始测试数据特征工程（预测模式）...")
        self.logger.info(f"输入数据形状: {df.shape}")
        
        # 复制数据
        df = df.copy()
        
        # 创建各类特征
        df = self.create_flight_complexity_features(df)
        df = self.create_time_features(df)
        df = self.create_price_features(df)
        df = self.create_carrier_features(df)
        df = self.create_aircraft_features(df)
        df = self.create_seat_features_group_wise(df)  # 改为组内比较
        df = self.create_baggage_features_group_wise(df)  # 改为组内比较
        df = self.create_cabin_features_group_wise(df)  # 改为组内比较
        df = self.create_route_features(df)
        df = self.create_user_features(df)
        df = self.create_group_features(df, fit_mode=False)
        df = self.create_interaction_features(df)
        
        self.logger.info(f"测试数据特征工程完成，最终特征数: {df.shape[1]}")
        self.logger.info("="*50)
        return df

if __name__ == "__main__":
    # 测试特征工程
    logger = get_logger('FeatureEngineerTest', 'feature_engineering_test.log')
    logger.info("测试特征工程模块...")
    
    # 假设已有预处理后的数据
    try:
        train_df = pd.read_parquet('train_processed.parquet')
        test_df = pd.read_parquet('test_processed.parquet')
        
        # 初始化特征工程器
        feature_engineer = FeatureEngineer('feature_engineering.log')
        
        # 处理训练数据
        train_featured = feature_engineer.fit_transform(train_df)
        logger.info(f"训练数据特征工程完成: {train_featured.shape}")
        
        # 处理测试数据
        test_featured = feature_engineer.transform(test_df)
        logger.info(f"测试数据特征工程完成: {test_featured.shape}")
        
        # 保存特征工程后的数据
        train_featured.to_parquet('train_featured.parquet', index=False)
        test_featured.to_parquet('test_featured.parquet', index=False)
        
        logger.info("特征工程完成并保存！")
        
    except FileNotFoundError:
        logger.error("请先运行数据预处理模块生成processed数据文件") 