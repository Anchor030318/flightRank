import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    def __init__(self):
        self.label_encoders = {}
        self.scalers = {}
        self.imputers = {}
        self.feature_columns = []
        
    def preprocess_time_features(self, df):
        """处理时间特征"""
        time_cols = ['legs0_arrivalAt', 'legs0_departureAt', 'legs1_arrivalAt', 'legs1_departureAt', 'requestDate']
        
        for col in time_cols:
            if col in df.columns:
                # 转换为datetime类型
                df[col] = pd.to_datetime(df[col], errors='coerce')
                
                # 提取时间特征
                df[f'{col}_hour'] = df[col].dt.hour
                df[f'{col}_day_of_week'] = df[col].dt.dayofweek
                df[f'{col}_month'] = df[col].dt.month
                df[f'{col}_day'] = df[col].dt.day
                
                # 删除原始时间列
                df = df.drop(columns=[col])
        
        return df
    
    def preprocess_duration_features(self, df):
        """处理持续时间特征"""
        duration_cols = ['legs0_duration', 'legs1_duration', 
                        'legs0_segments0_duration', 'legs0_segments1_duration', 
                        'legs0_segments2_duration', 'legs0_segments3_duration',
                        'legs1_segments0_duration', 'legs1_segments1_duration',
                        'legs1_segments2_duration', 'legs1_segments3_duration']
        
        for col in duration_cols:
            if col in df.columns:
                # 转换为分钟数
                df[f'{col}_minutes'] = df[col].apply(self._duration_to_minutes)
                df = df.drop(columns=[col])
        
        return df
    
    def _duration_to_minutes(self, duration_str):
        """将持续时间字符串转换为分钟数"""
        if pd.isna(duration_str) or duration_str == '':
            return 0
        
        try:
            # 处理格式如 "02:30:00" 或 "2:30"
            if ':' in str(duration_str):
                parts = str(duration_str).split(':')
                if len(parts) == 3:
                    hours, minutes, seconds = map(int, parts)
                    return hours * 60 + minutes + seconds / 60
                elif len(parts) == 2:
                    hours, minutes = map(int, parts)
                    return hours * 60 + minutes
            return 0
        except:
            return 0
    
    def preprocess_categorical_features(self, df, fit_mode=True):
        """处理分类特征"""
        categorical_cols = [
            'searchRoute', 'frequentFlyer', 'ranker_id',
            'legs0_segments0_aircraft_code', 'legs0_segments0_arrivalTo_airport_city_iata',
            'legs0_segments0_arrivalTo_airport_iata', 'legs0_segments0_departureFrom_airport_iata',
            'legs0_segments0_flightNumber', 'legs0_segments0_marketingCarrier_code',
            'legs0_segments0_operatingCarrier_code'
        ]
        
        # 添加所有segments的分类特征
        for leg in ['legs0', 'legs1']:
            for seg in ['segments0', 'segments1', 'segments2', 'segments3']:
                cols = [
                    f'{leg}_{seg}_aircraft_code',
                    f'{leg}_{seg}_arrivalTo_airport_city_iata',
                    f'{leg}_{seg}_arrivalTo_airport_iata',
                    f'{leg}_{seg}_departureFrom_airport_iata',
                    f'{leg}_{seg}_flightNumber',
                    f'{leg}_{seg}_marketingCarrier_code',
                    f'{leg}_{seg}_operatingCarrier_code'
                ]
                categorical_cols.extend(cols)
        
        # 处理分类特征
        for col in categorical_cols:
            if col in df.columns:
                # 填充缺失值
                df[col] = df[col].fillna('unknown')
                
                if fit_mode:
                    # 训练模式：创建和拟合编码器
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                    self.label_encoders[col] = le
                else:
                    # 预测模式：使用已有编码器
                    if col in self.label_encoders:
                        le = self.label_encoders[col]
                        # 处理未见过的类别
                        df[col] = df[col].apply(lambda x: x if x in le.classes_ else 'unknown')
                        df[col] = le.transform(df[col].astype(str))
                    else:
                        df[col] = 0  # 如果没有编码器，设为0
        
        return df
    
    def preprocess_numerical_features(self, df, fit_mode=True):
        """处理数值特征"""
        numerical_cols = [
            'companyID', 'corporateTariffCode', 'nationality', 'pricingInfo_passengerCount',
            'profileId', 'taxes', 'totalPrice'
        ]
        
        # 添加所有segments的数值特征
        for leg in ['legs0', 'legs1']:
            for seg in ['segments0', 'segments1', 'segments2', 'segments3']:
                cols = [
                    f'{leg}_{seg}_baggageAllowance_quantity',
                    f'{leg}_{seg}_baggageAllowance_weightMeasurementType',
                    f'{leg}_{seg}_cabinClass',
                    f'{leg}_{seg}_seatsAvailable'
                ]
                numerical_cols.extend(cols)
        
        # 添加miniRules特征
        mini_rules_cols = [
            'miniRules0_monetaryAmount', 'miniRules0_percentage', 'miniRules0_statusInfos',
            'miniRules1_monetaryAmount', 'miniRules1_percentage', 'miniRules1_statusInfos',
            'pricingInfo_isAccessTP'
        ]
        numerical_cols.extend(mini_rules_cols)
        
        # 处理数值特征
        for col in numerical_cols:
            if col in df.columns:
                # 填充缺失值
                if fit_mode:
                    imputer = SimpleImputer(strategy='median')
                    df[col] = imputer.fit_transform(df[[col]]).ravel()
                    self.imputers[col] = imputer
                else:
                    if col in self.imputers:
                        df[col] = self.imputers[col].transform(df[[col]]).ravel()
                    else:
                        df[col] = df[col].fillna(0)
        
        return df
    
    def preprocess_boolean_features(self, df):
        """处理布尔特征"""
        boolean_cols = ['bySelf', 'isAccess3D', 'isVip', 'sex']
        
        for col in boolean_cols:
            if col in df.columns:
                df[col] = df[col].astype(int)
        
        return df
    
    def create_price_features(self, df):
        """创建价格相关特征"""
        if 'totalPrice' in df.columns and 'taxes' in df.columns:
            # 基础价格（不含税）
            df['base_price'] = df['totalPrice'] - df['taxes']
            
            # 税收比例
            df['tax_ratio'] = df['taxes'] / (df['totalPrice'] + 1e-8)
            
            # 价格分位数特征
            df['price_rank'] = df['totalPrice'].rank(pct=True)
        
        return df
    
    def create_segment_features(self, df):
        """创建航段特征"""
        # 统计每个航班的航段数量
        for leg in ['legs0', 'legs1']:
            segment_count = 0
            for seg in ['segments0', 'segments1', 'segments2', 'segments3']:
                col = f'{leg}_{seg}_aircraft_code'
                if col in df.columns:
                    segment_count += (~df[col].isna()).astype(int)
            df[f'{leg}_segment_count'] = segment_count
        
        return df
    
    def fit_transform(self, df):
        """训练时的数据预处理"""
        print("开始数据预处理...")
        
        # 复制数据
        df = df.copy()
        
        # 各种预处理步骤
        df = self.preprocess_time_features(df)
        df = self.preprocess_duration_features(df)
        df = self.preprocess_categorical_features(df, fit_mode=True)
        df = self.preprocess_numerical_features(df, fit_mode=True)
        df = self.preprocess_boolean_features(df)
        df = self.create_price_features(df)
        df = self.create_segment_features(df)
        
        # 保存特征列名
        if 'selected' in df.columns:
            self.feature_columns = [col for col in df.columns if col not in ['Id', 'selected']]
        else:
            self.feature_columns = [col for col in df.columns if col != 'Id']
        
        print(f"预处理完成，特征数量: {len(self.feature_columns)}")
        return df
    
    def transform(self, df):
        """预测时的数据预处理"""
        print("开始测试数据预处理...")
        
        # 复制数据
        df = df.copy()
        
        # 各种预处理步骤
        df = self.preprocess_time_features(df)
        df = self.preprocess_duration_features(df)
        df = self.preprocess_categorical_features(df, fit_mode=False)
        df = self.preprocess_numerical_features(df, fit_mode=False)
        df = self.preprocess_boolean_features(df)
        df = self.create_price_features(df)
        df = self.create_segment_features(df)
        
        # 确保特征列与训练时一致
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0
        
        print(f"测试数据预处理完成，特征数量: {len(self.feature_columns)}")
        return df
    
    def get_feature_columns(self):
        """获取特征列名"""
        return self.feature_columns

if __name__ == "__main__":
    # 测试数据预处理
    print("测试数据预处理模块...")
    
    # 读取数据
    train_df = pd.read_parquet('data/train.parquet')
    test_df = pd.read_parquet('data/test.parquet')
    
    # 初始化预处理器
    preprocessor = DataPreprocessor()
    
    # 处理训练数据
    train_processed = preprocessor.fit_transform(train_df)
    print(f"训练数据处理完成: {train_processed.shape}")
    
    # 处理测试数据
    test_processed = preprocessor.transform(test_df)
    print(f"测试数据处理完成: {test_processed.shape}")
    
    # 保存处理后的数据
    train_processed.to_parquet('train_processed.parquet', index=False)
    test_processed.to_parquet('test_processed.parquet', index=False)
    
    print("数据预处理完成并保存！") 