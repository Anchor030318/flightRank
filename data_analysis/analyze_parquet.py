import pandas as pd
import sys

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)



def analyze_parquet(file_path, output_path="output.txt"):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"正在读取文件: {file_path}\n")
        df = pd.read_parquet(file_path)
        f.write("\n==== 基本信息 ====\n")
        f.write(f"行数: {df.shape[0]}, 列数: {df.shape[1]}\n")
        f.write(f"字段名: {list(df.columns)}\n")
        f.write("\n==== 数据类型 ====\n")
        f.write(f"{df.dtypes}\n")
        f.write("\n==== 前5行数据 ====\n")
        f.write(f"{df.head()}\n")
        f.write("\n==== 缺失值统计 ====\n")
        f.write(f"{df.isnull().sum()}\n")
        f.write("\n==== 唯一值数量 ====\n")
        f.write(f"{df.nunique()}\n")
        f.write("\n==== 描述性统计 ====\n")
        f.write(f"{df.describe(include='all')}\n")
        
        # 新增：统计每个 profileid 的唯一 ranker_id 数量分布
        if 'profileId' in df.columns and 'ranker_id' in df.columns:
            f.write("\n==== 每个 profileId 的唯一 ranker_id 数量分布 ====\n")
            profile_ranker_counts = df.groupby('profileId')['ranker_id'].nunique()
            f.write(f"描述性统计:\n{profile_ranker_counts.describe()}\n")
            f.write(f"\n分布直方图（区间: 请求数-用户数）:\n{profile_ranker_counts.value_counts().sort_index()}\n")
        else:
            f.write("\n未找到 profileid 或 ranker_id 字段，无法统计分布。\n")

if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("用法: python analyze_parquet.py 文件路径 [输出文件路径]")
    else:
        file_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) == 3 else "output.txt"
        analyze_parquet(file_path, output_path)