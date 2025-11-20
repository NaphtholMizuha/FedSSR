import polars as pl
import glob
from pathlib import Path
import os
import argparse
import pandas as pd
import numpy as np

def process_logs(output_format=None):
    try:
        # 1. 寻找文件
        log_path = "/home/wuzihou/code/FedMozi-1/log/shufflenet-dir/**/*.parquet"
        files = glob.glob(log_path, recursive=True)

        if not files:
            print(f"No .parquet files found in: {log_path}")
            return

        # 2. 读取并提取数据
        results = []
        for file in files:
            try:
                p = Path(file)
                algorithm = p.parts[-3]
                condition = p.stem

                df = pl.read_parquet(file)

                if "acc" in df.columns:
                    # 取 Top-5 的平均值作为该次实验的性能指标
                    mean_top5_acc = df.select(
                        pl.col("acc").sort(descending=True).head(5).mean()
                    ).item()

                    results.append({
                        "algorithm": algorithm,
                        "condition": condition,
                        "mean_top5_acc": mean_top5_acc
                    })
                
            except Exception as e:
                print(f"Error processing file {file}: {e}")

        if results:
            df_results = pl.DataFrame(results)

            # 3. 聚合：同时计算 均值(mean) 和 标准差(std)
            # 注意：如果某个条件下只有一次实验，std 会是 null，我们需要填充为 0
            df_agg = df_results.group_by(["algorithm", "condition"]).agg([
                pl.mean("mean_top5_acc").alias("avg_acc"),
                pl.std("mean_top5_acc").fill_null(0.0).alias("std_acc")
            ])

            # 4. 格式化：转换为 Pandas 进行字符串格式化处理
            # 目标格式：保留4位均值 ± 保留2位标准差
            pdf_agg = df_agg.to_pandas()
            
            # 使用 f-string 格式化
            pdf_agg['formatted_cell'] = pdf_agg.apply(
                lambda row: f"{row['avg_acc']:.4f} ± {row['std_acc']:.2f}", 
                axis=1
            )

            # 5. 透视 (Pivot)
            # 现在 values 是字符串类型的 'formatted_cell'
            df_pivot = pdf_agg.pivot(
                index="algorithm",
                columns="condition",
                values="formatted_cell"
            ).sort_values("algorithm")
            
            # 6. 过滤与列排序
            # 过滤掉 FedMozi (如需保留请注释掉此行)
            df_pivot = df_pivot[df_pivot.index != "FedMozi"]

            # 调整列顺序，将 'noattack' 放在第一列
            if "noattack" in df_pivot.columns:
                cols = ["noattack"] + [col for col in df_pivot.columns if col != "noattack"]
                df_pivot = df_pivot[cols]

            # 7. 输出
            # 因为已经是格式化好的字符串，不需要再指定 float_format
            if output_format == 'latex':
                print(df_pivot.to_latex(index=True))
            elif output_format == 'markdown':
                print(df_pivot.to_markdown(index=True))
            else:
                print("Here is the resulting dataframe (Mean ± Std):")
                print(df_pivot)

        else:
            print("No valid results found to process.")

    except ImportError as e:
        print(f"Error: A required library is not installed. Details: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process log files and output a summary table.')
    parser.add_argument('--format', type=str, choices=['latex', 'markdown'], help='Output format for the table.')
    args = parser.parse_args()
    process_logs(args.format)