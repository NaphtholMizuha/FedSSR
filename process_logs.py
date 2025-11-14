import polars as pl
import glob
from pathlib import Path
import os
import argparse
import pandas as pd

def process_logs(output_format=None):
    try:
        files = glob.glob("/home/wuzihou/code/FedMozi-1/log/resnet/**/*.parquet", recursive=True)

        if not files:
            print("No .parquet files found in the log directory.")
            return

        results = []
        for file in files:
            try:
                p = Path(file)
                algorithm = p.parts[-3]
                condition = p.stem

                df = pl.read_parquet(file)

                if "acc" in df.columns:
                    mean_top5_acc = df.select(
                        pl.col("acc").sort(descending=True).head(5).mean()
                    ).item()

                    results.append({
                        "algorithm": algorithm,
                        "condition": condition,
                        "mean_top5_acc": mean_top5_acc
                    })
                else:
                    print(f"Warning: 'acc' column not found in {file}")

            except Exception as e:
                print(f"Error processing file {file}: {e}")

        if results:
            df_results = pl.DataFrame(results)
            df_final = df_results.group_by(["algorithm", "condition"]).agg(
                pl.mean("mean_top5_acc").alias("avg_acc")
            )

            df_pivot = df_final.pivot(
                index="algorithm",
                on="condition",
                values="avg_acc"
            ).sort("algorithm")
            
            df_pivot = df_pivot.filter(pl.col("algorithm") != "FedMozi")

            if "noattack" in df_pivot.columns:
                cols = ["algorithm", "noattack"] + [col for col in df_pivot.columns if col not in ["algorithm", "noattack"]]
                df_pivot = df_pivot.select(cols)

            if output_format == 'latex':
                print(df_pivot.to_pandas().to_latex(index=False))
            elif output_format == 'markdown':
                print(df_pivot.to_pandas().to_markdown(index=False))
            else:
                print("Here is the resulting dataframe:")
                print(df_pivot)
        else:
            print("No results to process. It's possible that no files had an 'acc' column.")

    except ImportError as e:
        print(f"Error: A required library is not installed. Please install it to run this script. Details: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process log files and output a summary table.')
    parser.add_argument('--format', type=str, choices=['latex', 'markdown'], help='Output format for the table.')
    args = parser.parse_args()
    process_logs(args.format)