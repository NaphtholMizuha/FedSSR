import polars as pl
from pathlib import Path

# === 1. 配置区域 ===
BASE_LOG_DIR = "log"

SCENARIO_MAP = {
    "ResNet (IID)": "resnet", 
    "ResNet (Dir)": "resnet-dir",
    "ShuffleNet (IID)": "shufflenet",
    "ShuffleNet (Dir)": "shufflenet-dir"
}

# 必须小写，用于排序
TARGET_METHOD_ORDER = ["fedavg", "fedssr", "rfed", "rflpa", "bsrfl"]

def extract_acc_top5(file_path):
    """读取parquet并计算Acc@5"""
    try:
        q = pl.scan_parquet(file_path)
        if "acc" in q.columns:
            return q.select(
                pl.col("acc").sort(descending=True).head(5).mean()
            ).collect().item()
    except Exception:
        return None
    return None

def collect_all_data(base_path, scenario_map):
    base_dir = Path(base_path)
    data_records = []
    
    print(f"Scanning directory: {base_dir.absolute()} ...")

    for scenario_label, folder_name in scenario_map.items():
        # 兼容 folder-name 和 folder_name
        scenario_path = base_dir / folder_name
        if not scenario_path.exists() and (base_dir / folder_name.replace("-", "_")).exists():
             scenario_path = base_dir / folder_name.replace("-", "_")

        if not scenario_path.exists():
            continue

        for method_dir in scenario_path.iterdir():
            if not method_dir.is_dir(): continue
            method_name = method_dir.name.lower()

            for seed_dir in method_dir.iterdir():
                if not seed_dir.is_dir() or not seed_dir.name.isdigit(): continue

                for file_path in seed_dir.glob("*.parquet"):
                    acc = extract_acc_top5(file_path)
                    if acc is not None:
                        data_records.append({
                            "scenario": scenario_label,
                            "method": method_name,
                            "attack": file_path.stem,
                            "acc": acc
                        })
    
    if not data_records: return None
    return pl.DataFrame(data_records)

def apply_latex_formatting(struct: dict) -> str:
    """
    根据 rank 应用 LaTeX 格式
    mean, std 均为原始 float (0.0 - 1.0)
    """
    mean_val = struct['mean']
    std_val = struct['std']
    rank = struct['rank']
    method = struct['method']
    
    # 基础字符串: 85.12 (0.34)
    # 强制保留两位小数
    base_str = f"{mean_val * 100:.2f} ({std_val * 100:.2f})"
    
    # FedAvg 不参与加粗/下划线
    if method == "fedavg":
        return base_str
    
    # Rank 1: 加粗 \textbf{}
    if rank == 1:
        return f"\\textbf{{{base_str}}}"
    
    # Rank 2: 下划线 \underline{}
    if rank == 2:
        return f"\\underline{{{base_str}}}"
        
    return base_str

def generate_latex(df: pl.DataFrame, method_cols: list, save_path: str):
    latex_lines = []
    latex_lines.append(r"\begin{table*}[htbp]")
    latex_lines.append(r"\centering")
    latex_lines.append(r"\caption{Top-5 Accuracy (\%) under different attacks. \textbf{Bold}: Best; \underline{Underline}: Second Best (excluding FedAvg).}")
    latex_lines.append(r"\label{tab:acc_summary}")
    
    col_str = "ll" + "c" * len(method_cols)
    latex_lines.append(r"\begin{tabular}{" + col_str + "}")
    latex_lines.append(r"\toprule")
    
    header_methods = [f"\\textbf{{{m.upper()}}}" for m in method_cols]
    header_row = r"\textbf{Scenario} & \textbf{Attack} & " + " & ".join(header_methods) + r" \\"
    latex_lines.append(header_row)
    latex_lines.append(r"\midrule")
    
    current_scenario = None
    rows = df.rows(named=True)
    
    for row in rows:
        scenario = row['scenario']
        attack = row['attack']
        
        if scenario != current_scenario:
            if current_scenario is not None:
                latex_lines.append(r"\midrule")
            scenario_cell = scenario # 可以包裹 \multirow
            current_scenario = scenario
        else:
            scenario_cell = "" 
            
        line_parts = [scenario_cell, attack]
        
        for method in method_cols:
            val = row[method]
            # 数据中已经包含了 \textbf 等 LaTeX 标记
            if val == "-":
                line_parts.append("-")
            else:
                line_parts.append(val)
        
        latex_lines.append(" & ".join(line_parts) + r" \\")

    latex_lines.append(r"\bottomrule")
    latex_lines.append(r"\end{tabular}")
    latex_lines.append(r"\end{table*}")
    
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(latex_lines))
    print(f"LaTeX saved to: {save_path}")

def main():
    # 1. 收集数据
    df = collect_all_data(BASE_LOG_DIR, SCENARIO_MAP)
    if df is None:
        print("No data found.")
        return

    # 2. 聚合 (Mean, Std)
    agg_df = df.group_by(["scenario", "attack", "method"]).agg([
        pl.col("acc").mean().alias("mean"),
        pl.col("acc").std().fill_null(0.0).alias("std")
    ])

    # 3. 计算排名 (Ranking)
    # 逻辑: 仅对非 fedavg 的方法进行排名
    # dense 排名: 90, 90, 80 => rank 1, 1, 2
    ranked_df = agg_df.with_columns(
        pl.when(pl.col("method") != "fedavg")
        .then(
            pl.col("mean")
            .rank(method="dense", descending=True)
            .over(["scenario", "attack"])
        )
        .otherwise(None) # FedAvg rank 为 null
        .alias("rank")
    )

    # 4. 格式化字符串 (应用 \textbf 和 \underline)
    fmt_df = ranked_df.with_columns(
        pl.struct(["mean", "std", "rank", "method"])
        .map_elements(apply_latex_formatting, return_dtype=pl.String)
        .alias("summary")
    )

    # 5. 透视表 (Pivot)
    pivot_df = fmt_df.pivot(
        on="method",
        index=["scenario", "attack"],
        values="summary"
    )

    # 6. 处理列顺序 (补全缺失列 + 排序)
    existing_cols = pivot_df.columns
    final_method_cols = []
    
    select_exprs = [pl.col("scenario"), pl.col("attack")]
    for method in TARGET_METHOD_ORDER:
        if method in existing_cols:
            select_exprs.append(pl.col(method))
            final_method_cols.append(method)
        else:
            select_exprs.append(pl.lit("-").alias(method))
            final_method_cols.append(method)

    # 7. 处理行顺序
    scenario_order = pl.DataFrame({
        "scenario": list(SCENARIO_MAP.keys()),
        "s_order": range(len(SCENARIO_MAP))
    })

    final_df = (
        pivot_df
        .join(scenario_order, on="scenario")
        .select(select_exprs + [pl.col("s_order")])
        .sort(["s_order", "attack"])
        .drop("s_order")
        .fill_null("-")
    )

    # === 输出 ===
    print("\n=== Generated Table Preview (Raw LaTeX strings) ===")
    with pl.Config(tbl_formatting="ASCII_MARKDOWN", tbl_rows=-1, fmt_str_lengths=50):
        print(final_df)
    
    # 保存 CSV (包含 LaTeX 命令，方便检查)
    final_df.write_csv("acc_summary_latex_raw.csv")

    # 生成最终 .tex 文件
    generate_latex(final_df, final_method_cols, "acc_summary.tex")

if __name__ == "__main__":
    main()