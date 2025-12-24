import polars as pl
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os

# 使用与现有代码一致的调色板
CB_PALETTE = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "black": "#000000",
}

def extract_acc_top5(file_path):
    """从parquet文件中提取Top-5准确率的平均值"""
    try:
        df = pl.read_parquet(file_path)
        if "acc" in df.columns:
            # 取Top-5的平均值
            mean_top5_acc = df.select(
                pl.col("acc").sort(descending=True).head(5).mean()
            ).item()
            return mean_top5_acc
        return None
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def collect_method_data(base_path, method_name):
    """收集指定方法下所有恶意客户端比例的数据"""
    method_path = Path(base_path) / method_name
    
    # 恶意客户端比例列表
    malicious_ratios = ['m10%', 'm25%', 'm40%', 'm50%', 'm60%', 'm75%']
    
    results = {}
    
    for ratio in malicious_ratios:
        acc_values = []
        
        # 遍历所有实验重复（0, 1, 2）
        for exp_id in range(3):
            exp_path = method_path / str(exp_id)
            parquet_file = exp_path / f"{ratio}.parquet"
            
            if parquet_file.exists():
                acc = extract_acc_top5(parquet_file)
                if acc is not None:
                    acc_values.append(acc)
        
        if acc_values:
            # 计算均值和标准差
            mean_acc = np.mean(acc_values)
            std_acc = np.std(acc_values) if len(acc_values) > 1 else 0.0
            results[ratio] = {'mean': mean_acc, 'std': std_acc, 'values': acc_values}
            print(f"{method_name} {ratio}: {mean_acc:.4f} ± {std_acc:.4f} (n={len(acc_values)})")
    
    return results

def plot_mclient_resnet_comparison():
    """绘制mclient-resnet下不同方法的Acc@5变化对比图"""
    base_path = "log/mclient-resnet"
    
    # 获取所有方法文件夹
    base_dir = Path(base_path)
    if not base_dir.exists():
        print(f"Error: Directory {base_path} does not exist!")
        return
    
    methods = [d.name for d in base_dir.iterdir() if d.is_dir()]
    print(f"Found methods: {methods}")
    
    # 收集每个方法的数据
    all_method_data = {}
    for method in methods:
        print(f"\nCollecting {method} data...")
        method_data = collect_method_data(base_path, method)
        if method_data:
            all_method_data[method] = method_data
    
    if not all_method_data:
        print("No data found!")
        return
    
    # 准备绘图数据
    ratios = ['m10%', 'm25%', 'm40%', 'm50%', 'm60%', 'm75%']
    ratio_labels = ['10%', '25%', '40%', '50%', '60%', '75%']
    
    # 创建图形
    plt.figure(figsize=(8, 5))
    
    # 为每个方法绘制折线
    colors = [CB_PALETTE["blue"], CB_PALETTE["orange"], CB_PALETTE["green"], 
              CB_PALETTE["vermillion"], CB_PALETTE["purple"], CB_PALETTE["sky"]]
    markers = ['o', 's', '^', 'D', 'v', '<']
    
    x_line = np.arange(len(ratio_labels))
    
    for i, (method, method_data) in enumerate(all_method_data.items()):
        means = []
        stds = []
        
        for ratio in ratios:
            if ratio in method_data:
                means.append(method_data[ratio]['mean'])
                stds.append(method_data[ratio]['std'])
            else:
                means.append(np.nan)
                stds.append(0)
        
        # 绘制折线
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        
        plt.plot(x_line, means, label=method.upper(), marker=marker, 
                linewidth=2, color=color, markersize=6)
        
        # 添加误差棒（如果有标准差）
        valid_indices = ~np.isnan(means)
        if np.any(valid_indices) and np.any(np.array(stds)[valid_indices] > 0):
            plt.errorbar(x_line[valid_indices], np.array(means)[valid_indices], 
                        yerr=np.array(stds)[valid_indices], 
                        fmt='none', color=color, alpha=0.5, capsize=3)
    
    plt.xlabel('Malicious Client Ratio')
    plt.ylabel('Acc@5')
    plt.title('Acc@5 vs Malicious Client Ratio (ResNet)')
    plt.xticks(x_line, ratio_labels)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    output_path = 'plot/mclient_resnet_acc5_comparison.png'
    os.makedirs('plot', exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nPlot saved to: {output_path}")
    
    # 打印数据摘要
    print("\n=== Data Summary ===")
    for method, method_data in all_method_data.items():
        print(f"\n{method.upper()}:")
        for i, ratio in enumerate(ratios):
            if ratio in method_data:
                mean_val = method_data[ratio]['mean']
                std_val = method_data[ratio]['std']
                print(f"  {ratio_labels[i]}: {mean_val:.4f} ± {std_val:.4f}")
            else:
                print(f"  {ratio_labels[i]}: No data")

if __name__ == "__main__":
    plot_mclient_resnet_comparison()