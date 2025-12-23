import polars as pl
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import glob

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

def collect_data(base_path, attack_type):
    """收集指定攻击类型下所有恶意客户端比例的数据"""
    attack_path = Path(base_path) / attack_type
    
    # 恶意客户端比例列表
    malicious_ratios = ['m10%', 'm25%', 'm40%', 'm50%', 'm60%', 'm75%']
    
    results = {}
    
    for ratio in malicious_ratios:
        acc_values = []
        
        # 遍历所有实验重复（0, 1, 2）
        for exp_id in range(3):
            exp_path = attack_path / str(exp_id)
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
            print(f"{attack_type} {ratio}: {mean_acc:.4f} ± {std_acc:.4f} (n={len(acc_values)})")
    
    return results

def plot_comparison():
    """绘制两种攻击下的Acc@5变化对比图"""
    base_path = "log/mclients-shufflenet"
    
    # 收集数据
    print("Collecting accent attack data...")
    accent_data = collect_data(base_path, "accent")
    
    print("\nCollecting minmax attack data...")
    minmax_data = collect_data(base_path, "minmax")
    
    # 准备绘图数据
    ratios = ['m10%', 'm25%', 'm40%', 'm50%', 'm60%', 'm75%']
    ratio_labels = ['10%', '25%', '40%', '50%', '60%', '75%']
    
    accent_means = []
    accent_stds = []
    minmax_means = []
    minmax_stds = []
    
    for ratio in ratios:
        if ratio in accent_data:
            accent_means.append(accent_data[ratio]['mean'])
            accent_stds.append(accent_data[ratio]['std'])
        else:
            accent_means.append(0)
            accent_stds.append(0)
            
        if ratio in minmax_data:
            minmax_means.append(minmax_data[ratio]['mean'])
            minmax_stds.append(minmax_data[ratio]['std'])
        else:
            minmax_means.append(0)
            minmax_stds.append(0)
    
    # 创建图形 - 只保留折线图，缩小尺寸，去掉标题
    plt.figure(figsize=(6, 4))
    
    # 简洁的折线图
    x_line = np.arange(len(ratio_labels))
    plt.plot(x_line, accent_means, label='Accent Attack', marker='o', linewidth=2, color='#ff7f0e')
    plt.plot(x_line, minmax_means, label='MinMax Attack', marker='s', linewidth=2, color='#1f77b4')
    
    plt.xlabel('Malicious Client Ratio')
    plt.ylabel('Acc@5')
    plt.xticks(x_line, ratio_labels)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('mclients_shufflenet_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 打印数据摘要
    print("\n=== Data Summary ===")
    print("Accent Attack:")
    for i, ratio in enumerate(ratios):
        print(f"  {ratio_labels[i]}: {accent_means[i]:.4f} ± {accent_stds[i]:.4f}")
    
    print("\nMinMax Attack:")
    for i, ratio in enumerate(ratios):
        print(f"  {ratio_labels[i]}: {minmax_means[i]:.4f} ± {minmax_stds[i]:.4f}")

if __name__ == "__main__":
    plot_comparison()