import polars as pl
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import glob
import re
import os
import argparse
from collections import defaultdict

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
    plt.figure(figsize=(5, 3))
    
    # 简洁的折线图
    x_line = np.arange(len(ratio_labels))
    plt.plot(x_line, accent_means, label='Accent Attack', marker='o', linewidth=2, color=CB_PALETTE["orange"])
    plt.plot(x_line, minmax_means, label='MinMax Attack', marker='s', linewidth=2, color=CB_PALETTE["blue"])
    
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

def _extract_credits_series(file_path, m_client, n_client):
    with open(file_path, "r") as f:
        s = f.read()
    matches = re.findall(r"credits:\s*\[([^\]]+)\]", s, flags=re.DOTALL)
    mali_avg = []
    beni_avg = []
    mali_max = []
    beni_min = []
    for m in matches:
        arr = np.fromstring(m.replace("\n", " "), sep=" ")
        if arr.size >= n_client:
            mseg = arr[:m_client]
            bseg = arr[m_client:n_client]
            mali_avg.append(mseg.mean())
            beni_avg.append(bseg.mean())
            mali_max.append(mseg.max())
            beni_min.append(bseg.min())
    return np.array(mali_avg), np.array(beni_avg), np.array(mali_max), np.array(beni_min)

def plot_credits(pattern, m_client, n_client):
    files = glob.glob(pattern, recursive=True)
    for fp in files:
        ma, ba, mmax, bmin = _extract_credits_series(fp, m_client, n_client)
        if ma.size == 0:
            continue
        x = np.arange(ma.size)
        plt.figure(figsize=(6, 3))
        plt.plot(x, ba, label="Benign avg", linewidth=2.5, linestyle="-", color=CB_PALETTE["blue"])
        plt.plot(x, bmin, label="Benign min", linewidth=1.5, linestyle="-", color=CB_PALETTE["sky"])
        plt.plot(x, ma, label="Malicious avg", linewidth=2.5, linestyle="--", color=CB_PALETTE["vermillion"])
        plt.plot(x, mmax, label="Malicious max", linewidth=1.5, linestyle="--", color=CB_PALETTE["orange"])
        plt.xlabel("Transmission Round")
        plt.ylabel("Reputation")
        plt.axhline(0, color="gray", linestyle="--", linewidth=1)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        out_path = os.path.join(os.path.dirname(fp), "credits_diverge.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")

def plot_credits_avg(base_dir, m_client, n_client, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    files = glob.glob(os.path.join(base_dir, "*", "*.log"))
    groups = defaultdict(list)
    for fp in files:
        scenario = Path(fp).stem
        groups[scenario].append(fp)
    for scenario, fps in groups.items():
        mas = []
        bas = []
        mmaxs = []
        bmins = []
        max_len = 0
        for fp in fps:
            ma, ba, mmax, bmin = _extract_credits_series(fp, m_client, n_client)
            if ma.size > 0:
                mas.append(ma)
                bas.append(ba)
                mmaxs.append(mmax)
                bmins.append(bmin)
                max_len = max(max_len, ma.size, ba.size, mmax.size, bmin.size)
        if not mas:
            continue
        def pad_and_stack(arrs, L):
            stacked = []
            for a in arrs:
                if a.size < L:
                    pad = np.full(L - a.size, np.nan, dtype=float)
                    a = np.concatenate([a, pad])
                stacked.append(a)
            return np.vstack(stacked)
        mas_mat = pad_and_stack(mas, max_len)
        bas_mat = pad_and_stack(bas, max_len)
        mmax_mat = pad_and_stack(mmaxs, max_len)
        bmin_mat = pad_and_stack(bmins, max_len)
        ma_mean = np.nanmean(mas_mat, axis=0)
        ba_mean = np.nanmean(bas_mat, axis=0)
        ma_max_mean = np.nanmean(mmax_mat, axis=0)
        ba_min_mean = np.nanmean(bmin_mat, axis=0)
        x = np.arange(max_len)
        plt.figure(figsize=(6, 3))
        plt.plot(x, ba_mean, label="Benign avg", linewidth=2.5, linestyle="-", color=CB_PALETTE["blue"])
        plt.plot(x, ba_min_mean, label="Benign min", linewidth=1.5, linestyle="-", color=CB_PALETTE["sky"])
        plt.plot(x, ma_mean, label="Malicious avg", linewidth=2.5, linestyle="--", color=CB_PALETTE["vermillion"])
        plt.plot(x, ma_max_mean, label="Malicious max", linewidth=1.5, linestyle="--", color=CB_PALETTE["orange"])
        plt.xlabel("Transmission Round")
        plt.ylabel("Reputation")
        plt.axhline(0, color="gray", linestyle="--", linewidth=1)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        out_path = os.path.join(out_dir, f"credits_{scenario}.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out_path}")
        
def plot_credits_best(base_dir, m_client, n_client, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    files = glob.glob(os.path.join(base_dir, "*", "*.log"))
    groups = defaultdict(list)
    for fp in files:
        scenario = Path(fp).stem
        groups[scenario].append(fp)
    for scenario, fps in groups.items():
        best_fp = None
        best_score = -np.inf
        best_series = None
        for fp in fps:
            ma, ba, mmax, bmin = _extract_credits_series(fp, m_client, n_client)
            if mmax.size == 0:
                continue
            sep = bmin - mmax
            score = np.nansum(sep)
            if score > best_score:
                best_score = score
                best_fp = fp
                best_series = (ma, ba, mmax, bmin)
        if best_series is None:
            continue
        ma, ba, mmax, bmin = best_series
        x = np.arange(ma.size)
        plt.figure(figsize=(6, 3))
        plt.plot(x, ba, label="Benign avg", linewidth=2.5, linestyle="-", color=CB_PALETTE["blue"])
        plt.plot(x, bmin, label="Benign min", linewidth=1.5, linestyle="-", color=CB_PALETTE["sky"])
        plt.plot(x, ma, label="Malicious avg", linewidth=2.5, linestyle="--", color=CB_PALETTE["vermillion"])
        plt.plot(x, mmax, label="Malicious max", linewidth=1.5, linestyle="--", color=CB_PALETTE["orange"])
        plt.xlabel("Transmission Round")
        plt.ylabel("Reputation")
        plt.axhline(0, color="gray", linestyle="--", linewidth=1)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        out_path = os.path.join(out_dir, f"credits_{scenario}_best.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out_path} (score={best_score:.4f}, file={best_fp})")

def plot_credits_best_grid(base_dir, m_client, n_client, out_dir, ncols=3):
    os.makedirs(out_dir, exist_ok=True)
    files = glob.glob(os.path.join(base_dir, "*", "*.log"))
    groups = defaultdict(list)
    for fp in files:
        scenario = Path(fp).stem
        groups[scenario].append(fp)
    scenarios = [s for s in sorted(groups.keys()) if s != "noattack"]
    series_map = {}
    for scenario in scenarios:
        fps = groups[scenario]
        best_score = -np.inf
        best_series = None
        for fp in fps:
            ma, ba, mmax, bmin = _extract_credits_series(fp, m_client, n_client)
            if mmax.size == 0:
                continue
            sep = bmin - mmax
            score = np.nansum(sep)
            if score > best_score:
                best_score = score
                best_series = (ma, ba, mmax, bmin)
        if best_series is not None:
            series_map[scenario] = best_series
    if not series_map:
        return
    n = len(series_map)
    cols = ncols
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3), squeeze=False)
    handles = None
    labels = None
    idx = 0
    for r in range(rows):
        for c in range(cols):
            ax = axes[r][c]
            if idx < len(scenarios):
                scenario = scenarios[idx]
                if scenario in series_map:
                    ma, ba, mmax, bmin = series_map[scenario]
                    x = np.arange(mmax.size)
                    l4, = ax.plot(x, ba, linewidth=2.5, linestyle="-", color=CB_PALETTE["blue"], label="Benign avg")
                    l2, = ax.plot(x, bmin, linewidth=1.5, linestyle="-", color=CB_PALETTE["sky"], label="Benign min")
                    l3, = ax.plot(x, ma, linewidth=2.5, linestyle="-", color=CB_PALETTE["vermillion"], label="Malicious avg")
                    l1, = ax.plot(x, mmax, linewidth=1.5, linestyle="-", color=CB_PALETTE["orange"], label="Malicious max")
                    ax.set_title({
                        'minmax': "Min-Max",
                        'minsum': "Min-Sum",
                        'scalesign': "ScaleSign-Max",
                        'sssum': "ScaleSign-Sum",
                        'ascent': "Ascent",
                        'lie': "Little-is-Enough",
                    }[scenario])
                    ax.set_xlabel("Transmission Round")
                    ax.set_ylabel("Reputation")
                    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
                    ax.grid(True, alpha=0.3)
                    if handles is None:
                        handles = [l1, l2, l3, l4]
                        labels = ["Malicious max", "Benign min", "Malicious avg", "Benign avg"]
                else:
                    ax.axis("off")
            else:
                ax.axis("off")
            idx += 1
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    if handles and labels:
        fig.legend(handles, labels, loc="lower center", ncol=4)
    out_path = os.path.join(out_dir, "credits_best_grid.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["comparison", "credits", "credits_avg", "credits_best", "credits_best_grid"], default="comparison")
    parser.add_argument("--pattern", default="log/shufflenet/fedssr/**/*.log")
    parser.add_argument("--m_client", type=int, default=5)
    parser.add_argument("--n_client", type=int, default=20)
    parser.add_argument("--base_dir", default="log/shufflenet/fedssr")
    parser.add_argument("--out_dir", default="fig")
    args = parser.parse_args()
    if args.mode == "comparison":
        plot_comparison()
    else:
        if args.mode == "credits":
            plot_credits(args.pattern, args.m_client, args.n_client)
        elif args.mode == "credits_avg":
            plot_credits_avg(args.base_dir, args.m_client, args.n_client, args.out_dir)
        elif args.mode == "credits_best":
            plot_credits_best(args.base_dir, args.m_client, args.n_client, args.out_dir)
        else:
            plot_credits_best_grid(args.base_dir, args.m_client, args.n_client, args.out_dir)
