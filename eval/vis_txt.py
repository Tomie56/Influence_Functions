import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from typing import Tuple, List, Dict

# -------------------------- 全局配置 --------------------------
# 输入文件路径
FILE_PATH_1 = "/mnt/afs/jingjinhao/project/montessori-instruct-mllm/results/2025-08-07-v1/loss_1_to_10000_my_eval_data_2025-08-07_19-55-27.txt"
FILE_PATH_2 = "/mnt/afs/jingjinhao/project/montessori-instruct-mllm/results/2025-08-08-v2/loss_1_to_10000_my_eval_data_2025-08-08_16-03-47.txt"

# 输出目录 (默认当前目录，可修改)
OUTPUT_DIR = "./overlap_analysis_results"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# 参数配置
CUSTOM_TOPK = 5000  # 比较前5000个
MAX_SAMPLE_RATIO = 1.0  # 折线图显示所有样本 (因为总数可能只有10000，显示全貌更好)
STEP_SAMPLE_RATIO = 0.05  # 折线图步长

# -------------------------- ACL 绘图风格配置 --------------------------
# 设置字体为 TrueType (Type 42)，这对 ACL/EMNLP 提交至关重要，便于后期编辑且避免 Type 3 字体报错
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams["font.family"] = "DejaVu Sans" # 论文常用衬线字体
plt.rcParams["font.size"] = 10
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.autolayout"] = True

# -------------------------- 数据加载函数 --------------------------
def load_txt_data(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    读取txt文件，假设文件内容为数字序列。
    返回: (values, ids)
    ids 默认为数据在文件中的顺序 (0, 1, 2...)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    print(f"Loading file: {file_path}")
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            # 处理可能的格式：去除方括号、逗号，按空白字符分割
            cleaned_content = content.replace('[', '').replace(']', '').replace(',', ' ')
            values = np.array([float(x) for x in cleaned_content.split()])
            ids = np.arange(len(values)) # 假设行号/顺序即为ID
            
        print(f"  - Loaded {len(values):,} values.")
        return values, ids
    except Exception as e:
        raise RuntimeError(f"Failed to load txt file: {e}")

# -------------------------- 统计与计算函数 --------------------------
def calculate_topk_overlap(ids1: np.ndarray, ids2: np.ndarray, 
                           scores1: np.ndarray, scores2: np.ndarray,
                           topk: int, is_ascending: bool = False) -> Tuple[int, float]:
    """
    计算TopK重合度
    is_ascending=False: 取数值最大的前K个 (Largest / Top)
    is_ascending=True:  取数值最小的前K个 (Smallest / Bottom)
    """
    # 排序并取TopK ID
    def get_topk_ids(scores, ids, k, asc):
        sorted_indices = np.argsort(scores)
        if not asc:
            sorted_indices = sorted_indices[::-1]
        return set(ids[sorted_indices[:k]])

    ids1_set = get_topk_ids(scores1, ids1, topk, is_ascending)
    ids2_set = get_topk_ids(scores2, ids2, topk, is_ascending)
    
    overlap_count = len(ids1_set & ids2_set)
    overlap_prob = overlap_count / topk if topk > 0 else 0.0
    
    return overlap_count, overlap_prob

# -------------------------- 可视化函数 (PDF输出) --------------------------

def plot_topk_histogram_pdf(topk: int, top_count: int, top_prob: float,
                            bottom_count: int, bottom_prob: float, filename: str):
    """
    绘制直方图并保存为PDF
    """
    labels = [f"Top {topk}\n(Highest Values)", f"Bottom {topk}\n(Lowest Values)"]
    counts = [top_count, bottom_count]
    probs = [top_prob, bottom_prob]
    colors = ["#4E79A7", "#F28E2B"] # 使用比较学术的配色
    
    fig, ax = plt.subplots(figsize=(6, 5)) # 尺寸适合论文单栏或半栏
    
    bars = ax.bar(labels, counts, width=0.5, color=colors, alpha=0.8, edgecolor="black")
    
    # 添加数值标签
    y_max = max(counts)
    y_offset = y_max * 0.02
    
    for bar, count, prob in zip(bars, counts, probs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + y_offset,
                f"{count:,}\n({prob:.1%})",
                ha="center", va="bottom", fontsize=11, fontweight="bold")
    
    ax.set_ylabel("Overlapping Sample Count")
    ax.set_title(f"Overlap Consistency (TopK={topk})")
    ax.set_ylim(0, y_max * 1.15)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    
    # 保存为PDF
    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    print(f"Saved histogram to: {save_path}")
    plt.close()

def plot_overlap_curve_pdf(ids1: np.ndarray, ids2: np.ndarray,
                           scores1: np.ndarray, scores2: np.ndarray,
                           filename: str):
    """
    绘制Overlap曲线并保存为PDF (包含High Values和Low Values两条线)
    """
    total = len(ids1)
    # 采样点：从1%到100%，每5%一个点，或者更细
    steps = np.linspace(int(total*0.01), total, 50, dtype=int)
    steps = np.unique(steps[steps > 0]) # 确保无重复且大于0
    
    forward_percents = []  # Largest scores overlap
    backward_percents = [] # Smallest scores overlap
    random_percents = []
    
    # 预先排序索引，加速循环
    # Descending (Largest first)
    idx1_desc = np.argsort(scores1)[::-1]
    idx2_desc = np.argsort(scores2)[::-1]
    # Ascending (Smallest first)
    idx1_asc = np.argsort(scores1)
    idx2_asc = np.argsort(scores2)
    
    print("Calculating curve points...")
    for k in steps:
        # Largest
        set1_high = set(ids1[idx1_desc[:k]])
        set2_high = set(ids2[idx2_desc[:k]])
        forward_percents.append(len(set1_high & set2_high) / k * 100)
        
        # Smallest
        set1_low = set(ids1[idx1_asc[:k]])
        set2_low = set(ids2[idx2_asc[:k]])
        backward_percents.append(len(set1_low & set2_low) / k * 100)
        
        # Random Baseline
        random_percents.append((k / total) * 100)

    # 绘图
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # 绘制曲线
    ax.plot(steps, forward_percents, label="Highest Values Overlap", 
            color="#E15759", linewidth=2, marker='o', markersize=4)
    ax.plot(steps, backward_percents, label="Lowest Values Overlap", 
            color="#4E79A7", linewidth=2, marker='s', markersize=4)
    ax.plot(steps, random_percents, label="Random Baseline", 
            color="gray", linestyle="--", alpha=0.7)
    
    ax.axhline(y=100, color="black", linestyle=":", alpha=0.3)
    
    ax.set_xlabel("Number of Top K Samples")
    ax.set_ylabel("Overlap Percentage (%)")
    ax.set_title(f"Overlap Consistency Curve (Total N={total})")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_xlim(0, total)
    ax.set_ylim(0, 105)
    
    # 保存为PDF
    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    print(f"Saved line chart to: {save_path}")
    plt.close()

# -------------------------- 主程序 --------------------------
if __name__ == "__main__":
    try:
        # 1. 加载数据
        scores1, ids1 = load_txt_data(FILE_PATH_1)
        scores2, ids2 = load_txt_data(FILE_PATH_2)
        
        # 截断数据以匹配长度 (取交集长度)
        # min_len = min(len(scores1), len(scores2))
        min_len = 5000
        if len(scores1) != len(scores2):
            print(f"Warning: File lengths differ ({len(scores1)} vs {len(scores2)}). Truncating to {min_len}.")
            scores1 = scores1[:min_len]
            ids1 = ids1[:min_len]
            scores2 = scores2[:min_len]
            ids2 = ids2[:min_len]
        
        # 2. 计算 Spearman 相关系数
        corr, p_val = spearmanr(scores1, scores2)
        print(f"\nSpearman Correlation: {corr:.4f} (p={p_val:.4g})")
        
        # 3. 计算 Top K (5000) Overlap
        # Largest (Top Loss?)
        top_count, top_prob = calculate_topk_overlap(ids1, ids2, scores1, scores2, CUSTOM_TOPK, is_ascending=False)
        # Smallest (Bottom Loss?)
        bot_count, bot_prob = calculate_topk_overlap(ids1, ids2, scores1, scores2, CUSTOM_TOPK, is_ascending=True)
        
        print(f"\nTop {CUSTOM_TOPK} Overlap (Highest Values): {top_count} ({top_prob:.2%})")
        print(f"Bottom {CUSTOM_TOPK} Overlap (Lowest Values): {bot_count} ({bot_prob:.2%})")
        
        # 4. 绘制直方图 PDF
        plot_topk_histogram_pdf(CUSTOM_TOPK, top_count, top_prob, bot_count, bot_prob, 
                                filename="overlap_histogram.pdf")
        
        # 5. 绘制曲线图 PDF
        plot_overlap_curve_pdf(ids1, ids2, scores1, scores2, 
                               filename="overlap_curve.pdf")
        
        print(f"\nAll tasks completed. Results saved in {OUTPUT_DIR}")

    except Exception as e:
        print(f"Error: {e}")