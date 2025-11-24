"""
分析 blank_ratio 高斯分布的概率分布
展示不同 std_ratio 值下的概率落在不同范围的情况
"""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib
from typing import Tuple
import platform

# 配置 matplotlib 中文字体
def setup_chinese_font():
    """设置 matplotlib 中文字体"""
    from matplotlib import font_manager
    
    system = platform.system()
    if system == 'Windows':
        # Windows 系统常用中文字体
        fonts = ['Microsoft YaHei', 'SimHei', 'SimSun', 'KaiTi', 'FangSong']
    elif system == 'Darwin':  # macOS
        fonts = ['PingFang SC', 'STHeiti', 'Arial Unicode MS', 'Heiti SC']
    else:  # Linux
        fonts = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'Noto Sans CJK SC', 'Droid Sans Fallback']
    
    # 获取所有可用字体
    available_fonts = [f.name for f in font_manager.fontManager.ttflist]
    
    # 尝试找到可用的中文字体
    for font in fonts:
        if font in available_fonts:
            try:
                plt.rcParams['font.sans-serif'] = [font] + plt.rcParams['font.sans-serif']
                plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
                # 测试字体是否可用
                fig, ax = plt.subplots(figsize=(1, 1))
                ax.text(0.5, 0.5, '测试', fontsize=12, fontfamily=font)
                plt.close(fig)
                print(f"Using Chinese font: {font}")
                return font
            except Exception as e:
                continue
    
    # 如果所有字体都不可用，使用英文标签
    print("Warning: No Chinese font found, using English labels")
    return None

# 初始化字体
_chinese_font = setup_chinese_font()


def calculate_probability_distribution(
    base: float,
    randmax: float,
    std_ratio: float,
    n_samples: int = 1000000
) -> dict:
    """
    计算截断高斯分布的概率统计
    
    Args:
        base: blank_ratio_base
        randmax: blank_ratio_randmax
        std_ratio: blank_ratio_std_ratio
        n_samples: 采样数量（用于验证）
    
    Returns:
        包含概率统计的字典
    """
    # 计算标准差
    sigma = randmax * std_ratio
    
    # 截断范围
    a = base  # 下界
    b = base + randmax  # 上界
    
    # 使用 scipy 的截断正态分布
    # 标准化到标准正态分布
    a_norm = (a - base) / sigma if sigma > 0 else -np.inf
    b_norm = (b - base) / sigma if sigma > 0 else np.inf
    
    # 创建截断正态分布
    trunc_norm = stats.truncnorm(a=a_norm, b=b_norm, loc=base, scale=sigma)
    
    # 计算不同区间的概率
    # 区间1: [base, base + randmax/3]
    # 区间2: [base + randmax/3, base + 2*randmax/3]
    # 区间3: [base + 2*randmax/3, base + randmax]
    
    range1_end = base + randmax / 3.0
    range2_end = base + 2.0 * randmax / 3.0
    
    prob_range1 = trunc_norm.cdf(range1_end) - trunc_norm.cdf(base)
    prob_range2 = trunc_norm.cdf(range2_end) - trunc_norm.cdf(range1_end)
    prob_range3 = trunc_norm.cdf(b) - trunc_norm.cdf(range2_end)
    
    # 计算 ±1σ, ±2σ, ±3σ 的概率（考虑截断）
    # 对于截断分布，我们需要计算实际落在这些范围内的概率
    sigma1_low = max(base, base - sigma)
    sigma1_high = min(b, base + sigma)
    prob_1sigma = trunc_norm.cdf(sigma1_high) - trunc_norm.cdf(sigma1_low) if sigma1_high > sigma1_low else 0
    
    sigma2_low = max(base, base - 2*sigma)
    sigma2_high = min(b, base + 2*sigma)
    prob_2sigma = trunc_norm.cdf(sigma2_high) - trunc_norm.cdf(sigma2_low) if sigma2_high > sigma2_low else 0
    
    sigma3_low = max(base, base - 3*sigma)
    sigma3_high = min(b, base + 3*sigma)
    prob_3sigma = trunc_norm.cdf(sigma3_high) - trunc_norm.cdf(sigma3_low) if sigma3_high > sigma3_low else 0
    
    # 采样验证
    samples = trunc_norm.rvs(n_samples)
    
    # 计算中位数和均值
    median = trunc_norm.median()
    mean = trunc_norm.mean()
    
    return {
        'base': base,
        'randmax': randmax,
        'std_ratio': std_ratio,
        'sigma': sigma,
        'range': (a, b),
        'mean': mean,
        'median': median,
        'prob_range1': prob_range1,  # [base, base+randmax/3]
        'prob_range2': prob_range2,  # [base+randmax/3, base+2*randmax/3]
        'prob_range3': prob_range3,  # [base+2*randmax/3, base+randmax]
        'prob_1sigma': prob_1sigma,
        'prob_2sigma': prob_2sigma,
        'prob_3sigma': prob_3sigma,
        'samples': samples,
    }


def print_statistics(stats_dict: dict) -> None:
    """打印统计信息"""
    print(f"\n{'='*60}")
    print(f"blank_ratio 分布统计")
    print(f"{'='*60}")
    print(f"基础值 (base):           {stats_dict['base']:.2f}%")
    print(f"随机范围 (randmax):      {stats_dict['randmax']:.2f}%")
    print(f"标准差比例 (std_ratio):  {stats_dict['std_ratio']:.3f}")
    print(f"实际标准差 (sigma):     {stats_dict['sigma']:.3f}%")
    print(f"分布范围:                [{stats_dict['range'][0]:.2f}%, {stats_dict['range'][1]:.2f}%]")
    print(f"均值:                    {stats_dict['mean']:.3f}%")
    print(f"中位数:                  {stats_dict['median']:.3f}%")
    
    print(f"\n概率分布（三等分区间）:")
    print(f"  [{stats_dict['base']:.2f}%, {stats_dict['base'] + stats_dict['randmax']/3:.2f}%]: "
          f"{stats_dict['prob_range1']*100:.2f}%")
    print(f"  [{stats_dict['base'] + stats_dict['randmax']/3:.2f}%, "
          f"{stats_dict['base'] + 2*stats_dict['randmax']/3:.2f}%]: "
          f"{stats_dict['prob_range2']*100:.2f}%")
    print(f"  [{stats_dict['base'] + 2*stats_dict['randmax']/3:.2f}%, "
          f"{stats_dict['base'] + stats_dict['randmax']:.2f}%]: "
          f"{stats_dict['prob_range3']*100:.2f}%")
    
    print(f"\n概率分布（标准差区间，考虑截断）:")
    sigma = stats_dict['sigma']
    base = stats_dict['base']
    randmax = stats_dict['randmax']
    
    # 计算实际范围
    range_1sigma = (max(base, base - sigma), min(base + randmax, base + sigma))
    range_2sigma = (max(base, base - 2*sigma), min(base + randmax, base + 2*sigma))
    range_3sigma = (max(base, base - 3*sigma), min(base + randmax, base + 3*sigma))
    
    print(f"  ±1σ [{range_1sigma[0]:.2f}%, {range_1sigma[1]:.2f}%]: "
          f"{stats_dict['prob_1sigma']*100:.2f}%")
    print(f"  ±2σ [{range_2sigma[0]:.2f}%, {range_2sigma[1]:.2f}%]: "
          f"{stats_dict['prob_2sigma']*100:.2f}%")
    print(f"  ±3σ [{range_3sigma[0]:.2f}%, {range_3sigma[1]:.2f}%]: "
          f"{stats_dict['prob_3sigma']*100:.2f}%")


def plot_distribution(stats_dict: dict, ax=None) -> None:
    """绘制分布图"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # 根据字体可用性选择标签语言
    use_chinese = _chinese_font is not None
    
    # 确保使用设置的字体
    if _chinese_font:
        plt.rcParams['font.sans-serif'] = [_chinese_font] + plt.rcParams['font.sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
    
    samples = stats_dict['samples']
    base = stats_dict['base']
    randmax = stats_dict['randmax']
    sigma = stats_dict['sigma']
    
    # 绘制直方图
    bins = np.linspace(base, base + randmax, 50)
    hist_label = 'Sampling Distribution' if not use_chinese else '采样分布'
    ax.hist(samples, bins=bins, density=True, alpha=0.6, color='skyblue', edgecolor='black', label=hist_label)
    
    # 绘制理论PDF
    x = np.linspace(base, base + randmax, 200)
    a_norm = (base - base) / sigma if sigma > 0 else -np.inf
    b_norm = (base + randmax - base) / sigma if sigma > 0 else np.inf
    trunc_norm = stats.truncnorm(a=a_norm, b=b_norm, loc=base, scale=sigma)
    pdf = trunc_norm.pdf(x)
    pdf_label = 'Theoretical PDF' if not use_chinese else '理论PDF'
    ax.plot(x, pdf, 'r-', lw=2, label=pdf_label)
    
    # 标记关键点
    ax.axvline(base, color='green', linestyle='--', lw=2, label=f'base = {base:.2f}%')
    ax.axvline(base + randmax, color='orange', linestyle='--', lw=2, label=f'max = {base + randmax:.2f}%')
    mean_label = f'mean = {stats_dict["mean"]:.2f}%' if not use_chinese else f'均值 = {stats_dict["mean"]:.2f}%'
    ax.axvline(stats_dict['mean'], color='red', linestyle=':', lw=2, label=mean_label)
    
    # 标记区间
    range1_end = base + randmax / 3.0
    range2_end = base + 2.0 * randmax / 3.0
    range1_label = f'Range1: {stats_dict["prob_range1"]*100:.1f}%' if not use_chinese else f'区间1: {stats_dict["prob_range1"]*100:.1f}%'
    range2_label = f'Range2: {stats_dict["prob_range2"]*100:.1f}%' if not use_chinese else f'区间2: {stats_dict["prob_range2"]*100:.1f}%'
    range3_label = f'Range3: {stats_dict["prob_range3"]*100:.1f}%' if not use_chinese else f'区间3: {stats_dict["prob_range3"]*100:.1f}%'
    ax.axvspan(base, range1_end, alpha=0.2, color='green', label=range1_label)
    ax.axvspan(range1_end, range2_end, alpha=0.2, color='yellow', label=range2_label)
    ax.axvspan(range2_end, base + randmax, alpha=0.2, color='red', label=range3_label)
    
    xlabel = 'Blank Ratio (%)' if not use_chinese else '空白比例 (%)'
    ylabel = 'Probability Density' if not use_chinese else '概率密度'
    title = f'blank_ratio Distribution (base={base:.1f}%, randmax={randmax:.1f}%, std_ratio={stats_dict["std_ratio"]:.3f})' if not use_chinese else f'blank_ratio 分布 (base={base:.1f}%, randmax={randmax:.1f}%, std_ratio={stats_dict["std_ratio"]:.3f})'
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)


def compare_std_ratios(base: float, randmax: float, std_ratios: list) -> None:
    """比较不同 std_ratio 值的分布"""
    n_ratios = len(std_ratios)
    fig, axes = plt.subplots(1, n_ratios, figsize=(5*n_ratios, 6))
    if n_ratios == 1:
        axes = [axes]
    
    for i, std_ratio in enumerate(std_ratios):
        stats_dict = calculate_probability_distribution(base, randmax, std_ratio)
        plot_distribution(stats_dict, ax=axes[i])
    
    plt.tight_layout()
    plt.savefig('blank_ratio_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\n图表已保存为: blank_ratio_comparison.png")


def main():
    """主函数"""
    import argparse
    parser = argparse.ArgumentParser(description='分析 blank_ratio 高斯分布')
    parser.add_argument('--base', type=float, default=0.0, help='blank_ratio_base (默认: 0.0)')
    parser.add_argument('--randmax', type=float, default=3.0, help='blank_ratio_randmax (默认: 3.0)')
    parser.add_argument('--std_ratio', type=float, default=0.33, help='blank_ratio_std_ratio (默认: 0.33)')
    parser.add_argument('--compare', action='store_true', help='比较多个 std_ratio 值')
    args = parser.parse_args()
    
    if args.compare:
        # 比较多个 std_ratio 值
        std_ratios = [0.2, 0.33, 0.5]
        print("比较不同 std_ratio 值的分布:")
        for std_ratio in std_ratios:
            stats_dict = calculate_probability_distribution(args.base, args.randmax, std_ratio)
            print_statistics(stats_dict)
        compare_std_ratios(args.base, args.randmax, std_ratios)
    else:
        # 单个分析
        stats_dict = calculate_probability_distribution(args.base, args.randmax, args.std_ratio)
        print_statistics(stats_dict)
        
        # 绘制图表
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_distribution(stats_dict, ax=ax)
        plt.tight_layout()
        plt.savefig('blank_ratio_distribution.png', dpi=150, bbox_inches='tight')
        print(f"\n图表已保存为: blank_ratio_distribution.png")
        plt.show()


if __name__ == '__main__':
    main()

