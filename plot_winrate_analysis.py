#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
绘制胜率分析图表

生成多维度胜率分析的可视化图表
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def plot_winrate_analysis():
    """绘制胜率分析图表"""
    
    # 读取详细数据
    df = pd.read_csv('winrate_analysis_detail.csv')
    
    # 创建大图
    fig = plt.figure(figsize=(20, 12))
    
    # ===== 1. DTE分析 =====
    ax1 = plt.subplot(2, 3, 1)
    dte_stats = df.groupby('dte_cat').agg({
        'win': ['count', 'sum', 'mean'],
        'pnl_pct': 'mean'
    })
    dte_stats.columns = ['count', 'wins', 'winrate', 'avg_pnl']
    dte_stats = dte_stats.sort_values('count', ascending=False)
    
    x = range(len(dte_stats))
    ax1.bar(x, dte_stats['winrate'], alpha=0.7, label='胜率')
    ax1.set_xticks(x)
    ax1.set_xticklabels(dte_stats.index, rotation=45, ha='right')
    ax1.set_ylabel('胜率', fontsize=12)
    ax1.set_title('DTE（到期天数）vs 胜率', fontsize=14, fontweight='bold')
    ax1.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='50%基准线')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # 添加交易数标注
    for i, (idx, row) in enumerate(dte_stats.iterrows()):
        ax1.text(i, row['winrate'] + 0.01, f"{int(row['count'])}笔", 
                ha='center', va='bottom', fontsize=9)
    
    # ===== 2. OTM分析 =====
    ax2 = plt.subplot(2, 3, 2)
    otm_stats = df.groupby('otm_pct_cat').agg({
        'win': ['count', 'sum', 'mean'],
        'pnl_pct': 'mean'
    })
    otm_stats.columns = ['count', 'wins', 'winrate', 'avg_pnl']
    otm_stats = otm_stats.sort_values('count', ascending=False)
    
    x = range(len(otm_stats))
    ax2.bar(x, otm_stats['winrate'], alpha=0.7, color='orange', label='胜率')
    ax2.set_xticks(x)
    ax2.set_xticklabels(otm_stats.index, rotation=45, ha='right')
    ax2.set_ylabel('胜率', fontsize=12)
    ax2.set_title('OTM比例 vs 胜率', fontsize=14, fontweight='bold')
    ax2.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='50%基准线')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    for i, (idx, row) in enumerate(otm_stats.iterrows()):
        ax2.text(i, row['winrate'] + 0.01, f"{int(row['count'])}笔", 
                ha='center', va='bottom', fontsize=9)
    
    # ===== 3. 交易时间分析 =====
    ax3 = plt.subplot(2, 3, 3)
    time_stats = df.groupby('trade_hour_cat').agg({
        'win': ['count', 'sum', 'mean'],
        'pnl_pct': 'mean'
    })
    time_stats.columns = ['count', 'wins', 'winrate', 'avg_pnl']
    time_stats = time_stats.sort_values('count', ascending=False)
    
    x = range(len(time_stats))
    ax3.bar(x, time_stats['winrate'], alpha=0.7, color='green', label='胜率')
    ax3.set_xticks(x)
    ax3.set_xticklabels(time_stats.index, rotation=45, ha='right')
    ax3.set_ylabel('胜率', fontsize=12)
    ax3.set_title('交易时间段 vs 胜率', fontsize=14, fontweight='bold')
    ax3.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='50%基准线')
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    for i, (idx, row) in enumerate(time_stats.iterrows()):
        ax3.text(i, row['winrate'] + 0.01, f"{int(row['count'])}笔", 
                ha='center', va='bottom', fontsize=9)
    
    # ===== 4. 流动性分析 =====
    ax4 = plt.subplot(2, 3, 4)
    share_stats = df.groupby('share_eqv_cat').agg({
        'win': ['count', 'sum', 'mean'],
        'pnl_pct': 'mean'
    })
    share_stats.columns = ['count', 'wins', 'winrate', 'avg_pnl']
    share_stats = share_stats.sort_values('count', ascending=False)
    
    x = range(len(share_stats))
    ax4.bar(x, share_stats['winrate'], alpha=0.7, color='purple', label='胜率')
    ax4.set_xticks(x)
    ax4.set_xticklabels(share_stats.index, rotation=45, ha='right')
    ax4.set_ylabel('胜率', fontsize=12)
    ax4.set_title('流动性（Share_eqv）vs 胜率', fontsize=14, fontweight='bold')
    ax4.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='50%基准线')
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)
    
    for i, (idx, row) in enumerate(share_stats.iterrows()):
        ax4.text(i, row['winrate'] + 0.01, f"{int(row['count'])}笔", 
                ha='center', va='bottom', fontsize=9)
    
    # ===== 5. 平均盈亏对比 =====
    ax5 = plt.subplot(2, 3, 5)
    all_stats = pd.DataFrame({
        'DTE': dte_stats['avg_pnl'].sort_index(),
        'OTM': otm_stats['avg_pnl'].sort_index(),
        'Time': time_stats['avg_pnl'].sort_index(),
        'Share_eqv': share_stats['avg_pnl'].sort_index()
    })
    
    # 绘制热力图
    dte_avg = dte_stats['avg_pnl'].values
    otm_avg = otm_stats['avg_pnl'].values
    time_avg = time_stats['avg_pnl'].values
    share_avg = share_stats['avg_pnl'].values
    
    categories = ['DTE', 'OTM', 'Time', 'Share_eqv']
    avg_pnl_by_cat = [dte_avg.mean(), otm_avg.mean(), time_avg.mean(), share_avg.mean()]
    
    bars = ax5.bar(categories, avg_pnl_by_cat, alpha=0.7, color=['blue', 'orange', 'green', 'purple'])
    ax5.set_ylabel('平均盈亏%', fontsize=12)
    ax5.set_title('各维度平均盈亏对比', fontsize=14, fontweight='bold')
    ax5.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax5.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, avg_pnl_by_cat):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:+.2%}', ha='center', va='bottom' if val > 0 else 'top', fontsize=10)
    
    # ===== 6. 综合评分（胜率 × 样本数） =====
    ax6 = plt.subplot(2, 3, 6)
    
    # 计算综合评分 = 胜率 × log(样本数)
    dte_score = dte_stats['winrate'] * np.log1p(dte_stats['count'])
    otm_score = otm_stats['winrate'] * np.log1p(otm_stats['count'])
    time_score = time_stats['winrate'] * np.log1p(time_stats['count'])
    share_score = share_stats['winrate'] * np.log1p(share_stats['count'])
    
    # 找出每个维度的最优配置
    best_dte = dte_stats.loc[dte_score.idxmax()]
    best_otm = otm_stats.loc[otm_score.idxmax()]
    best_time = time_stats.loc[time_score.idxmax()]
    best_share = share_stats.loc[share_score.idxmax()]
    
    # 绘制表格
    table_data = [
        ['DTE', dte_score.idxmax(), f"{best_dte['winrate']:.1%}", f"{int(best_dte['count'])}"],
        ['OTM', otm_score.idxmax(), f"{best_otm['winrate']:.1%}", f"{int(best_otm['count'])}"],
        ['Time', time_score.idxmax(), f"{best_time['winrate']:.1%}", f"{int(best_time['count'])}"],
        ['Share_eqv', share_score.idxmax(), f"{best_share['winrate']:.1%}", f"{int(best_share['count'])}"]
    ]
    
    table = ax6.table(cellText=table_data,
                     colLabels=['维度', '最优配置', '胜率', '样本数'],
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.2, 0.35, 0.2, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # 设置表头样式
    for i in range(4):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # 设置行颜色
    colors = ['#E3F2FD', '#FFE0B2', '#C8E6C9', '#E1BEE7']
    for i in range(1, 5):
        for j in range(4):
            table[(i, j)].set_facecolor(colors[i-1])
    
    ax6.axis('off')
    ax6.set_title('最优配置总结', fontsize=14, fontweight='bold', pad=20)
    
    # 总标题
    fig.suptitle('回测结果多维度胜率分析 - 可视化报告', 
                fontsize=18, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # 保存图表
    output_file = 'winrate_analysis_chart.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✅ 图表已保存到: {output_file}")
    
    # 显示图表
    # plt.show()  # 如果需要显示，取消注释

if __name__ == '__main__':
    print("\n" + "="*80)
    print("📊 生成胜率分析可视化图表")
    print("="*80 + "\n")
    
    plot_winrate_analysis()
    
    print("\n" + "="*80)
    print("✅ 完成！")
    print("="*80 + "\n")

