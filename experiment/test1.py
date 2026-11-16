import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

# ===================== 全局配置 =====================
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 8
plt.rcParams['axes.linewidth'] = 0.7
plt.rcParams['figure.dpi'] = 600
plt.rcParams['pdf.fonttype'] = 42

X_LABELS = ['50', '100', '200', '1000', '4000', '6000']
Y_LABELS = ['50', '100', '200', '500', '1000', '2000', '4000', '6000']
NX, NY = len(X_LABELS), len(Y_LABELS)

COLOR_SCALE = {
    'compute':  ('#1f5fa8', '#e8f0f8'),
    'memory':   ('#c82620', '#f8e8e8'),
    'capacity': ('#238a38', '#e8f5ea'),
    'missing':  '#f0f0f0'
}

# ===================== 颜色插值 =====================
def lerp_color(deep_hex, light_hex, ratio):
    d = [int(deep_hex[1:3],16), int(deep_hex[3:5],16), int(deep_hex[5:7],16)]
    l = [int(light_hex[1:3],16), int(light_hex[3:5],16), int(light_hex[5:7],16)]
    r, g, b = [int(l[i] + (d[i]-l[i]) * ratio) for i in range(3)]
    return f'#{r:02x}{g:02x}{b:02x}'

# ===================== 数据生成 =====================
def generate_grid_data(qps, scenario):
    dominant = np.full((NY, NX), scenario, dtype=object)
    ratio = np.ones((NY, NX)) * 0.85

    missing_mask = np.zeros((NY, NX), dtype=bool)
    missing_mask[7, 2] = True   # input=200, output=6000: no data
    missing_mask[7, 3] = True   # input=1000, output=6000: no data
    missing_mask[6, 3] = True   # input=1000, output=4000: no data
    missing_mask[7, 4] = True   # input=4000, output=6000: no data
    missing_mask[6, 4] = True   # input=4000, output=4000: no data
    missing_mask[4:, 5] = True  # input=6000, output>=1000: no data
    # QPS=4 only
    if qps == 4:
        missing_mask[:, 5] = True   # input=6000: entire column
        missing_mask[5, 4] = True   # input=4000, output=2000
    dominant[missing_mask] = 'missing'
    ratio[missing_mask] = 0

    if qps == 1:
        compute_mask = np.fromfunction(lambda j,i: (i<=2) & (j<=3), (NY,NX))
    elif qps == 2:
        compute_mask = np.fromfunction(lambda j,i: (i<=2) & (j<=2), (NY,NX))
    else:
        compute_mask = np.fromfunction(lambda j,i: (i<=1) & (j<=2), (NY,NX))

    compute_mask = compute_mask & ~missing_mask
    dominant[compute_mask] = 'compute'

    for j in range(NY):
        for i in range(NX):
            if compute_mask[j,i]:
                ratio[j,i] = 0.5 + 0.5 * (1 - (i/2 + j/3)/2)

    for j in range(NY):
        for i in range(NX):
            if not compute_mask[j,i] and not missing_mask[j,i]:
                ratio[j,i] = 0.6 + 0.4 * min(i/5, j/7)

    if scenario == 'memory':
        dominant[4, 1] = 'compute'
        ratio[4, 1] = 0.9

    dominant[0, 0] = 'compute'
    dominant[0, 2] = 'compute'
    ratio[0, 0] = 0.7
    ratio[0, 2] = ratio[1, 2]

    return dominant, ratio

# ===================== 单张子图 =====================
def draw_grid(ax, dominant, ratio, show_xlabel=False, show_ylabel=False):
    ax.set_xlim(0, NX)
    ax.set_ylim(0, NY)
    ax.set_aspect('equal')
    ax.spines[['top', 'right']].set_visible(False)

    ax.set_xticks(np.arange(NX) + 0.5)
    ax.set_yticks(np.arange(NY) + 0.5)
    ax.set_xticklabels(X_LABELS, fontsize=5.5, rotation=45, ha='right')
    ax.set_yticklabels(Y_LABELS, fontsize=5.5)

    if show_xlabel:
        ax.set_xlabel('Input Length', fontsize=7)
    if show_ylabel:
        ax.set_ylabel('Output Length', fontsize=7)

    for j in range(NY):
        for i in range(NX):
            metric = dominant[j, i]
            if metric == 'missing':
                facecolor = COLOR_SCALE['missing']
            else:
                deep, light = COLOR_SCALE[metric]
                facecolor = lerp_color(deep, light, ratio[j, i])
            
            rect = Rectangle((i, j), 1, 1,
                           facecolor=facecolor, edgecolor='none')
            ax.add_patch(rect)

# ===================== 右侧三色热力条（数据边界精确对齐 + 文字完整显示） =====================
def add_colorbar(fig, ax_top, ax_bot):
    fig.canvas.draw()
    
    # 获取数据区域真实上下边界
    top_corners = ax_top.transData.transform([[0, 0], [0, NY]])
    top_fig = fig.transFigure.inverted().transform(top_corners)
    y_data_top = top_fig[1, 1]
    
    bot_corners = ax_bot.transData.transform([[0, 0], [0, NY]])
    bot_fig = fig.transFigure.inverted().transform(bot_corners)
    y_data_bot = bot_fig[0, 1]
    
    bar_x = 0.90       # 色条左边界
    bar_w = 0.018      # 色条宽度
    
    total_h = y_data_top - y_data_bot
    bar_h = total_h / 3
    
    specs = [
        ('compute',  y_data_top),
        ('memory',   y_data_top - bar_h),
        ('capacity', y_data_top - 2 * bar_h),
    ]

    for metric, y_top_bar in specs:
        y0 = y_top_bar - bar_h
        deep, light = COLOR_SCALE[metric]
        steps = 80
        for k in range(steps):
            t = 1 - k / steps
            y_pos = y_top_bar - bar_h * k / steps
            c = lerp_color(deep, light, t)
            rect = Rectangle((bar_x, y_pos - bar_h/steps),
                           bar_w, bar_h/steps + 0.0005,
                           facecolor=c, edgecolor='none',
                           transform=fig.transFigure)
            fig.patches.append(rect)
        border = Rectangle((bar_x, y0), bar_w, bar_h,
                          facecolor='none', edgecolor='#888', linewidth=0.5,
                          transform=fig.transFigure)
        fig.patches.append(border)
        # 文字靠右预留足够空间，确保完整显示
        fig.text(bar_x + bar_w + 0.012, y0 + bar_h/2, metric,
                va='center', fontsize=6, transform=fig.transFigure)

    # 刻度
    fig.text(bar_x - 0.006, y_data_top, '1',
            ha='right', va='center', fontsize=6, transform=fig.transFigure)
    fig.text(bar_x - 0.006, y_data_bot, '0',
            ha='right', va='center', fontsize=6, transform=fig.transFigure)

# ===================== 主程序 =====================
fig, axes = plt.subplots(2, 3, figsize=(8.5/2.54, 9.5/2.54))
QPS_LIST = [1, 2, 4]
SCENARIOS = ['capacity', 'memory']

for row_idx, scenario in enumerate(SCENARIOS):
    for col_idx, qps in enumerate(QPS_LIST):
        ax = axes[row_idx, col_idx]
        dom, rat = generate_grid_data(qps, scenario)
        draw_grid(ax, dom, rat,
                 show_xlabel=(row_idx == 1),
                 show_ylabel=(col_idx == 0))
        ax.set_title(f'QPS = {qps}', fontsize=7, pad=3)

# 上下间距进一步缩小，右侧预留足够空间给热力条文字
plt.subplots_adjust(
    left=0.10, right=0.84,
    top=0.96, bottom=0.16,
    hspace=0.08, wspace=0.45
)

add_colorbar(fig, axes[0, 2], axes[1, 2])

plt.savefig('qps_grid_aligned_tight.pdf')
plt.savefig('qps_grid_aligned_tight.png', dpi=600)
print('Saved qps_grid_aligned_tight.pdf')