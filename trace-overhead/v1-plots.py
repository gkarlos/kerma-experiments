import os, sys
import statistics as st
import matplotlib.pyplot as plt
import numpy as np

DATA = {
    "rodinia" : {
        "backprop": {
            "runtimes"       : [0.445 + 0.228, 0.480 + 0.203, 0.456 + 0.218, 0.456 + 0.228, 0.452 + 0.226],
            "block_runtimes" : [0.712 + 0.219, 0.652 + 0.248, 0.637 + 0.240, 0.687 + 0.265, 0.655 + 0.207],
            "warp_runtimes"  : [0.677 + 0.219, 0.707 + 0.204, 0.691 + 0.232, 0.664 + 0.197, 0.629 + 0.225],
            "size": 663424,
            "block_size": 897304,
            "warp_size": 897304
        },
        "bfs": {
            "runtimes"       : [0.103 + 0.111, 0.081 + 0.105, 0.101 + 0.132, 0.100 + 0.113, 0.092 + 0.129],
            "block_runtimes" : [0.117 + 0.165, 0.131 + 0.135, 0.114 + 0.115, 0.115 + 0.107, 0.126 + 0.113],
            "warp_runtimes"  : [0.132 + 0.136, 0.142 + 0.125, 0.121 + 0.125, 0.121 + 0.108, 0.122 + 0.113],
            "size": 646144,
            "block_size": 851448,
            "warp_size": 855544
        },
        "cfd": {
            "runtimes"       : [17.758 + 7.3580, 18.110 + 7.5910, 17.765 + 7.7620, 17.794 + 7.4680, 18.158 + 7.4000],
            "block_runtimes" : [31.063 + 12.503, 31.186 + 12.611, 31.781 + 12.129, 31.870 + 12.416, 31.304 + 12.258],
            "warp_runtimes"  : [31.284 + 12.373, 32.253 + 12.990, 31.200 + 12.717, 31.294 + 12.768, 31.416 + 12.347],
            "size": 737000,
            "block_size": 1078544,
            "warp_size": 1082640
        },
        "gaussian": {
            "runtimes"       : [0.013 + 0.160, 0.016 + 0.134, 0.017 + 0.125, 0.012 + 0.145, 0.013 + 0.148],
            "block_runtimes" : [2.966 + 0.117, 2.907 + 0.124, 2.954 + 0.120, 2.903 + 0.112, 3.063 + 0.128],
            "warp_runtimes"  : [2.950 + 0.120, 2.902 + 0.129, 2.906 + 0.124, 2.993 + 0.165, 3.014 + 0.104],
            "size": 654920,
            "block_size": 860928,
            "warp_size": 865024
        }
    }
}

PLOTS_DIR="plots"
SUITES=['rodinia']

pre_wd = os.getcwd()
os.chdir(os.path.dirname(os.path.abspath(__file__)))
cwd = os.getcwd()
plot_dir = os.path.join(cwd, PLOTS_DIR)
if not os.path.exists(plot_dir):
    os.mkdir(plot_dir)



for suite in SUITES:
    names = []
    rt_block_overheads = []
    rt_warp_overheads  = []
    sz_block_overheads = []
    sz_warp_overheads  = []

    for bench in DATA[suite]:
        data = DATA[suite][bench]
        names.append(bench)
        rt_block_overheads.append(st.mean(data['block_runtimes']) / st.mean(data['runtimes']))
        rt_warp_overheads.append(st.mean(data['warp_runtimes']) / st.mean(data['runtimes']))
        sz_block_overheads.append(data['block_size'] / data['size'])
        sz_warp_overheads.append(data['warp_size'] / data['size'])

    width = 0.35

    rt_plot, rt_ax = plt.subplots(figsize=(max(6,len(names) * 0.45),5))
    rt_x_axis = np.arange(0, len(names), 1)

    sz_plot, sz_ax = plt.subplots(figsize=(max(6,len(names) * 0.45),5))
    sz_x_axis = np.arange(0, len(names), 1)

    rt_warp_bars = rt_ax.bar(rt_x_axis, rt_warp_overheads, width, color=(0.2, 0.8, 0.6, 1))
    rt_block_bars = rt_ax.bar(rt_x_axis + width, rt_block_overheads, width, color=(0.2, 0.4, 0.6, 1))
    rt_ax.tick_params(axis='x', rotation=90)
    rt_ax.set_xticks(rt_x_axis + width / 2)
    rt_ax.set_xticklabels(names)
    rt_ax.set_ylim(ymin=1);

    sz_warp_bars = sz_ax.bar(sz_x_axis, sz_warp_overheads, width, color=(0.2, 0.8, 0.6, 1))
    sz_block_bars = sz_ax.bar(sz_x_axis + width, sz_block_overheads, width,  color=(0.2, 0.4, 0.6, 1))
    sz_ax.tick_params(axis='x', rotation=90)
    sz_ax.set_xticks(sz_x_axis + width / 2)
    sz_ax.set_xticklabels(names)
    sz_ax.set_ylim(ymin=1);

    rt_ax.set_ylabel("overhead")
    rt_ax.set_title("Runtime overhead of tracing in " + suite + " benchmarks (user+sys)");
    rt_ax.legend(['warp_mode', 'block_mode'], loc='upper left')

    rt_plot.tight_layout()
    rt_plot.savefig(os.path.join(plot_dir, suite + '_runtime_overhead.png'))

    sz_ax.set_ylabel("overhead")
    sz_ax.set_title("Size overhead of tracing in " + suite + " benchmarks");
    sz_ax.legend(['warp_mode', 'block_mode'], loc='upper left')

    sz_plot.tight_layout()
    sz_plot.savefig(os.path.join(plot_dir, suite + '_size_overhead.png'))

    print(names)
    print(rt_block_overheads)
    print(rt_warp_overheads)
    print(sz_block_overheads)
    print(sz_warp_overheads)
