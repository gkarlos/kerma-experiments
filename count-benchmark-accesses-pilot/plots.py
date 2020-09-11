import os, sys
import matplotlib.pyplot as plt
import json
import pprint
import numpy as np

VERBOSE="-v" in sys.argv
MANUAL_FILE="manual.json"
PLOTS_DIR="plots"
SUITES=['rodinia', 'polybench']

def msg(*args, **kwargs):
    if VERBOSE:
        print(*args, **kwargs)

pre_wd = os.getcwd()
os.chdir(os.path.dirname(os.path.abspath(__file__)))
cwd = os.getcwd()
plot_dir = os.path.join(cwd, PLOTS_DIR)
if not os.path.exists(plot_dir):
    os.mkdir(plot_dir)

data={
    "rodinia": {
        "manual": {
            "all": {
                "x-axis": [],
                'values': {
                    'all': [],
                    "static": [],
                    "dynamic": [],
                    "inloop": {
                        "static": [],
                        "dynamic": []
                    }
                }
            },
            "kern": { }
        }
    },
    "polybench": {
        "manual": {
            "all": {
                "x-axis": [],
                'values': {
                    'all': [],
                    "static": [],
                    "dynamic": [],
                    "inloop": {
                        "static": [],
                        "dynamic": []
                    }
                }
            },
            "kern": { }
        }
    }
}

### Read the data
for suite in SUITES:
    suite_dir = os.path.join(cwd, suite)
    for bench in sorted(os.listdir(suite_dir)):
       datafile_manual = os.path.join(suite_dir, bench, MANUAL_FILE)
       with open(datafile_manual) as f:
           json_data = json.load(f)
           for version_data in json_data:
               version_name = bench if not len(version_data['version']) else bench + '_' + version_data['version']
               data[suite]['manual']['all']['x-axis'].append(version_name)
               data[suite]['manual']['kern'][version_name] = {}

               sum_all, sum_static, sum_inloop_static, sum_inloop_dynamic = 0,0,0,0

               for kernel in version_data['data']:
                   sum_all += version_data['data'][kernel]['all']
                   sum_static += version_data['data'][kernel]['static']
                   sum_inloop_static += version_data['data'][kernel]['inloop']['static']
                   sum_inloop_dynamic += version_data['data'][kernel]['inloop']['dynamic']

                   data[suite]['manual']['kern'][version_name][kernel] = [version_data['data'][kernel]['all'],\
                                                                          version_data['data'][kernel]['static'],\
                                                                          version_data['data'][kernel]['inloop']['static'],\
                                                                          version_data['data'][kernel]['inloop']['dynamic']]

               data[suite]['manual']['all']['values']['all'].append(sum_all)
               data[suite]['manual']['all']['values']['static'].append(sum_static)
               data[suite]['manual']['all']['values']['dynamic'].append(sum_all - sum_static)
               data[suite]['manual']['all']['values']['inloop']['static'].append(sum_inloop_static)
               data[suite]['manual']['all']['values']['inloop']['dynamic'].append(sum_inloop_dynamic)


###
### Per-kernel plot
###
msg("+ Build Per-kernel plot")
for suite in SUITES:
    suite_data = data[suite]
    # suite_data = data['rodinia']
    suite_kernel_plot, ax = plt.subplots(figsize=(max(6,len(suite_data['manual']['kern']) * 0.45),5))

    x_axis_groups = []
    data_static_percent = []
    data_inloop_static_percent = []
    data_dynamic_percent = []
    data_inloop_dynamic_percent = []

    for version in suite_data['manual']['kern']:
        x_axis_groups.append(list(suite_data['manual']['kern'][version].keys()))
        for kernel in suite_data['manual']['kern'][version]:
            kernel_data = suite_data['manual']['kern'][version][kernel]
            data_static_percent.append(kernel_data[1] / kernel_data[0])
            data_inloop_static_percent.append(kernel_data[2] / kernel_data[0])
            data_dynamic_percent.append((kernel_data[0] - kernel_data[1]) / kernel_data[0])
            data_inloop_dynamic_percent.append(kernel_data[3] / kernel_data[0])

    x_axis_labels = [kernel for kernels in x_axis_groups for kernel in kernels]
    x_axis = np.arange(0,len(x_axis_labels),1)

    bars_static = ax.bar(x_axis, data_static_percent, color=(0.2, 0.4, 0.6, 0.6), tick_label=x_axis_labels)
    bars_inloop_static = ax.bar(x_axis, data_inloop_static_percent)
    bars_dynamic = ax.bar(x_axis, data_dynamic_percent, bottom=data_static_percent,color="coral")
    bars_inloop_dynamic = ax.bar(x_axis, data_inloop_dynamic_percent, bottom=[(1-v) for v in data_inloop_dynamic_percent], color="red")

    axr = ax.twinx()
    axr.set_ylim(1,0)

    ax.tick_params(axis='x', rotation=90)
    ax.set_ylabel("Ratio over #MemOps")
    ax.set_title("MemOp instructions in " + suite + " benchmarks (per kernel)", y=1.05)

    cmap = plt.cm.get_cmap('tab20', len(x_axis_groups))
    colors = []

    for i in range(len(x_axis_groups)):
        for lbl in x_axis_groups[i]:
            colors.append(cmap(i))

    for i in range(len(colors)):
        ax.get_xticklabels()[i].set_color(colors[i])


    idx=0
    for version in suite_data['manual']['kern']:
        for kernel in suite_data['manual']['kern'][version]:
            kernel_data = suite_data['manual']['kern'][version][kernel]
            if kernel_data[0] > 0:
                bar = bars_static[idx]
                yval = bar.get_height()
                ax.text( bar.get_x() + bar.get_width()/2.0, yval - 0.05, str(kernel_data[1]),fontsize=6, ha='center', alpha=0.5)
            if kernel_data[2] > 0:
                bar = bars_inloop_static[idx]
                yval = bar.get_height()
                ax.text( bar.get_x() + bar.get_width()/2.0, yval - 0.05, str(kernel_data[2]),fontsize=6, ha='center', alpha=0.5)
            if kernel_data[0] - kernel_data[1] > 0: #dynamic
                ax.text(bar.get_x() + bar.get_width()/2.0, 1.005 ,str(kernel_data[0] - kernel_data[1])+'/'+str(kernel_data[3]),\
                        va='bottom', ha='center', alpha=0.5, fontsize=6)
            idx = idx + 1

    ax.legend(['static', 'static_inloop', 'dyn', 'dyn_inloop'], loc='upper left', prop={'size':6.5}, ncol=1,bbox_to_anchor=(-0.15, -0.1))

    suite_kernel_plot.tight_layout()
    suite_kernel_plot.savefig(os.path.join(plot_dir, suite + '_manual_per_kernel.png'))


###
### Cumulative plot
###
msg("+ Building Cumulative plot")
for suite in SUITES:
    suite_data = data[suite]
    x_axis=suite_data['manual']['all']['x-axis']

    suite_manual_plot = plt.figure()
    ax = suite_manual_plot.add_subplot(111)

    data_static_percent = [ (sta_v / all_v) for (sta_v,all_v) in zip(suite_data['manual']['all']['values']['static'],\
                                                                     suite_data['manual']['all']['values']['all'])]
    data_inloop_static_percent = [ (sta_v / all_v) for (sta_v,all_v) in zip(suite_data['manual']['all']['values']['inloop']['static'],\
                                                                            suite_data['manual']['all']['values']['all'])]
    data_dynamic_percent = [ (dyn_v / all_v) for (dyn_v,all_v) in zip(suite_data['manual']['all']['values']['dynamic'],\
                                                                      suite_data['manual']['all']['values']['all'])]
    data_inloop_dynamic_percent = [ (dyn_v / all_v) for (dyn_v,all_v) in zip(suite_data['manual']['all']['values']['inloop']['dynamic'],\
                                                                             suite_data['manual']['all']['values']['all'])]

    bars_static = ax.bar(x_axis, data_static_percent,color=(0.2, 0.4, 0.6, 0.6))
    bars_inloop_static = ax.bar(x_axis, data_inloop_static_percent)
    bars_dynamic = ax.bar(x_axis, data_dynamic_percent, bottom=[(1 - v) for v in data_dynamic_percent],color="coral")
    bars_inloop_dynamic = ax.bar(x_axis, data_inloop_dynamic_percent, bottom=[(1-v) for v in data_inloop_dynamic_percent], color="red")

    for i,bar in enumerate(bars_static):
        if suite_data['manual']['all']['values']['static'][i] > 0:
            yval = bar.get_height()
            ax.text( bar.get_x() + bar.get_width()/2.0, yval - 0.05, str(suite_data['manual']['all']['values']['static'][i]),\
                        fontsize=7, ha='center', alpha=0.5)

    for i,bar in enumerate(bars_inloop_static):
        if suite_data['manual']['all']['values']['inloop']['static'][i] > 0:
            yval = bar.get_height()
            ax.text( bar.get_x() + bar.get_width()/2.0, yval - 0.05, str(suite_data['manual']['all']['values']['inloop']['static'][i]),\
                        fontsize=7, ha='center', alpha=0.5)

    for i,bar in enumerate(bars_dynamic):
        if suite_data['manual']['all']['values']['dynamic'][i] > 0:
            ax.text(bar.get_x() + bar.get_width()/2.0, 1.01,
                str(suite_data['manual']['all']['values']['dynamic'][i])+'/'+\
                str(suite_data['manual']['all']['values']['inloop']['dynamic'][i]),\
                    va='bottom', ha='center', alpha=0.5, fontsize=7)

    ax.legend(['static', 'static_inloop', 'dyn', 'dyn_inloop'], loc='upper left', prop={'size':6.5}, ncol=2,bbox_to_anchor=(-0.15, -0.4))
    ax.set_ylim(0,1);

    plt.xticks(rotation=90,figure=suite_manual_plot)

    ax.set_ylabel("Ratio over #MemOps")
    plt.title("MemOp instructions in " + suite + " benchmarks (cumulative)", y=1.05,figure=suite_manual_plot)

    axr = ax.twinx()
    axr.set_ylim(1,0)

    msg("+ Writing cumulative plot to file")

    suite_manual_plot.tight_layout()
    suite_manual_plot.savefig(os.path.join(plot_dir, suite + '_manual_cumulative.png'), bbox_inches = "tight")

# restore cwd
os.chdir(pre_wd)