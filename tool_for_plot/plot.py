import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import os.path as osp
import numpy as np
from arguments import parse_args
import re

DIV_LINE_WIDTH = 50

# Global vars for tracking and labeling data at load time.
exp_idx = 0
units = dict()

def plot_data(data, xaxis='Epoch', value="AverageEpRet", condition="Condition1", smooth=1, **kwargs):
    if smooth > 1:
        """
        smooth data with moving window average.
        that is,
            smoothed_y[t] = average(y[t-k], y[t-k+1], ..., y[t+k-1], y[t+k])
        where the "smooth" param is width of that window (2k+1)
        """
        y = np.ones(smooth)
        i=0
        for datum in data:
            # import ipdb
            # ipdb.set_trace()
            # y = np.ones(smooth)
            # if datum[condition][0] == 'SAC-FSP' or datum[condition][0] == 'SAC-raw':
            #     smooth_new = 195
            # else:
            #     smooth_new = smooth
            # y = np.ones(smooth_new)
            x = np.asarray(datum[value])
            z = np.ones(len(x))
            smoothed_x = np.convolve(x,y,'same') / np.convolve(z,y,'same')
            print('i={}, smoothed_x.size={},datum[value].size={}'.format(i,smoothed_x.size,datum[value].size))
            datum[value] = smoothed_x
            i=i+1

    if isinstance(data, list):
        data = pd.concat(data, ignore_index=True)
    # sns.set_context("poster", rc={"lines.linewidth": 3})
    sns.set(style="whitegrid", font_scale=1)
    sns.despine(top=True, right=True, left=False, bottom=False)
    # flatui = ["#1F77B3", "#FE7E0e", "#9267BB","#D52728", "#2C9F2C" , "#E277C1"] # blue, orange, purple, red, green, pink
    # flatui = ["#FF0000", "#185C8A", "#9267BB", "#FF8000", "#2C9F2C" , "#E277C1"] # red, blue, purple, orange, green, pink
    flatui = ["#9267BB", "#185C8A", "#FF0000", "#E277C1", "#2C9F2C" , "#FF8000"] # purple, blue, red, pink, green, orange FF8000 F9AB00  FB8D2B
    # sns.tsplot(data=data, time=xaxis, value=value, unit="Unit", condition=condition, ci='sd', **kwargs)
    with sns.color_palette(flatui):
        sns.lineplot(data=data, x=xaxis, y=value, hue=condition, errorbar='sd', legend=False, lw=2.5, **kwargs)
    """
    If you upgrade to any version of Seaborn greater than 0.8.1, switch from 
    tsplot to lineplot replacing L29 with:

        sns.lineplot(data=data, x=xaxis, y=value, hue=condition, ci='sd', **kwargs)

    Changes the colorscheme and the default legend style, though.
    """
    # plt.legend(loc='best').set_draggable(True)
    #plt.legend(loc='upper center', ncol=3, handlelength=1,
    #           borderaxespad=0., prop={'size': 13})

    """
    For the version of the legend used in the Spinning Up benchmarking page, 
    swap L38 with:

    plt.legend(loc='upper center', ncol=6, handlelength=1,
               mode="expand", borderaxespad=0., prop={'size': 13})
    """

    xscale = np.max(np.asarray(data[xaxis])) > 5e3
    if xscale:
        # Just some formatting niceness: x-axis scale in scientific notation if max x is large
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

    plt.tight_layout(pad=0.5)

def get_datasets(logdir, condition=None, cond1='Condition1'):
    """
    Recursively look through logdir for output files produced by
    spinup.logx.Logger. 

    Assumes that any file "progress.txt" is a valid hit. 
    """
    global exp_idx
    global units
    datasets = []
    for root, _, files in os.walk(logdir):
        if 'progress.csv' in files:
            exp_name = None
            try:
                config_path = open(os.path.join(root,'config.json'))
                config = json.load(config_path)
                if 'exp_name' in config:
                    exp_name = config['exp_name']
            except:
                print('No file named config.json')
            condition1 = condition or exp_name or 'exp'
            condition2 = condition1 + '-' + str(exp_idx)
            exp_idx += 1
            if condition1 not in units:
                units[condition1] = 0
            unit = units[condition1]
            units[condition1] += 1

            try:
                # exp_data = pd.read_table(os.path.join(root,'progress.csv'))
                exp_data = pd.read_csv(os.path.join(root, 'progress.csv'))
            except:
                print('Could not read from %s'%os.path.join(root,'progress.csv'))
                continue
            performance = 'AverageTestEpRet' if 'AverageTestEpRet' in exp_data else 'AverageEpRet'
            exp_data.insert(len(exp_data.columns),'Unit',unit)
            exp_data.insert(len(exp_data.columns), cond1, condition1)
            exp_data.insert(len(exp_data.columns),'Condition2',condition2)
            exp_data.insert(len(exp_data.columns),'Performance',exp_data['Value'])
            datasets.append(exp_data)
    return datasets


def get_all_datasets(all_logdirs, legend=None, select=None, exclude=None, cond1='Condition1'):
    """
    For every entry in all_logdirs,
        1) check if the entry is a real directory and if it is, 
           pull data from it; 

        2) if not, check to see if the entry is a prefix for a 
           real directory, and pull data from that.
    """
    logdirs = []
    
    for logdir in all_logdirs:
        if osp.isdir(logdir) and logdir[-1]==os.sep:
            logdirs += [logdir]
        else:
            basedir = osp.dirname(logdir)
            fulldir = lambda x : osp.join(basedir, x)
            prefix = logdir.split(os.sep)[-1]
            listdir= os.listdir(basedir)
            logdirs += sorted([fulldir(x) for x in listdir if prefix in x])

    """
    Enforce selection rules, which check logdirs for certain substrings.
    Makes it easier to look at graphs from particular ablations, if you
    launch many jobs at once with similar names.
    """
    if select is not None:
        logdirs = [log for log in logdirs if all(x in log for x in select)]
    if exclude is not None:
        logdirs = [log for log in logdirs if all(not(x in log) for x in exclude)]

    # Verify logdirs
    print('Plotting from...\n' + '='*DIV_LINE_WIDTH + '\n')
    for logdir in logdirs:
        print(logdir)
    print('\n' + '='*DIV_LINE_WIDTH)

    # Make sure the legend is compatible with the logdirs
    assert not(legend) or (len(legend) == len(logdirs)), \
        "Must give a legend title for each set of experiments."

    # Load data from logdirs
    data = []
    if legend:
        for log, leg in zip(logdirs, legend):
            data += get_datasets(log, leg, cond1)
    else:
        for log in logdirs:
            data += get_datasets(log, cond1)
    return data


def make_plots(all_logdirs, legend=None, xaxis=None, values=None, cond1='Condition1', count=False,  
               font_scale=1.5, smooth=1, select=None, exclude=None, estimator='mean', img_save_dir=None, env='HalfCheetah'):
    data = get_all_datasets(all_logdirs, legend, select, exclude, cond1)
    values = values if isinstance(values, list) else [values]
    condition = 'Condition2' if count else cond1
    estimator = getattr(np, estimator)      # choose what to show on main curve: mean? max? min?
    for value in values:
        plt.figure()
        plot_data(data, xaxis=xaxis, value=value, condition=condition, smooth=smooth, estimator=estimator)
        if img_save_dir is not None:
            dir = img_save_dir + '-' + value + '.pdf'
            plt.title(env.split("-")[0], fontsize=25) 
            plt.xlabel('Steps', fontsize=20)
            plt.ylabel('Evaluation returns', fontsize=20)
            # plt.grid()
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)
            ax = plt.gca()
            ax.ticklabel_format(style='sci', scilimits=(-1,2), axis='y')
            ax.spines['top'].set_color('none') 
            ax.spines['right'].set_color('none') 
            ax.spines['bottom'].set_color('black') 
            ax.spines['left'].set_color('black')
            ax.spines['bottom'].set_linewidth(1.5)
            ax.spines['left'].set_linewidth(1.5)
            ax.yaxis.get_offset_text().set(size=15)
            ax.xaxis.get_offset_text().set(size=15) 
            plt.savefig(dir, dpi=300, bbox_inches='tight') 
    plt.show()


def main(logdir, img_save_dir, legend, env):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('logdir', nargs='*')
    parser.add_argument('--legend', '-l', default=['raw', 'OFE', 'FSP'], nargs='*')
    parser.add_argument('--xaxis', '-x', default='step')
    # ['reward_mean_10_epochs', 'value_loss', 'action_loss', 'dist_entropy']
    parser.add_argument('--values', '-y', default=['Value'], nargs='*')
    parser.add_argument('--cond1', default='Auxiliary Task')
    parser.add_argument('--count', action='store_true')
    parser.add_argument('--smooth', '-s', type=int, default=35)
    parser.add_argument('--select', nargs='*')
    parser.add_argument('--exclude', nargs='*')
    parser.add_argument('--est', default='mean')
    args = parser.parse_args()
    """

    Args: 
        logdir (strings): As many log directories (or prefixes to log 
            directories, which the plotter will autocomplete internally) as 
            you'd like to plot from.

        legend (strings): Optional way to specify legend for the plot. The 
            plotter legend will automatically use the ``exp_name`` from the
            config.json file, unless you tell it otherwise through this flag.
            This only works if you provide a name for each directory that
            will get plotted. (Note: this may not be the same as the number
            of logdir args you provide! Recall that the plotter looks for
            autocompletes of the logdir args: there may be more than one 
            match for a given logdir prefix, and you will need to provide a 
            legend string for each one of those matches---unless you have 
            removed some of them as candidates via selection or exclusion 
            rules (below).)

        xaxis (string): Pick what column from data is used for the x-axis.
             Defaults to ``TotalEnvInteracts``.

        value (strings): Pick what columns from data to graph on the y-axis. 
            Submitting multiple values will produce multiple graphs. Defaults
            to ``Performance``, which is not an actual output of any algorithm.
            Instead, ``Performance`` refers to either ``AverageEpRet``, the 
            correct performance measure for the on-policy algorithms, or
            ``AverageTestEpRet``, the correct performance measure for the 
            off-policy algorithms. The plotter will automatically figure out 
            which of ``AverageEpRet`` or ``AverageTestEpRet`` to report for 
            each separate logdir.

        count: Optional flag. By default, the plotter shows y-values which
            are averaged across all results that share an ``exp_name``, 
            which is typically a set of identical experiments that only vary
            in random seed. But if you'd like to see all of those curves 
            separately, use the ``--count`` flag.

        smooth (int): Smooth data by averaging it over a fixed window. This 
            parameter says how wide the averaging window will be.

        select (strings): Optional selection rule: the plotter will only show
            curves from logdirs that contain all of these substrings.

        exclude (strings): Optional exclusion rule: plotter will only show 
            curves from logdirs that do not contain these substrings.

    """
    args.logdir = logdir
    args.legend = legend
    make_plots(args.logdir, args.legend, args.xaxis, args.values, args.cond1, args.count,
               smooth=args.smooth, select=args.select, exclude=args.exclude,
               estimator=args.est, img_save_dir=img_save_dir, env=env)

if __name__ == "__main__":
    args = parse_args()
    # env_list = ['HalfCheetah-v2', 'Walker2d-v2', 'Ant-v2', 'Swimmer-v2']
    env_list = ['HalfCheetah-v2', 'Walker2d-v2', 'Hopper-v2', 'Ant-v2', 'Swimmer-v2', 'Humanoid-v2']
    # env_list = ['Ant-v2']
    policy_list = ['SAC', 'PPO']
    aux_list = ['raw', 'OFE', 'FSP']
    # args.policy = args.policy + '-' + aux 
    seed_list = [0,1,2,5,6,7,8,10,11,12]
    data_save_dir0 = 'results/data/exp_main'
    if not os.path.exists('results/img'):
        os.mkdir('results/img')

    for env in env_list:

        logdirs = []
        legend = []

        for policy in policy_list:
            '''make img-save-dir'''
            img_save_dir0 = 'results/img/exp_main'
            # img_save_subdir = os.path.join(img_save_dir, policy)
            if not os.path.exists(img_save_dir0):
                os.mkdir(img_save_dir0)
            # if not os.path.exists(img_save_dir):
            #     os.mkdir(img_save_dir)
            #     os.mkdir(img_save_subdir)   
            # elif not os.path.exists(img_save_subdir):
            #     os.mkdir(img_save_subdir)

            img_save_dir = os.path.join(img_save_dir0, env)
            
            for aux in aux_list:
                logdir0 = os.path.join(data_save_dir0, policy + '-' + aux +  '-' + env)
                # import ipdb
                # ipdb.set_trace()
                seed_list = [str(re.findall(r"\d+", x)[-1]) for x in os.listdir(logdir0)]
                seed_list.sort()
                for i in seed_list:
                    logdir = os.path.join(data_save_dir0, policy + '-' + aux +  '-' + env,\
                                                policy+ '-' + aux + '-' + env + '-s' + str(i))
                    logdir += os.sep
                    logdirs.append(logdir)
                    legend.append('{}-{}'.format(policy, aux))

        main(logdirs, img_save_dir, legend, env)