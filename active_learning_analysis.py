import os
import glob
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from regression_experiments import RESULTS_DIR
from regression_data import REGRESSION_DATA
from utils_analysis import make_clean_method_names, build_table

# enable background tiles on plots
sns.set(color_codes=True)


def fix_early_runs(pickle_files):
    for result in pickle_files:
        log = pd.read_pickle(result)

        # eliminate POOPS priors
        log = log[log.Prior != 'vamp_poops']
        log = log[log.Prior != 'vamp_trainable_poops']
        log = log[log.Prior != 'vbem_poops']

        # fix percentage offset for our models
        for prior in log['Prior'].unique():
            if prior != 'N/A':
                log.loc[log.Prior == prior, 'Percent'] = log[log.Prior == 'N/A']

        # save the result
        log.to_pickle(result)


def generate_plots(results, metric):

    # methods and order in which we want to plot (if they are available)
    method_order = ['Detlefsen', 'Normal', 'Student',
                    'Gamma-Normal (VAP)', 'Gamma-Normal (Gamma)',
                    'Gamma-Normal (VAMP)', 'Gamma-Normal (VAMP*)',
                    'Gamma-Normal (xVAMP)', 'Gamma-Normal (xVAMP*)',
                    'Gamma-Normal (VBEM)', 'Gamma-Normal (VBEM*)']

    # generate subplots
    n_rows = 2
    n_cols = int(np.ceil(len(results) / n_rows))
    fig, ax = plt.subplots(n_rows, n_cols, **{'figsize': (3.25 * n_cols, 3.25 * n_rows + 0.25)})
    ax = np.reshape(ax, -1)

    # make it tight
    plt.subplots_adjust(left=0.05, bottom=0.20, right=0.98, top=0.95, wspace=0.20, hspace=0.25)

    # plot results
    for i, dataset in enumerate(results):

        # make clean method names for plots
        result = make_clean_method_names(results[dataset]).rename(columns={'Percent': 'Ratio'})

        # get methods for which we have data
        methods_with_data = result['Method'].unique()

        # plot results
        ax[i].set_title(dataset)
        sns.lineplot(x='Ratio', y=metric, hue='Method', hue_order=method_order, ci=95,  data=result, ax=ax[i])

        # y label once per row
        if i % n_cols != 0:
            ax[i].set_ylabel('')

        # x label only bottom row
        if i // n_cols < n_rows - 1:
            ax[i].set_xlabel('')

        # shared legend
        if i == (n_rows - 1) * n_cols:
            ax[i].legend(ncol=(len(methods_with_data) + 1) // 2, loc='lower left', bbox_to_anchor=(0.2, -0.55))
        else:
            ax[i].legend().remove()

    return fig


def integrate_active_learning_curves(log, **kwargs):
    return pd.DataFrame(log.groupby(['Algorithm', 'Prior', log.index], sort=False)[kwargs['metric']].sum()).reset_index()


def active_learning_analysis():

    # experiment directory
    experiment_dir = os.path.join(RESULTS_DIR, 'active_learning_uci')

    # load results for each data set
    results = dict()
    for dataset in REGRESSION_DATA.keys():
        result_dir = os.path.join(experiment_dir, dataset)
        if os.path.exists(result_dir):
            logger = pd.DataFrame()
            for p in glob.glob(os.path.join(result_dir, '*.pkl')):
                if '_prior' in p:
                    continue
                logger = logger.append(pd.read_pickle(p))
            results.update({dataset: logger})

    # generate plots
    for metric in ['LL', 'Mean RMSE']:
        generate_plots(results, metric).savefig(os.path.join('assets', 'fig_al_' + metric.lower() + '.pdf'))

    # print result tables
    max_cols = 5
    process_fn = [integrate_active_learning_curves]
    with open(os.path.join('assets', 'active_learning_uci_ll.tex'), 'w') as f:
        print(build_table(results, 'LL', order='max', max_cols=max_cols, bold_statistical_ties=False, process_fn=process_fn)[0], file=f)
    with open(os.path.join('assets', 'active_learning_uci_mean_rmse.tex'), 'w') as f:
        print(build_table(results, 'Mean RMSE', order='min', max_cols=max_cols, bold_statistical_ties=False, process_fn=process_fn)[0], file=f)


if __name__ == '__main__':

    # make assets folder if it doesn't already exist
    if not os.path.exists('assets'):
        os.mkdir('assets')

    # run analysis accordingly
    active_learning_analysis()

    # hold the plots
    plt.show()
