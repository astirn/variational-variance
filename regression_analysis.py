import os
import glob
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from regression_experiments import RESULTS_DIR
from regression_data import generate_toy_data, REGRESSION_DATA
from utils_analysis import make_clean_method_names, build_table, champions_club_table

# enable background tiles on plots
sns.set(color_codes=True)


def regression_subplot(method, ll_logger, data_logger, mv_logger, ax, color, sparse, homoscedastic):
    ll_logger = ll_logger.loc[ll_logger.index <= 5]
    data_logger = data_logger.loc[data_logger.index <= 5]
    mv_logger = mv_logger.loc[mv_logger.index <= 5]

    # get best performance for this dataset/algorithm/prior combination
    if not sparse:
        i_best = ll_logger[ll_logger.Method == method]['LL'].astype(float).idxmax()
    else:
        i_best = ll_logger[ll_logger.Method == method]['Mean RMSL2'].astype(float).idxmin()

    # plot the training data
    data = data_logger[data_logger.Method == method].loc[i_best]
    sns.scatterplot(data['x'], data['y'], ax=ax, color=color)

    # plot the model's mean and standard deviation
    model = mv_logger[mv_logger.Method == method].loc[i_best]
    model['x'] = model['x'].astype(float)
    model['mean(y|x)'] = model['mean(y|x)'].astype(float)
    model['std(y|x)'] = model['std(y|x)'].astype(float)
    ax.plot(model['x'], model['mean(y|x)'], color=color)
    ax.fill_between(model['x'],
                    model['mean(y|x)'] - 2 * model['std(y|x)'],
                    model['mean(y|x)'] + 2 * model['std(y|x)'],
                    color=color, alpha=0.25)

    # plot the true mean and standard deviation
    _, _, x_eval, true_mean, true_std = generate_toy_data(homoscedastic=homoscedastic)
    ax.plot(x_eval, true_mean, '--k', alpha=0.5)
    ax.plot(x_eval, true_mean + 2 * true_std, ':k', alpha=0.5)
    ax.plot(x_eval, true_mean - 2 * true_std, ':k', alpha=0.5)

    # make it pretty
    ax.set_title(model['Method'].unique()[0])


def toy_regression_plot(ll_logger, data_logger, mv_logger, sparse, homoscedastic):
    # make clean method names for report
    ll_logger = make_clean_method_names(ll_logger)
    data_logger = make_clean_method_names(data_logger)
    mv_logger = make_clean_method_names(mv_logger)

    # get methods for which we have data
    methods_with_data = ll_logger['Method'].unique()

    # methods and order in which we want to plot (if they are available)
    method_order = ['Detlefsen', 'Detlefsen (fixed)', 'Normal', 'Student',
                    'Gamma-Normal (VAP)', 'Gamma-Normal (Gamma)',
                    'Gamma-Normal (VAMP)', 'Gamma-Normal (VAMP*)',
                    'Gamma-Normal (xVAMP)', 'Gamma-Normal (xVAMP*)',
                    'Gamma-Normal (VBEM)', 'Gamma-Normal (VBEM*)']

    # size toy data figure
    n_rows, n_cols = 3, len(method_order) // 2
    fig = plt.figure(figsize=(2.9 * n_cols, 2.9 * n_rows), constrained_layout=False)
    gs = fig.add_gridspec(n_rows, n_cols)
    for i in range(n_cols):
        fig.add_subplot(gs[0, i])
        fig.add_subplot(gs[1, i])
        fig.add_subplot(gs[2, i])

    # make it tight
    plt.subplots_adjust(left=0.05, bottom=0.07, right=0.98, top=0.95, wspace=0.15, hspace=0.15)

    # get color cycle
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    # set x limit according to data generation
    x_lim = [-0.5, 8.0] if sparse else [-5, 15.0]

    # plot toy regression subplots
    for i in range(n_cols):

        # first row subplots
        method1 = method_order[2 * i]
        if method1 in methods_with_data:
            ax = fig.axes[n_rows * i]
            regression_subplot(method1, ll_logger, data_logger, mv_logger, ax, colors[0], sparse, homoscedastic)
            ax.set_xlim(x_lim)
            ax.set_ylim([-25, 25])
            ax.set_xlabel('')
            ax.set_xticklabels([])
            if i > 0:
                ax.set_ylabel('')
                ax.set_yticklabels([])

        # second row subplots
        method2 = method_order[2 * i + 1]
        if method2 in methods_with_data:
            ax = fig.axes[n_rows * i + 1]
            regression_subplot(method2, ll_logger, data_logger, mv_logger, ax, colors[1], sparse, homoscedastic)
            ax.set_xlim(x_lim)
            ax.set_ylim([-25, 25])
            ax.set_xlabel('')
            ax.set_xticklabels([])
            if i > 0:
                ax.set_ylabel('')
                ax.set_yticklabels([])

        # third row subplots
        ax = fig.axes[n_rows * i + 2]
        _, _, x_eval, _, true_std = generate_toy_data(homoscedastic=homoscedastic)
        ax.plot(x_eval, true_std, 'k', label='truth')
        for j, method in enumerate([method1, method2]):
            if method in methods_with_data:
                data = mv_logger[mv_logger.Method == method]
                mean_of_std = data.groupby(['Algorithm', 'x'])[['std(y|x)']].mean().reset_index()
                std_of_std = data.groupby(['Algorithm', 'x'])[['std(y|x)']].std().reset_index()
                ax.plot(mean_of_std['x'], mean_of_std['std(y|x)'], color=colors[j])
                ax.fill_between(std_of_std['x'],
                                mean_of_std['std(y|x)'] - 2 * std_of_std['std(y|x)'],
                                mean_of_std['std(y|x)'] + 2 * std_of_std['std(y|x)'],
                                color=colors[j], alpha=0.25)
        ax.legend().remove()
        ax.set_xlim(x_lim)
        ax.set_ylim([0, 5])
        if i > 0:
            ax.set_ylabel('')
            ax.set_yticklabels([])

    return fig


def toy_regression_analysis(dataset):

    # get all the pickle files
    data_pickles = set(glob.glob(os.path.join(RESULTS_DIR, '*', dataset, '*_data.pkl')))
    mv_pickles = set(glob.glob(os.path.join(RESULTS_DIR, '*', dataset, '*_mv.pkl')))
    prior_pickles = set(glob.glob(os.path.join(RESULTS_DIR, '*', dataset, '*_prior.pkl')))
    ll_pickles = set(glob.glob(os.path.join(RESULTS_DIR, '*', dataset, '*.pkl')))
    ll_pickles = ll_pickles - data_pickles.union(mv_pickles, prior_pickles)

    # aggregate results into single data frame
    ll_logger = pd.DataFrame()
    for p in ll_pickles:
        ll_logger = ll_logger.append(pd.read_pickle(p))
    data_logger = pd.DataFrame()
    for p in data_pickles:
        data_logger = data_logger.append(pd.read_pickle(p))
    mv_logger = pd.DataFrame()
    for p in mv_pickles:
        mv_logger = mv_logger.append(pd.read_pickle(p))

    # generate plot
    fig = toy_regression_plot(ll_logger, data_logger, mv_logger, 'sparse' in dataset, 'homoscedastic' in dataset)
    fig.savefig(os.path.join('assets', 'fig_' + dataset.replace('-', '_') + '.png'))
    fig.savefig(os.path.join('assets', 'fig_' + dataset.replace('-', '_') + '.pdf'))


def drop_detlefsen(df, **kwargs):
    return df[df.Algorithm != 'Detlefsen']


def uci_reported(metric):
    others = None
    if metric == 'LL':
        others = pd.DataFrame(
            data={'boston':                 '\\textit{-2.30$\pm$0.04}',
                  'carbon':                 '--',
                  'concrete':               '\\textit{-3.10$\pm$0.02}',
                  'energy':                 '--',  # '\\textit{-0.68$\pm$0.02}',
                  'naval':                  '--',  # '\\textit{7.13$\pm$0.02}',
                  'power plant':            '\\textit{-2.83$\pm$0.01}',
                  'superconductivity':      '--',
                  'wine-red':               '--',
                   'wine-white':            '--',
                   'yacht':                 '\\textit{-1.03$\pm$0.03}'},
            index=pd.Index([('\citet{sun2019functional}', 'N/A')], names=['Algorithm', 'Prior']))
        # others = others.append(pd.DataFrame(
        #     data={'boston':                 '\\textit{' + '{:.2f}'.format(0.48 - 0.5 * np.log(84.419556)) + '}',
        #           'carbon':                 '--',
        #           'concrete':               '\\textit{' + '{:.2f}'.format(0.93 - 0.5 * np.log(278.8088)) + '}',
        #           'energy':                 '--',
        #           'naval':                  '--',
        #           'power plant':            '\\textit{' + '{:.2f}'.format(0.87 - 0.5 * np.log(291.2519)) + '}',
        #           'superconductivity':      '\\textit{' + '{:.2f}'.format(0.96 - 0.5 * np.log(1173.3062)) + '}',
        #           'wine-red':               '\\textit{' + '{:.2f}'.format(-0.49 - 0.5 * np.log(0.65176064)) + '}',
        #           'wine-white':             '--',
        #           'yacht':                  '\\textit{' + '{:.2f}'.format(2.3 - 0.5 * np.log(229.09421)) + '}'},
        #     index=pd.Index([('\citet{sicking2021novel}', 'N/A')], names=['Algorithm', 'Prior'])))
    elif metric == 'Mean RMSE':
        others = pd.DataFrame(
            data={'boston':                 '\\textit{2.38$\pm$0.10}',
                  'carbon':                 '--',
                  'concrete':               '\\textit{4.94$\pm$0.18}',
                  'energy':                 '--',  # '\\textit{0.41$\pm$0.02}',
                  'naval':                  '--',  # '\\textit{1.2e-04$\pm$0.00}',
                  'power plant':            '\\textit{4.10$\pm$0.05}',
                  'superconductivity':      '--',
                  'wine-red':               '--',
                  'wine-white':             '--',
                  'yacht':                  '\\textit{0.61$\pm$0.07}'},
            index=pd.Index([('\citet{sun2019functional}', 'N/A')], names=['Algorithm', 'Prior']))
        others = others.append(pd.DataFrame(
            data={'boston':                 '\\textit{' + '{:.2f}'.format(0.33 * 84.419556 ** 0.5) + '}',
                  'carbon':                 '--',
                  'concrete':               '\\textit{' + '{:.2f}'.format(0.25 * 278.8088 ** 0.5) + '}',
                  'energy':                 '--',
                  'naval':                  '--',
                  'power plant':            '\\textit{' + '{:.2f}'.format(0.22 * 291.2519 ** 0.5) + '}',
                  'superconductivity':      '\\textit{' + '{:.2f}'.format(0.32 * 1173.3062 ** 0.5) + '}',
                  'wine-red':               '\\textit{' + '{:.2f}'.format(0.80 * 0.65176064 ** 0.5) + '}',
                  'wine-white':             '--',
                  'yacht':                  '\\textit{' + '{:.2f}'.format(0.08 * 229.09421 ** 0.5) + '}'},
            index=pd.Index([('\citet{sicking2021novel}', 'N/A')], names=['Algorithm', 'Prior'])))
    return others


def uci_regression_analysis():

    # experiment directory
    experiment_dir = os.path.join(RESULTS_DIR, 'regression_uci')

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
            results.update({dataset: make_clean_method_names(logger)})

    # make latex tables
    max_cols = 5
    champions_club = []
    for metric in ['LL', 'Mean RMSE', 'Var Bias', 'Sample RMSE']:
        order = 'max' if metric == 'LL' else 'min'
        with open(os.path.join('assets', 'regression_uci_' + metric.lower().replace(' ', '_') + '.tex'), 'w') as f:
            table, cc = build_table(results, metric, order, max_cols, bold_statistical_ties=False, others=uci_reported(metric))
            print(table.replace('NaN', '--'), file=f)
        champions_club.append(cc)

    # print champions club
    with open(os.path.join('assets', 'regression_uci_champions_club.tex'), 'w') as f:
        print(champions_club_table(champions_club), file=f)


if __name__ == '__main__':
    # script arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, default='uci', help='experiment in {toy, toy-sparse, uci}')
    args = parser.parse_args()

    # make assets folder if it doesn't already exist
    if not os.path.exists('assets'):
        os.mkdir('assets')

    # run experiments accordingly
    if args.experiment == 'toy':
        for experiment in ['toy', 'toy-homoscedastic', 'toy-sparse', 'toy-homoscedastic-sparse']:
            toy_regression_analysis(experiment)
    else:
        uci_regression_analysis()

    # hold the plots
    plt.show()
