import os
import glob
import pickle
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from scipy.stats import ttest_ind_from_stats, ks_2samp

from generative_experiments import METHODS


def string_table(df):
    for col in df.columns:
        df[col] = df[col].apply(lambda x: '{:.2f}'.format(x) if abs(x) > 0.1 else '{:.1e}'.format(x))
    return df


def keep_fashion(results):
    return results[results.Dataset == 'fashion mnist']


def keep_mnist(results):
    return results[results.Dataset == 'mnist']


def keep_svhn(results):
    return results[results.Dataset == 'svhn cropped']


def keep_celeb(results):
    return results[results.Dataset == 'celeb a']


def generative_tables(results, bold_statistical_ties, stat_test, fn=None):
    assert stat_test in {'Whelch', 'K-S'}

    # apply any processing function
    if fn is not None:
        results = fn(results)

    # drop non-reported columns
    results = results.drop(['Best Epoch'], axis=1)

    # loop over the datasets
    table = pd.DataFrame()
    for dataset in results['Dataset'].unique():

        # get number of trials
        n = max(results[results.Dataset == dataset].index) + 1

        # get means and standard deviations
        mean = pd.DataFrame(results[results.Dataset == dataset].groupby(['Dataset', 'Method'], sort=False).mean())
        std = pd.DataFrame(results[results.Dataset == dataset].groupby(['Dataset', 'Method'], sort=False).std(ddof=1))

        # build string table
        df = string_table(mean.copy(deep=True))
        if n >= 2:
            df += '$\\pm$' + string_table(std.copy(deep=True))

        # reset indices for dataset indexing
        mean = mean.reset_index()
        std = std.reset_index()
        df = df.reset_index()

        # loop over the metrics
        for metric in set(df.columns) - {'Dataset', 'Method'}:

            # get index of top performer
            if metric in {'ELBO', 'LL'}:
                i_best = mean[metric].idxmax()
            else:
                i_best = mean[metric].abs().idxmin()

            # bold winner
            df.loc[mean[metric].index[i_best], metric] = '\\textbf{' + df.loc[mean[metric].index[i_best], metric] + '}'

            # bold statistical ties if sufficient trials and requested
            if n >= 2 and bold_statistical_ties:

                # get null hypothesis
                best_method = mean.loc[i_best, 'Method']
                null_samples = results[(results.Dataset == dataset) & (results.Method == best_method)][metric]
                null_mean = mean.loc[i_best, metric]
                null_std = std.loc[i_best, metric]

                # loop over the methods
                for i_method in mean.index:

                    # compute p-value
                    method = mean.loc[i_method, 'Method']
                    method_samples = results[(results.Dataset == dataset) & (results.Method == method)][metric]
                    method_mean = mean.loc[i_method, metric]
                    method_std = std.loc[i_method, metric]
                    if stat_test == 'Whelch':
                        p = ttest_ind_from_stats(null_mean, null_std, n, method_mean, method_std, n, False)[1]
                    else:
                        p = ks_2samp(null_samples.to_numpy(), method_samples.to_numpy())[-1]

                    # bold statistical ties for best
                    if p >= 0.05 and i_best != i_method:
                        new_string = '\\textbf{' + df.loc[i_method, metric] + '}'
                        df.loc[i_method, metric] = new_string

        # append table
        table = table.append(df)

    # make the table pretty
    table.Method = pd.Categorical(table.Method, categories=[method['name'] for method in METHODS])
    table = table.sort_values('Method')
    if len(table['Dataset'].unique()) == 1:
        del table['Dataset']
        table = table.set_index(keys=['Method']).sort_index()
    else:
        table = table.set_index(keys=['Dataset', 'Method']).sort_index()
    table = table[['ELBO', 'LL', 'Mean RMSE', 'Var Bias', 'Sample RMSE']]
    return table.to_latex(escape=False)


def image_reshape(x, column_wise):
    return tf.concat(tf.unstack(x, axis=0), axis=int(not column_wise))


def generative_plots(experiment_dir, results, abridge=True, column_wise=False):

    # select only a few methods for report
    if abridge:
        step = 10
        results = results[results.Method.isin(['Fixed-Var. VAE (1.0)',
                                               'Fixed-Var. VAE (0.001)',
                                               'VAE',
                                               'MAP-VAE',
                                               'Student-VAE',
                                               'V3AE-Gamma',
                                               'V3AE-VBEM*'])]
    else:
        step = 2

    # loop over the experiments and methods
    for dataset in results['Dataset'].unique():

        # get methods
        methods = results[results.Dataset == dataset]['Method'].unique()

        # configure plots to have borders
        plt.rcParams["axes.edgecolor"] = "0.15"
        plt.rcParams["axes.linewidth"] = 0.25

        # initialize figure
        sample_scale = 0.5
        if not column_wise:
            n_rows, n_cols = (len(methods), 1)
            figsize = (sample_scale * 100 / step, 4 * sample_scale * len(methods))
        else:
            n_rows, n_cols = (1, len(methods))
            figsize = (4 * sample_scale * len(methods), sample_scale * 100 / step)

        fig, ax = plt.subplots(n_rows, n_cols, figsize=figsize)
        if not column_wise:
            plt.subplots_adjust(left=0.01 * step, bottom=0.01, right=0.99, top=0.99, wspace=0.0, hspace=0.0)
        else:
            plt.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.9, wspace=0.0, hspace=0.0)

        # loop over the methods in the specified order
        i = -1
        all_methods = [method['name'] for method in METHODS]
        for method in all_methods:
            if method not in methods:
                continue
            i += 1

            # load plot data
            plot_file = method.replace('*', 't') + '_plots.pkl'
            with open(os.path.join(experiment_dir, dataset.replace(' ', '_'), plot_file), 'rb') as f:
                plots = pickle.load(f)

            # grab original data
            x = np.squeeze(image_reshape(plots['x'][0::step], column_wise))

            # grab the mean, std, and samples
            i_plot = results[(results.Dataset == dataset) & (results.Method == method)]['Sample RMSE'].idxmin()
            mean = np.squeeze(image_reshape(plots['reconstruction'][i_plot]['mean'][0::step], column_wise))
            std = np.squeeze(image_reshape(plots['reconstruction'][i_plot]['std'][0::step], column_wise))
            if len(std.shape) == 3:
                std = 1 - std
            sample = np.squeeze(image_reshape(plots['reconstruction'][i_plot]['sample'][0::step], column_wise))
            ax[i].imshow(np.concatenate((x, mean, std, sample), axis=int(column_wise)), vmin=0, vmax=1, cmap='Greys')
            ax[i].set_xticks([])
            ax[i].set_yticks([])
            if method == 'Fixed-Var. VAE (1.0)':
                method = 'VAE ($\\sigma^2 = 1.0$)'
            if method == 'Fixed-Var. VAE (0.001)':
                method = 'VAE ($\\sigma^2 = 0.001$)'
            if column_wise:
                ax[i].set_xlabel(method, fontsize=16)
                ax[i].xaxis.set_label_position('top')
            else:
                ax[i].set_ylabel(method, fontsize=16)

        # save figure
        postfix = '_abridged' if abridge else ''
        fig.savefig(os.path.join('assets', 'fig_vae_samples_' + dataset + postfix + '.pdf').replace(' ', '_'))
        if not abridge:
            fig.savefig(os.path.join('assets', 'fig_vae_samples_' + dataset + postfix + '.png').replace(' ', '_'))


def generative_analysis():

    # experiment directory
    experiment_dir = os.path.join('results', 'vae', 'dense')

    # load results for reach data set
    results = pd.DataFrame()
    for dataset in os.listdir(experiment_dir):
        logger = pd.DataFrame()
        for p in glob.glob(os.path.join(experiment_dir, dataset, '*_metrics.pkl')):
            logger = logger.append(pd.read_pickle(p))
        logger['Dataset'] = dataset.replace('_', ' ')
        results = results.append(logger)

    # drop svhn results
    results = results[results.Dataset != 'svhn cropped']

    # build tables
    with open(os.path.join('assets', 'generative_table.tex'), 'w') as f:
        print(generative_tables(results, bold_statistical_ties=True, stat_test='K-S'), file=f)
    with open(os.path.join('assets', 'generative_table_fashion_mnist.tex'), 'w') as f:
        print(generative_tables(results, bold_statistical_ties=True, stat_test='K-S', fn=keep_fashion), file=f)
    with open(os.path.join('assets', 'generative_table_mnist.tex'), 'w') as f:
        print(generative_tables(results, bold_statistical_ties=True, stat_test='K-S', fn=keep_mnist), file=f)
    with open(os.path.join('assets', 'generative_table_celeb_a.tex'), 'w') as f:
        print(generative_tables(results, bold_statistical_ties=True, stat_test='K-S', fn=keep_celeb), file=f)

    # generate plots
    generative_plots(experiment_dir, results, abridge=False, column_wise=False)
    generative_plots(experiment_dir, results, abridge=True, column_wise=True)


if __name__ == '__main__':
    # script arguments
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    # make assets folder if it doesn't already exist
    if not os.path.exists('assets'):
        os.mkdir('assets')

    # run analysis accordingly
    generative_analysis()
