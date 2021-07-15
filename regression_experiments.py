import os
import sys
import copy
import torch
import pickle
import argparse
import warnings
import numpy as np
import pandas as pd
import sklearn as skl
import tensorflow as tf

from regression_data import generate_toy_data
from callbacks import RegressionCallback
from regression_models import prior_params, NormalRegression, StudentRegression, VariationalPrecisionNormalRegression
from utils_model import monte_carlo_student_t

# import Detlefsen baseline model
sys.path.append(os.path.join(os.getcwd(), 'john-master'))
from toy_regression import detlefsen_toy_baseline
from experiment_regression import detlefsen_uci_baseline

# set results directory globally since its used all over this file
RESULTS_DIR = 'results'


class MeanVarianceLogger(object):
    def __init__(self, df_data=None, df_eval=None):
        self.cols_data = ['Algorithm', 'Prior', 'x', 'y']
        self.df_data = pd.DataFrame(columns=['Algorithm', 'Prior', 'x', 'y']) if df_data is None else df_data
        self.cols_eval = ['Algorithm', 'Prior', 'x', 'mean(y|x)', 'std(y|x)']
        self.df_eval = pd.DataFrame(columns=self.cols_eval) if df_eval is None else df_eval

    @staticmethod
    def __to_list(val):
        if isinstance(val, tf.Tensor):
            val = val.numpy()
        assert isinstance(val, np.ndarray)
        val = np.squeeze(val)
        return val.tolist()

    def update(self, algorithm, prior, x_train, y_train, x_eval, mean, std, trial):

        # update training points data frame
        algorithm_list = [algorithm] * len(x_train)
        prior_list = [prior] * len(x_train)
        x_train = self.__to_list(x_train)
        y_train = self.__to_list(y_train)
        df_new = pd.DataFrame(dict(zip(self.cols_data, (algorithm_list, prior_list, x_train, y_train))),
                              index=[trial] * len(x_train))
        self.df_data = self.df_data.append(df_new)

        # update evaluation data frame
        algorithm_list = [algorithm] * len(x_eval)
        prior_list = [prior] * len(x_eval)
        x_eval = self.__to_list(x_eval)
        mean = self.__to_list(mean)
        std = self.__to_list(std)
        df_new = pd.DataFrame(dict(zip(self.cols_eval, (algorithm_list, prior_list, x_eval, mean, std))),
                              index=[trial] * len(x_eval))
        self.df_eval = self.df_eval.append(df_new)


def compute_metrics(y_eval, y_mean, y_std, y_new):
    y_eval = tf.cast(y_eval, tf.float64)
    y_mean = tf.cast(y_mean, tf.float64)
    y_std = tf.cast(y_std, tf.float64)
    y_new = tf.cast(y_new, tf.float64)
    mean_residuals = y_mean - y_eval
    var_residuals = y_std ** 2 - mean_residuals ** 2
    sample_residuals = y_new - y_eval
    metrics = {
        'Mean Bias': tf.reduce_mean(mean_residuals).numpy(),
        'Mean RMSE': tf.sqrt(tf.reduce_mean(mean_residuals ** 2)).numpy(),
        'Var Bias': tf.reduce_mean(var_residuals).numpy(),
        'Var RMSE': tf.sqrt(tf.reduce_mean(var_residuals ** 2)).numpy(),
        'Sample Bias': tf.reduce_mean(sample_residuals).numpy(),
        'Sample RMSE': tf.sqrt(tf.reduce_mean(sample_residuals ** 2)).numpy()
    }
    return metrics


def train_and_eval(dataset, algo, prior, epochs, batch_size, x_train, y_train, x_eval, y_eval, parallel, **kwargs):

    # toy data configuration
    if 'toy' in dataset:

        # hyper-parameters
        d_hidden = 50
        f_hidden = 'elu'
        learning_rate = 5e-3
        num_mc_samples = 50
        early_stopping = False

        # prior parameters
        k = 20
        u = np.expand_dims(np.linspace(np.min(x_eval), np.max(x_eval), k), axis=-1)
        a, b = prior_params(kwargs.get('precisions'), prior_fam='Gamma')

    # UCI data configuration
    else:

        # hyper-parameters
        n_hidden = 2 if '2x' in algo else 1
        d_hidden = 100 if dataset in {'protein', 'year'} else 50
        f_hidden = 'elu'
        learning_rate = 1e-3
        num_mc_samples = 20
        early_stopping = True

        # prior parameters
        if kwargs.get('k') is None:
            k = None
            u = None
        else:
            k = kwargs.get('k')
            u = x_train[np.random.choice(x_train.shape[0], min(x_train.shape[0], k), replace=False)]
        a = kwargs.get('a')
        b = kwargs.get('b')

    # create TF data loaders
    ds_train = tf.data.Dataset.from_tensor_slices({'x': x_train, 'y': y_train})
    ds_train = ds_train.shuffle(10000, reshuffle_each_iteration=True).batch(batch_size)
    ds_eval = tf.data.Dataset.from_tensor_slices({'x': x_eval, 'y': y_eval})
    ds_eval = ds_eval.shuffle(10000, reshuffle_each_iteration=True).batch(batch_size)

    # pick appropriate model and gradient clip value
    if algo == 'Normal':
        model = NormalRegression
    elif algo == 'Student':
        model = StudentRegression
    else:
        model = VariationalPrecisionNormalRegression

    # declare model instance
    mdl = model(d_in=x_train.shape[1],
                n_hidden=n_hidden,
                d_hidden=d_hidden,
                f_hidden=f_hidden,
                d_out=y_train.shape[1],
                y_mean=0.0 if 'toy' in dataset else np.mean(y_train, axis=0),
                y_var=1.0 if 'toy' in dataset else np.var(y_train, axis=0),
                prior_type=prior,
                prior_fam='Gamma',
                num_mc_samples=num_mc_samples,
                a=a,
                b=b,
                k=k,
                u=u)

    # relevant metric names
    model_ll = 'val_Model LL'
    mean_mse = 'val_Mean MSE'

    # train the model
    callbacks = [RegressionCallback(epochs, parallel)]
    if early_stopping:
        callbacks += [tf.keras.callbacks.EarlyStopping(monitor=model_ll, min_delta=1e-4, patience=50, mode='max',
                                                       restore_best_weights=True)]
    mdl.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss=[None])
    hist = mdl.fit(ds_train, validation_data=ds_eval, epochs=epochs, verbose=0, callbacks=callbacks)

    # test for NaN's
    nan_detected = bool(np.sum(np.isnan(hist.history['loss'])))

    # get index of best validation log likelihood
    i_best = np.nanargmax(hist.history[model_ll])
    if np.nanargmax(hist.history[model_ll]) >= 0.9 * epochs:
        warnings.warn('Model not converged!')

    # retrieve performance metrics
    ll = hist.history[model_ll][i_best]
    mean_rmse = np.sqrt(hist.history[mean_mse][i_best])

    # evaluate predictive model with increased Monte-Carlo samples (if sampling is used by the particular model)
    mdl.num_mc_samples = 2000
    y_mean, y_std, y_new = mdl.predictive_moments_and_samples(x_eval)
    metrics = {'LL': ll, 'Mean RMSL2': mean_rmse}
    metrics.update(compute_metrics(y_eval, y_mean, y_std, y_new))

    return mdl, metrics, y_mean, y_std, nan_detected


def run_experiments(algo, dataset, mode='resume', parallel=False, **kwargs):
    assert algo in {'Detlefsen', 'Detlefsen (fixed)', 'Normal', 'Student', 'Gamma-Normal', 'Gamma-Normal (2x)'}
    assert not (algo == 'Detlefsen (fixed)' and 'toy' not in dataset)
    assert mode in {'replace', 'resume'}

    # parse algorithm/prior names
    if 'Gamma-Normal' in algo:
        prior_fam = 'Gamma'
        prior_type = kwargs.pop('prior_type')
        base_name = algo + '_' + prior_type
    else:
        prior_fam = ''
        prior_type = 'N/A'
        base_name = algo

    # dataset specific hyper-parameters
    n_trials = 10 if dataset in {'toy', 'toy-sparse', 'protein', 'year'} else 20
    batch_size = 500 if 'toy' in dataset else 256
    if 'toy' in dataset:
        batch_iterations = int(6e3)
    elif dataset in {'carbon', 'naval', 'power plant', 'superconductivity'}:
        batch_iterations = int(1e5)
    else:
        batch_iterations = int(2e4)

    # establish experiment directory
    experiment_dir = 'regression_toy' if 'toy' in dataset else 'regression_uci'
    os.makedirs(os.path.join(RESULTS_DIR, experiment_dir), exist_ok=True)

    # parse prior type hyper-parameters
    if prior_type == 'Standard' and 'toy' not in dataset:

        # if prior parameters not provided, use best discovered parameter set from VBEM
        if kwargs.get('a') is None or kwargs.get('b') is None:
            relevant_prior_file = os.path.join(RESULTS_DIR, experiment_dir, dataset, algo + '_VBEM_prior.pkl')
            assert os.path.exists(relevant_prior_file)
            prior_params = pd.read_pickle(relevant_prior_file)
            a, b = np.squeeze(pd.DataFrame(prior_params.groupby(['a', 'b'])['wins'].sum().idxmax()).to_numpy())
            kwargs.update({'a': a, 'b': b})

        base_name += ('_' + str(kwargs.get('a')) + '_' + str(kwargs.get('b')))
        hyper_params = 'a={:f},b={:f}'.format(kwargs.get('a'), kwargs.get('b'))
    elif 'VAMP' in prior_type or prior_type == 'VBEM*':
        base_name += ('_' + str(kwargs.get('k')))
        hyper_params = 'k={:d}'.format(kwargs.get('k'))
    else:
        hyper_params = 'None'
    base_name = base_name.replace(' ', '').replace('*', 't')

    # make sure results subdirectory exists
    os.makedirs(os.path.join(RESULTS_DIR, experiment_dir, dataset), exist_ok=True)

    # create full file names
    logger_file = os.path.join(RESULTS_DIR, experiment_dir, dataset, base_name + '.pkl')
    nan_file = os.path.join(RESULTS_DIR, experiment_dir, dataset, base_name + '_nan_log.txt')
    data_file = os.path.join(RESULTS_DIR, experiment_dir, dataset, base_name + '_data.pkl')
    mv_file = os.path.join(RESULTS_DIR, experiment_dir, dataset, base_name + '_mv.pkl')
    prior_file = os.path.join(RESULTS_DIR, experiment_dir, dataset, base_name + '_prior.pkl')

    # load results if we are resuming
    if mode == 'resume' and os.path.exists(logger_file):
        logger = pd.read_pickle(logger_file)
        if 'toy' in dataset:
            mv_logger = MeanVarianceLogger(df_data=pd.read_pickle(data_file), df_eval=pd.read_pickle(mv_file))
        if prior_type == 'VBEM':
            vbem_logger = pd.read_pickle(prior_file)
        t_start = max(logger.index)
        print('Resuming', dataset, algo, prior_type, 'at trial {:d}'.format(t_start + 2))

    # otherwise, initialize the loggers
    else:
        logger = pd.DataFrame(columns=['Algorithm', 'Prior', 'Hyper-Parameters', 'LL',
                                       'Mean RMSL2', 'Mean Bias', 'Mean RMSE',
                                       'Var Bias', 'Var RMSE',
                                       'Sample Bias', 'Sample RMSE'])
        if os.path.exists(nan_file):
            os.remove(nan_file)
        if 'toy' in dataset:
            mv_logger = MeanVarianceLogger()
        if prior_type == 'VBEM':
            vbem_logger = pd.DataFrame(columns=['a', 'b', 'wins'])
        t_start = -1

    # loop over the trials
    for t in range(t_start + 1, n_trials):
        if not parallel:
            print('\n*****', dataset, 'trial {:d}/{:d}:'.format(t + 1, n_trials), algo, prior_type, '*****')

        # set random number seeds
        seed = args.seed_init * (t + 1)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # toy data
        if 'toy' in dataset:

            # generate data
            x_train, y_train, x_eval, y_eval, true_std = generate_toy_data(num_samples=batch_size,
                                                                           sparse=('sparse' in dataset))

            # compute true precisions
            kwargs.update({'precisions': 1 / true_std[(np.min(x_train) <= x_eval) * (x_eval <= np.max(x_train))] ** 2})

        # uci data
        else:

            # load and split data
            with open(os.path.join('data', dataset, dataset + '.pkl'), 'rb') as f:
                data_dict = pickle.load(f)
            x, y = data_dict['data'], data_dict['target']
            x_train, x_eval, y_train, y_eval = skl.model_selection.train_test_split(x, y, test_size=0.1)

            # scale features
            x_scale = skl.preprocessing.StandardScaler().fit(x_train)
            x_train = x_scale.transform(x_train)
            x_eval = x_scale.transform(x_eval)

        # compute epochs to correspond to the number of batch iterations (as used by Detlefsen)
        epochs = round(batch_iterations / int(np.ceil(x_train.shape[0] / batch_size)))

        # run appropriate algorithm
        nan_detected = False
        if algo == 'Detlefsen' and 'toy' in dataset:
            ll, mean_rmse, mean, std = detlefsen_toy_baseline(x_train, y_train, x_eval, y_eval, bug_fix=False)
            metrics = {'LL': ll, 'Mean RMSL2': mean_rmse}

        elif algo == 'Detlefsen (fixed)' and 'toy' in dataset:
            ll, mean_rmse, mean, std = detlefsen_toy_baseline(x_train, y_train, x_eval, y_eval, bug_fix=True)
            metrics = {'LL': ll, 'Mean RMSL2': mean_rmse}

        elif algo == 'Detlefsen' and 'toy' not in dataset:
            ll, rmsl2, mean, var_samples = detlefsen_uci_baseline(x_train, y_train, x_eval, y_eval,
                                                                  batch_iterations, batch_size, copy.deepcopy(parser))
            py_x = monte_carlo_student_t(mean, 1 / var_samples)
            metrics = {'LL': ll, 'Mean RMSL2': rmsl2}
            metrics.update(compute_metrics(y_eval, py_x.mean(), py_x.stddev(), py_x.sample()))

        else:
            mdl, metrics, mean, std, nan_detected = train_and_eval(dataset, algo, prior_type, epochs, batch_size,
                                                                   x_train, y_train, x_eval, y_eval, parallel, **kwargs)

        # save top priors for VBEM
        if prior_type == 'VBEM':
            indices, counts = np.unique(np.argmax(mdl.pi(x_eval), axis=1), return_counts=True)
            for i, c in zip(indices, counts):
                a = tf.nn.softplus(mdl.u[i]).numpy()[0]
                b = tf.nn.softplus(mdl.v[i]).numpy()[0]
                vbem_logger = vbem_logger.append(pd.DataFrame({'a': a, 'b': b, 'wins': c}, index=[t]))
                vbem_logger.to_pickle(prior_file)

        # print update
        print(dataset, algo, prior_type, '{:d}/{:d}:'.format(t + 1, n_trials))
        print(metrics)

        # print and log NaNs
        if nan_detected:
            print('**** NaN Detected ****')
            print(dataset, prior_fam, prior_type, t + 1, file=open(nan_file, 'a'))

        # save results
        metrics.update({'Algorithm': algo, 'Prior': prior_type, 'Hyper-Parameters': hyper_params})
        logger = logger.append(pd.DataFrame(metrics, index=[t]))
        logger.to_pickle(logger_file)
        if 'toy' in dataset:
            mv_logger.update(algo, prior_type, x_train, y_train, x_eval, mean, std, trial=t)
            mv_logger.df_data.to_pickle(data_file)
            mv_logger.df_eval.to_pickle(mv_file)


if __name__ == '__main__':

    # script arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', type=str, default='Gamma-Normal (2x)', help='algorithm')
    parser.add_argument('--dataset', type=str, default='boston', help='data set = {toy, toy-sparse} union UCI sets')
    parser.add_argument('--mode', type=str, default='resume', help='mode in {replace, resume}')
    parser.add_argument('--prior_type', type=str, default='VBEM*', help='prior type')
    parser.add_argument('--a', type=float, help='standard prior parameter')
    parser.add_argument('--b', type=float, help='standard prior parameter')
    parser.add_argument('--k', type=int, default=100, help='number of mixing prior components')
    parser.add_argument('--parallel', type=int, default=0, help='adjust console print out for parallel runs')
    parser.add_argument('--seed_init', default=1234, type=int, help='random seed init, multiplied by trial number')
    args = parser.parse_args()

    # check inputs
    assert args.dataset in {'toy', 'toy-sparse'}.union(set(os.listdir('data')))

    # assemble configuration dictionary
    KWARGS = {}
    if args.prior_type is not None:
        KWARGS.update({'prior_type': args.prior_type})
    if args.a is not None:
        KWARGS.update({'a': args.a})
    if args.b is not None:
        KWARGS.update({'b': args.b})
    if args.k is not None:
        KWARGS.update({'k': args.k})

    # make result directory if it doesn't already exist
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # run experiments
    run_experiments(args.algorithm, args.dataset, args.mode, bool(args.parallel), **KWARGS)
