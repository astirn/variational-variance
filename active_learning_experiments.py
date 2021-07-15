import os
import sys
import copy
import torch
import pickle
import argparse
import numpy as np
import pandas as pd
import sklearn as skl
import tensorflow as tf

# results directory and regression model training/evaluation
from regression_experiments import RESULTS_DIR, train_and_eval, compute_metrics
from utils_model import monte_carlo_student_t

# import Detlefsen baseline model
sys.path.append(os.path.join(os.getcwd(), 'john-master'))
from experiment_active_learning import detlefsen_uci_baseline


def update_training_set(x_train, y_train, x_pool, y_pool, var, num_to_add):
    i_sort = np.argsort(var)
    x_train = np.concatenate((x_train, x_pool[i_sort[-num_to_add:]]), axis=0)
    y_train = np.concatenate((y_train, y_pool[i_sort[-num_to_add:]]), axis=0)
    x_pool = x_pool[i_sort[:-num_to_add]]
    y_pool = y_pool[i_sort[:-num_to_add]]
    return x_train, y_train, x_pool, y_pool


def run_experiments(algo, dataset, mode='resume', parallel=False, **kwargs):
    assert algo in {'Detlefsen', 'Normal', 'Student', 'Gamma-Normal'}
    assert dataset != 'toy'
    assert mode in {'replace', 'resume'}

    # parse algorithm/prior names
    if algo == 'Gamma-Normal':
        prior_fam = 'Gamma'
        prior_type = kwargs.pop('prior_type')
        base_name = algo + '_' + prior_type
    else:
        prior_fam = ''
        prior_type = 'N/A'
        base_name = algo

    # dataset specific hyper-parameters
    n_trials = 5
    n_al_steps = 10
    batch_size = 256
    if dataset in {'carbon', 'naval', 'power plant', 'superconductivity'}:
        batch_iterations = int(1e5)
    else:
        batch_iterations = int(2e4)

    # establish experiment directory
    experiment_dir = 'active_learning_uci'
    os.makedirs(os.path.join(RESULTS_DIR, experiment_dir), exist_ok=True)

    # parse prior type hyper-parameters
    if prior_type == 'Standard' and dataset != 'toy':
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

    # load results if we are resuming
    if mode == 'resume' and os.path.exists(logger_file):
        logger = pd.read_pickle(logger_file)
        t_start = max(logger.index)
        print('Resuming', dataset, algo, prior_type, 'at trial {:d}'.format(t_start + 2))

    # otherwise, initialize the loggers
    else:
        logger = pd.DataFrame(columns=['Algorithm', 'Prior', 'Hyper-Parameters', 'Percent', 'LL',
                                       'Mean RMSL2', 'Mean Bias', 'Mean RMSE',
                                       'Var Bias', 'Var RMSE',
                                       'Sample Bias', 'Sample RMSE'])
        if os.path.exists(nan_file):
            os.remove(nan_file)
        t_start = -1

    # load data
    with open(os.path.join('data', dataset, dataset + '.pkl'), 'rb') as f:
        data_dict = pickle.load(f)
    x, y = data_dict['data'], data_dict['target']

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

        # split the data
        x_train, x_eval, y_train, y_eval = skl.model_selection.train_test_split(x, y, test_size=0.2)
        xt, xp, yt, yp = skl.model_selection.train_test_split(x_train, y_train, test_size=0.75)
        num_to_add = round(xp.shape[0] * 0.05)

        # loop over active learning steps
        for i in range(n_al_steps):
            if not parallel:
                print('----- AL Step {:d}/{:d}:'.format(i + 1, n_al_steps), '-----')

            # scale features
            scale = skl.preprocessing.StandardScaler().fit(xt)

            # compute epochs to correspond to the number of batch iterations (as used by Detlefsen)
            iterations = round(batch_iterations * xt.shape[0] / (xt.shape[0] + xp.shape[0]))
            epochs = round(iterations / int(np.ceil(x_train.shape[0] / batch_size)))

            # run appropriate algorithm
            nan_detected = False
            if algo == 'Detlefsen':
                ll, rmsl2, m_test, v_test, m_pool, v_pool = detlefsen_uci_baseline(x_train=scale.transform(xt),
                                                                                   y_train=yt,
                                                                                   x_pool=scale.transform(xp),
                                                                                   y_pool=yp,
                                                                                   x_test=scale.transform(x_eval),
                                                                                   y_test=y_eval,
                                                                                   iterations=iterations,
                                                                                   batch_size=batch_size,
                                                                                   parser=copy.deepcopy(parser))
                py_x_test = monte_carlo_student_t(m_test, 1 / v_test)
                metrics = {'LL': ll, 'Mean RMSL2': rmsl2}
                metrics.update(compute_metrics(y_eval, py_x_test.mean(), py_x_test.stddev(), py_x_test.sample()))
                py_x_pool = monte_carlo_student_t(m_pool, 1 / v_pool)
                var = np.prod(py_x_pool.variance(), axis=-1)

            else:
                mdl, metrics, mean, std, nan_detected = train_and_eval(dataset, algo, prior_type, epochs, batch_size,
                                                                       scale.transform(xt), yt,
                                                                       scale.transform(x_eval), y_eval,
                                                                       parallel, **kwargs)
                mdl.num_mc_samples = 2000
                var = np.prod(mdl.predictive_moments_and_samples(scale.transform(xp))[1] ** 2, axis=-1)

            # print update
            print(dataset, algo, prior_type,
                  'Trial {:d}/{:d}:'.format(t + 1, n_trials),
                  'Step {:d}/{:d}:'.format(i + 1, n_al_steps))
            print(metrics)

            # print and log NaNs
            if nan_detected:
                print('**** NaN Detected ****')
                print(dataset, prior_fam, prior_type, t + 1, file=open(nan_file, 'a'))

            # update results
            per = xt.shape[0] / (xt.shape[0] + xp.shape[0])
            metrics.update({'Algorithm': algo, 'Prior': prior_type, 'Hyper-Parameters': hyper_params, 'Percent': per})
            logger = logger.append(pd.DataFrame(metrics, index=[t]))

            # add highest variance points to training set
            xt, yt, xp, yp = update_training_set(xt, yt, xp, yp, var, num_to_add)

        # save results
        logger.to_pickle(logger_file)


if __name__ == '__main__':

    # script arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', type=str, default='Normal', help='algorithm')
    parser.add_argument('--dataset', type=str, default='boston', help='one of the UCI sets')
    parser.add_argument('--mode', type=str, default='resume', help='mode in {replace, resume}')
    parser.add_argument('--prior_type', type=str, help='prior type')
    parser.add_argument('--a', type=float, help='standard prior parameter')
    parser.add_argument('--b', type=float, help='standard prior parameter')
    parser.add_argument('--k', type=int, help='number of mixing prior components')
    parser.add_argument('--parallel', type=int, default=0, help='adjust console print out for parallel runs')
    parser.add_argument('--seed_init', default=1234, type=int, help='random seed init, multiplied by trial number')
    args = parser.parse_args()

    # check inputs
    assert args.dataset in set(os.listdir('data'))

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
