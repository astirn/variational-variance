import argparse
from abc import ABC

import numpy as np
import tensorflow as tf
import scipy.stats as sps
import tensorflow_probability as tfp

import seaborn as sns
from matplotlib import pyplot as plt

from utils_model import expected_log_normal, monte_carlo_student_t, VariationalVariance
from callbacks import RegressionCallback
from regression_data import generate_toy_data


# configure GPUs
for gpu in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, enable=True)
tf.config.experimental.set_visible_devices(tf.config.list_physical_devices('GPU')[0], 'GPU')


def neural_network(d_in, n_hidden, d_hidden, f_hidden, d_out, f_out=None, name=None):
    nn = tf.keras.Sequential(name=name)
    nn.add(tf.keras.layers.InputLayer(d_in))
    for _ in range(n_hidden):
        nn.add(tf.keras.layers.Dense(d_hidden, f_hidden))
    nn.add(tf.keras.layers.Dense(d_out, f_out))
    return nn


class LocationScaleRegression(tf.keras.Model, ABC):

    def __init__(self, y_mean, y_var):
        super(LocationScaleRegression, self).__init__()

        # save configuration
        self.y_mean = tf.constant(y_mean, dtype=tf.float32)
        self.y_var = tf.constant(y_var, dtype=tf.float32)
        self.y_std = tf.sqrt(self.y_var)

    def whiten_targets(self, y):
        return (y - self.y_mean) / self.y_std

    def de_whiten_mean(self, mu):
        return mu * self.y_std + self.y_mean

    def de_whiten_stddev(self, sigma):
        return sigma * self.y_std

    def de_whiten_precision(self, precision):
        return precision / self.y_var

    def de_whiten_log_precision(self, log_precision):
        return log_precision - tf.math.log(self.y_var)

    def squared_errors(self, mean, y):
        return tf.norm(y - self.de_whiten_mean(mean), axis=-1) ** 2

    def call(self, inputs, **kwargs):
        self.objective(x=inputs['x'], y=inputs['y'])
        return tf.constant(0.0, dtype=tf.float32)


class NormalRegression(LocationScaleRegression, ABC):

    def __init__(self, d_in, n_hidden, d_hidden, f_hidden, d_out, y_mean, y_var, **kwargs):
        super(NormalRegression, self).__init__(y_mean, y_var)
        assert isinstance(d_in, int) and d_in > 0
        assert isinstance(d_hidden, int) and d_hidden > 0
        assert isinstance(d_out, int) and d_out > 0

        # build parameter networks
        self.mean = neural_network(d_in, n_hidden, d_hidden, f_hidden, d_out, f_out=None, name='mu')
        self.precision = neural_network(d_in, n_hidden, d_hidden, f_hidden, d_out, f_out='softplus', name='lambda')

    def ll(self, y, mean, precision, whiten_targets):
        if whiten_targets:
            y = self.whiten_targets(y)
        else:
            mean = self.de_whiten_mean(mean)
            precision = self.de_whiten_precision(precision)
        return tfp.distributions.MultivariateNormalDiag(loc=mean, scale_diag=precision ** -0.5).log_prob(y)

    def objective(self, x, y):

        # run parameter networks
        mean = self.mean(x)
        precision = self.precision(x)

        # compute log likelihood on whitened targets
        ll = self.ll(y, mean, precision, whiten_targets=True)

        # use negative log likelihood on whitened targets as minimization objective
        self.add_loss(-tf.reduce_mean(ll))

        # compute de-whitened log likelihood
        ll_de_whitened = self.ll(y, mean, precision, whiten_targets=False)

        # assign model's log likelihood (Bayesian methods will use log posterior predictive likelihood instead)
        ll_model = ll_de_whitened

        # observation metrics
        self.add_metric(ll, name='LL', aggregation='mean')
        self.add_metric(ll_de_whitened, name='LL (de-whitened)', aggregation='mean')
        self.add_metric(ll_model, name='Model LL', aggregation='mean')
        self.add_metric(self.squared_errors(mean, y), name='Mean MSE', aggregation='mean')

    def predictive_moments_and_samples(self, x):
        """Predictive model is the multivariate normal employed during MLE but with rescaled parameters"""
        n_dist = tfp.distributions.MultivariateNormalDiag(loc=self.de_whiten_mean(self.mean(x)),
                                                          scale_diag=self.de_whiten_stddev(self.precision(x) ** -0.5))
        return n_dist.mean().numpy(), n_dist.stddev().numpy(), n_dist.sample().numpy()


class PredictiveStudent(LocationScaleRegression, ABC):
    def __init__(self, y_mean, y_var):
        super(PredictiveStudent, self).__init__(y_mean, y_var)

    def predictive_moments_and_samples(self, x):
        """Posterior predictive is Student's T but with rescaled parameters"""

        # establish distribution
        alpha = self.alpha(x)
        loc = self.de_whiten_mean(self.mu(x))
        scale = self.de_whiten_stddev(tf.sqrt(self.beta(x) / alpha))
        t_dist = tfp.distributions.StudentT(df=2 * alpha, loc=loc, scale=scale)
        t_dist = tfp.distributions.Independent(t_dist, reinterpreted_batch_ndims=1)

        # compute approximate moments
        samples_f64 = tf.cast(t_dist.sample(self.num_mc_samples), tf.float64)
        mean_approx = tf.cast(tf.reduce_mean(samples_f64, axis=0), tf.float32)
        stddev_approx = tf.cast(tf.math.reduce_std(samples_f64, axis=0), tf.float32)

        # take analytic moments where possible
        mean = tf.where(tf.greater(alpha, 0.5), t_dist.mean(), mean_approx)
        stddev = tf.where(tf.greater(alpha, 1.0), t_dist.stddev(), stddev_approx)

        return mean.numpy(), stddev.numpy(), t_dist.sample().numpy()


class StudentRegression(PredictiveStudent, ABC):

    def __init__(self, d_in, n_hidden, d_hidden, f_hidden, d_out, y_mean, y_var, num_mc_samples, **kwargs):
        super(StudentRegression, self).__init__(y_mean, y_var)
        assert isinstance(d_in, int) and d_in > 0
        assert isinstance(d_hidden, int) and d_hidden > 0
        assert isinstance(d_out, int) and d_out > 0
        assert isinstance(num_mc_samples, int) and num_mc_samples > 0

        # save configuration
        self.num_mc_samples = num_mc_samples

        # build parameter networks
        self.mu = neural_network(d_in, n_hidden, d_hidden, f_hidden, d_out, f_out=None, name='mu')
        self.alpha_network = neural_network(d_in, n_hidden, d_hidden, f_hidden, d_out, f_out='softplus', name='alpha')
        self.alpha = lambda x: self.alpha_network(x) + 1
        self.beta = neural_network(d_in, n_hidden, d_hidden, f_hidden, d_out, f_out='softplus', name='beta')

    def ll(self, y, mu, alpha, beta, whiten_targets):
        if whiten_targets:
            y = self.whiten_targets(y)
            loc = mu
            scale = tf.sqrt(beta / alpha)
        else:
            loc = self.de_whiten_mean(mu)
            scale = self.de_whiten_stddev(tf.sqrt(beta / alpha))
        py_x = tfp.distributions.StudentT(df=2 * alpha, loc=loc, scale=scale)
        return tfp.distributions.Independent(py_x, reinterpreted_batch_ndims=1).log_prob(y)

    def objective(self, x, y):

        # run parameter networks
        mu = self.mu(x)
        alpha = self.alpha(x)
        beta = self.beta(x)

        # compute log likelihood on whitened targets
        ll = self.ll(y, mu, alpha, beta, whiten_targets=True)

        # use negative log likelihood on whitened targets as minimization objective
        self.add_loss(-tf.reduce_mean(ll))

        # compute de-whitened log likelihood
        ll_de_whitened = self.ll(y, mu, alpha, beta, whiten_targets=False)

        # assign model's log likelihood (Bayesian methods will use log posterior predictive likelihood instead)
        ll_model = ll_de_whitened

        # observation metrics
        self.add_metric(ll, name='LL', aggregation='mean')
        self.add_metric(ll_de_whitened, name='LL (de-whitened)', aggregation='mean')
        self.add_metric(ll_model, name='Model LL', aggregation='mean')
        self.add_metric(self.squared_errors(mu, y), name='Mean MSE', aggregation='mean')


class VariationalPrecisionNormalRegression(PredictiveStudent, VariationalVariance, ABC):

    def __init__(self, d_in, n_hidden, d_hidden, f_hidden, d_out, y_mean, y_var, prior_type, prior_fam, num_mc_samples, **kwargs):
        PredictiveStudent.__init__(self, y_mean, y_var)
        VariationalVariance.__init__(self, d_out, prior_type, prior_fam, **kwargs)
        assert isinstance(d_in, int) and d_in > 0
        assert isinstance(d_hidden, int) and d_hidden > 0
        assert isinstance(d_out, int) and d_out > 0
        assert isinstance(num_mc_samples, int) and num_mc_samples > 0

        # save configuration
        self.num_mc_samples = num_mc_samples

        # build parameter networks
        self.mu = neural_network(d_in, n_hidden, d_hidden, f_hidden, d_out, f_out=None, name='mu')
        alpha_f_out = 'softplus' if self.prior_fam == 'Gamma' else None
        self.alpha_network = neural_network(d_in, n_hidden, d_hidden, f_hidden, d_out, f_out=alpha_f_out, name='alpha')
        self.alpha = lambda x: self.alpha_network(x) + 1
        self.beta = neural_network(d_in, n_hidden, d_hidden, f_hidden, d_out, f_out='softplus', name='beta')
        if self.prior_type in {'xVAMP', 'xVAMP*', 'VBEM', 'VBEM*'}:
            self.pi = neural_network(d_in, n_hidden, d_hidden, f_hidden, self.u.shape[0], f_out='softmax', name='pi')

    def expected_ll(self, y, mu, alpha, beta, whiten_targets):

        # compute expected precision and log precision under the variational posterior
        expected_precision = self.expected_precision(alpha, beta)
        expected_log_precision = self.expected_log_precision(alpha, beta)

        # whiten things accordingly
        if whiten_targets:
            y = self.whiten_targets(y)
        else:
            mu = self.de_whiten_mean(mu)
            expected_precision = self.de_whiten_precision(expected_precision)
            expected_log_precision = self.de_whiten_log_precision(expected_log_precision)

        return expected_log_normal(y, mu, expected_precision, expected_log_precision)

    def objective(self, x, y):

        # run parameter networks
        mu = self.mu(x)
        alpha = self.alpha(x)
        beta = self.beta(x)

        # variational family
        qp, p_samples = self.variational_precision(alpha, beta, leading_mc_dimension=False)

        # expected log likelihood on whitened targets
        ell = self.expected_ll(y, mu, alpha, beta, whiten_targets=True)

        # compute KL divergence w.r.t. p(lambda)
        vamp_samples = tf.expand_dims(self.u, axis=0) if 'VAMP' in self.prior_type else None
        dkl = self.dkl_precision(qp, p_samples, pi_parent_samples=tf.expand_dims(x, axis=0), vamp_samples=vamp_samples)

        # use negative evidence lower bound as minimization objective
        elbo = ell - dkl
        self.add_loss(-tf.reduce_mean(elbo))

        # compute de-whitened expected log likelihood
        ell_de_whitened = self.expected_ll(y, mu, alpha, beta, whiten_targets=False)

        # assign model's log likelihood as the log posterior predictive likelihood
        ll_model = self.log_posterior_predictive_likelihood(y, mu, alpha, beta, p_samples)

        # observation metrics
        self.add_metric(elbo, name='ELBO', aggregation='mean')
        self.add_metric(ell, name='ELL', aggregation='mean')
        self.add_metric(dkl, name='KL', aggregation='mean')
        self.add_metric(ell_de_whitened, name='ELL (de-whitened)', aggregation='mean')
        self.add_metric(ll_model, name='Model LL', aggregation='mean')
        self.add_metric(self.squared_errors(mu, y), name='Mean MSE', aggregation='mean')

    def log_posterior_predictive_likelihood(self, y, mu, alpha, beta, p_samples):
        loc = self.de_whiten_mean(mu)
        if self.prior_fam == 'Gamma':
            py_x = tfp.distributions.StudentT(df=2 * alpha, loc=loc, scale=self.de_whiten_stddev(tf.sqrt(beta / alpha)))
            return tfp.distributions.Independent(py_x, reinterpreted_batch_ndims=1).log_prob(y)
        elif self.prior_fam == 'LogNormal':
            return monte_carlo_student_t(loc, self.de_whiten_precision(p_samples)).log_prob(y)


def prior_params(precisions, prior_fam):
    if prior_fam == 'Gamma':
        a, _, b_inv = sps.gamma.fit(precisions, floc=0)
        b = 1 / b_inv
    else:
        a, b = np.mean(np.log(precisions)), np.std(np.log(precisions))
    print(prior_fam, 'Prior:', a, b)
    return a, b


def fancy_plot(x_train, y_train, x_eval, true_mean, true_std, mdl_mean, mdl_std, title):
    # squeeze everything
    x_train = np.squeeze(x_train)
    y_train = np.squeeze(y_train)
    x_eval = np.squeeze(x_eval)
    true_mean = np.squeeze(true_mean)
    true_std = np.squeeze(true_std)
    mdl_mean = np.squeeze(mdl_mean)
    mdl_std = np.squeeze(mdl_std)

    # get a new figure
    fig, ax = plt.subplots(2, 1)
    fig.suptitle(title)

    # plot the data
    sns.scatterplot(x_train, y_train, ax=ax[0])

    # plot the true mean and standard deviation
    ax[0].plot(x_eval, true_mean, '--k')
    ax[0].plot(x_eval, true_mean + true_std, ':k')
    ax[0].plot(x_eval, true_mean - true_std, ':k')

    # plot the model's mean and standard deviation
    l = ax[0].plot(x_eval, mdl_mean)[0]
    ax[0].fill_between(x_eval[:, ], mdl_mean - mdl_std, mdl_mean + mdl_std, color=l.get_color(), alpha=0.5)
    ax[0].plot(x_eval, true_mean, '--k')

    # clean it up
    ax[0].set_ylim([-20, 20])
    ax[0].set_ylabel('y')

    # plot the std
    ax[1].plot(x_eval, mdl_std, label='predicted')
    ax[1].plot(x_eval, true_std, '--k', label='truth')
    ax[1].set_ylim([0, 5])
    ax[1].set_xlabel('x')
    ax[1].set_ylabel('std(y|x)')
    plt.legend()

    return fig


if __name__ == '__main__':

    # enable background tiles on plots
    sns.set(color_codes=True)

    # unit tests for LocationScaleRegression class
    lcr = LocationScaleRegression(y_mean=np.random.normal(0, 1), y_var=np.random.gamma(1, 1))
    a = tf.random.gamma(shape=[100], alpha=1, beta=1, dtype=tf.float32)
    b = tf.random.gamma(shape=[100], alpha=1, beta=1, dtype=tf.float32)
    scale1 = lcr.de_whiten_stddev(tf.sqrt(b / a))
    scale2 = lcr.de_whiten_precision(a / b) ** -0.5
    assert tf.reduce_max(tf.math.squared_difference(scale1, scale2)) < 1e-10

    # script arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', type=str, default='Gamma-Normal', help='algorithm')
    parser.add_argument('--prior_type', default='xVAMP*', type=str, help='prior type')
    parser.add_argument('--sparse', default=1, type=int, help='sparse toy data option')
    parser.add_argument('--seed', default=1234, type=int, help='prior type')
    args = parser.parse_args()

    # random number seeds
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # set configuration
    D_HIDDEN = 50
    PRIOR_TYPE = args.prior_type
    PRIOR_FAM = 'Gamma' if 'Gamma' in args.algorithm else 'LogNormal'
    N_MC_SAMPLES = 50
    LEARNING_RATE = 1e-2
    EPOCHS = int(6e3)

    # load data
    x_train, y_train, x_eval, true_mean, true_std = generate_toy_data(sparse=bool(args.sparse))
    ds_train = tf.data.Dataset.from_tensor_slices({'x': x_train, 'y': y_train}).batch(x_train.shape[0])

    # compute standard prior according to prior family
    A, B = prior_params(1 / true_std[(np.min(x_train) <= x_eval) * (x_eval <= np.max(x_train))] ** 2, PRIOR_FAM)

    # VAMP prior pseudo-input initializers
    U = np.expand_dims(np.linspace(np.min(x_eval), np.max(x_eval), 20), axis=-1)

    # pick the appropriate model
    if args.algorithm == 'Normal':
        MODEL = NormalRegression
    elif args.algorithm == 'Student':
        MODEL = StudentRegression
    else:
        MODEL = VariationalPrecisionNormalRegression

    # declare model instance
    mdl = MODEL(d_in=x_train.shape[1],
                d_hidden=D_HIDDEN,
                f_hidden='elu',
                d_out=y_train.shape[1],
                y_mean=0.0,
                y_var=1.0,
                prior_type=PRIOR_TYPE,
                prior_fam=PRIOR_FAM,
                num_mc_samples=N_MC_SAMPLES,
                a=A,
                b=B,
                k=20,
                u=U)

    # build the model. loss=[None] avoids warning "Output output_1 missing from loss dictionary".
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    mdl.compile(optimizer=optimizer, run_eagerly=False)

    # train model
    hist = mdl.fit(ds_train, epochs=EPOCHS, verbose=0, callbacks=[RegressionCallback(EPOCHS)])

    # evaluate predictive model with increased Monte-Carlo samples (if sampling is used by the particular model)
    mdl.num_mc_samples = 2000
    mdl_mean, mdl_std, mdl_samples = mdl.predictive_moments_and_samples(x_eval)

    # plot results for toy data
    fig = plt.figure()
    fig.suptitle(args.algorithm)
    plt.plot(hist.history['Model LL'])
    fancy_plot(x_train, y_train, x_eval, true_mean, true_std, mdl_mean, mdl_std, args.algorithm)
    plt.show()
