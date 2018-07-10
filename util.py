"""
Python and numpy functions.
"""
from tf_ops import softmax_dist_loss, diag_mvn_loss, full_mvn_loss
import numpy as np
import scipy.sparse as sps
from batch import DayBatcher
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import itertools
# TODO: Check it apr really needs separate inverse and reverse arguments, if so document, if not remove one.


def confusion(truth, guess):
    """

    :param truth:
    :param guess:
    :return:
    """
    cf = np.zeros((guess.shape[1], guess.shape[1]), dtype=np.float32)
    for idx in range(truth.shape[0]):
        cf[truth[idx], np.argmax(guess[idx, :])] += 1.0
    return cf


def confusion_matrix(cm, classes,
                     title=None,
                     cmap=plt.cm.Blues,
                     figname='confusion.png',
                     diag_outline=True,
                     experiment_values=[],
                     verbose=True):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.

    :param cm: numpy 2d confusion matrix
    :param classes: List of class labels
    :param title: Title of plot
    :param cmap:  A color map for coloring the matrix according to value
    :param folder: Where to save matrix (include '/' at end)
    :param diag_outline: Whether to print red box around diagonal matrix entries.
    :param verbose: Whether to print values inside confusion matrix
    """

    np.set_printoptions(precision=2)
    fig, ax = plt.subplots(1)
    rects = []
    if diag_outline:

        for i in range(cm.shape[0]):
            rects.append(patches.Rectangle((i + -0.5,i + -0.5),1,1,
                                     linewidth=1,edgecolor='m',
                                     facecolor='none'))
        for i in range(cm.shape[0]):
            ax.add_patch(rects[i])
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    if title:
        plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    ax.set_xlim([-0.5,cm.shape[0]-0.5])
    thresh = cm.max() / 2.

    if verbose:
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, int(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(figname)
    plt.show()

def make_feature_spec(dataspec):
    """
    Makes lists of all the continuous and categorical features to be used as input features of a neural network.

    :param dataspec: (dict) From a json specification of the purpose of fields in the csv input file (See docs for formatting)
    :return: (dict) features {'categorical': [categorical_feature_1, ..., categorical_feature_j],
                              'continuous': [continuous_feature_1, ..., continuous_feature_k]}
    """
    spec = {k: v for k, v in dataspec.iteritems() if k != 'num_features'}
    feature_spec = {'categorical': [], 'continuous': []}
    for key, field in spec.iteritems():
        if field['num_classes'] == 0 and field['feature']:
            feature_spec['continuous'].append(key)
        if field['num_classes'] > 0 and field['feature']:
            feature_spec['categorical'].append(key)
    return feature_spec


def make_loss_spec(dataspec, mvn):
    """
    Makes a list of tuples for each target to be used in training a multiple output neural network modeling a
    mixed joint distribution of discrete and continuous variables.
    :param dataspec: (dict) From a json specification of the purpose of fields in the csv input file (See docs for formatting)
    :param mvn: Tensorflow function for calculating type of multivariate loss for continuous target vectors.
                Can be tf_ops.diag_mvn_loss, tf_ops.full_mvn_loss, tf_ops.eyed_mvn_loss
    :return: A list of tuples of the form: (target_name, loss_function, dimension) where dimension
             is the dimension of the target vector (for categorical features this is the number of classes, for continuous
             targets this is the size of the continuous target vector)
    """
    spec = {k:v for k,v in dataspec.iteritems() if k != 'num_features'}
    loss_spec = []
    for key, field in spec.iteritems():
        if field['num_classes'] == 0 and field['target']:
            loss_spec.append((key, mvn, len(field['index'])))
        if field['num_classes'] > 0 and field['target']:
            loss_spec.append((key, softmax_dist_loss, field['num_classes']))
    return loss_spec


def to_one_hot(X, dim=None):
    """
    :param X: Vector of indices
    :param dim: Dimension of indexing
    :return: A sparse csr_matrix of one hots.
    """

    if dim is None:
        dim = np.amax(X) + 1

    return sps.csr_matrix(([1.0] * X.shape[0], (range(X.shape[0]),
                                                X.astype(int))),
                          shape=(X.shape[0], dim))


def get_multivariate_loss_names(loss_spec):
    """
    For use in conjunction with `tf_ops.multivariate_loss`. Gives the names of all contributors (columns) of the loss matrix.

    :param loss_spec: A list of 3-tuples of the form (input_name, loss_function, dimension) where
                        input_name is the same as a target in datadict,
                        loss_function takes two parameters, a target and prediction,
                        and dimension is the dimension of the target.
    :return: loss_names is a list concatenated_feature_size long with names of all loss contributors.
    """

    loss_names, log_det_names = [], []
    for i, (input_name, loss_func, dimension) in enumerate(loss_spec):
        if loss_func == softmax_dist_loss:  # discrete
            loss_names.append("loss_%s" % input_name)
        else:  # continuous
            if loss_func == diag_mvn_loss or loss_func == full_mvn_loss:
                log_det_names.append("loss_%s.logdet" % input_name)
            for k in range(dimension):
                loss_names.append("loss_%s.%d" % (input_name, k))

    loss_names.extend(log_det_names)

    return loss_names


def get_mask(lens, num_tokens):
    """
    For masking output of lm_rnn for jagged sequences for correct gradient update.
    Sequence length of 0 will output nan for that row of mask so don't do this.

    :param lens: Numpy vector of sequence lengths
    :param num_tokens: (int) Number of predicted tokens in sentence.
    :return: A numpy array mask MB X num_tokens
             For each row there are: lens[i] values of 1/lens[i]
                                     followed by num_tokens - lens[i] zeros
    """
    mask_template = np.repeat(np.arange(num_tokens).reshape(1, -1), lens.shape[0], axis=0)
    return (mask_template < lens.reshape([-1, 1])).astype(float) / lens.reshape([-1, 1]).astype(float)


class RunningMean:
    """
    Calculates the batchwise running mean from rows, columns, or values of a matrix.
    """
    def __init__(self, axis=0):
        """

        :param axis: The axis to calculate the running mean over. If axis==None then the running mean for the entire array is taken.
        """
        self.n = 0.0  # total number of samples
        self.avg = 0.0
        self.axis = axis

    def __call__(self, samples):
        """

        :param samples: a matrix of samples to incorporate into running mean
        :return: running average over axis
        """

        if self.axis is not None:
            m = float(samples.shape[self.axis])  # num_new_samples
        else:
            m = np.prod(np.array(samples.shape))
        self.n += m
        self.avg = ((self.n - m) / self.n) * self.avg + np.sum(samples, axis=self.axis) / self.n  # second term = (new_avg*m)/n
        return self.avg


class ExponentialRunningMean:
    """
    Calculates the running mean of row vectors batchwise given a sequence of matrices.

    """

    def __init__(self, alpha=1.0):
        """

        :param alpha: (float)  Higher alpha discounts older observations faster.
                                The smaller the alpha, the further you take into consideration the past.
        """
        self.mean = None
        self.alpha = alpha

    def __call__(self, samples):
        """

        :param samples: a matrix of samples to incorporate into running mean
        :return: running average over axis
        """
        if self.mean is None:
            self.mean = np.mean(samples, axis=0).reshape([1, -1])
        else:
            old_mean = self.mean[-1, :]
            self.mean = np.empty((0, samples.shape[1]))
            for i in range(samples.shape[0]):
                new_mean = (1 - self.alpha)*old_mean + self.alpha*samples[i, :]
                self.mean = np.vstack([self.mean, new_mean])
                old_mean = new_mean
        return self.mean


def apr(fname, dayrange, inverse=False, skiprows=0, reverse=True):
    """
    Calculates the average percentile ranking of a red team event given a loss file.
    :param fname: Name of loss file.
    :param dayrange: Range of days to calcuate APR. (For dev/test splits)
    :param inverse:
    :param skiprows: Whether or not to skip first line of loss file.
    :param reverse: (bool) If True then highest number gets rank 1, if False lowest number gets rank 1
    :return:
    """
    loss_matrix = np.loadtxt(fname, skiprows=skiprows)
    time = loss_matrix[:, 0]
    days = np.unique(time)
    daylist = [loss_matrix[loss_matrix[:, 0] == d] for d in
               days[np.logical_and(days >= dayrange[0], days < dayrange[1])]]
    red_rank_perc_vec = np.empty([0, ], dtype=float)
    for ind, day in enumerate(daylist):
        sorted_day = np.array(sorted(day.tolist(), key=lambda x: x[3], reverse=reverse))
        dayred = sorted_day[:, 2]
        redranks = (np.nonzero(dayred)[0].astype(float) + 1.0)/float(len(dayred))
        if inverse:
            redranks = 1.0 - redranks
        red_rank_perc_vec = np.concatenate([red_rank_perc_vec , redranks])
    return np.mean(red_rank_perc_vec)


class Parser(argparse.ArgumentParser):

    def _get_option_tuples(self, option_string):
        result = []

        # option strings starting with two prefix characters are only
        # split at the '='
        chars = self.prefix_chars
        if option_string[0] in chars and option_string[1] in chars:
            if '=' in option_string:
                option_prefix, explicit_arg = option_string.split('=', 1)
            else:
                option_prefix = option_string
                explicit_arg = None
            for option_string in self._option_string_actions:
                if option_string == option_prefix:
                    action = self._option_string_actions[option_string]
                    tup = action, option_string, explicit_arg
                    result.append(tup)

        # single character options can be concatenated with their arguments
        # but multiple character options always have to have their argument
        # separate
        elif option_string[0] in chars and option_string[1] not in chars:
            option_prefix = option_string
            explicit_arg = None
            short_option_prefix = option_string[:2]
            short_explicit_arg = option_string[2:]

            for option_string in self._option_string_actions:
                if option_string == short_option_prefix:
                    action = self._option_string_actions[option_string]
                    tup = action, option_string, short_explicit_arg
                    result.append(tup)
                elif option_string == option_prefix:
                    action = self._option_string_actions[option_string]
                    tup = action, option_string, explicit_arg
                    result.append(tup)

        # shouldn't ever get here
        else:
            self.error(_('unexpected option string: %s') % option_string)

        # return the collected option tuples
        return result
