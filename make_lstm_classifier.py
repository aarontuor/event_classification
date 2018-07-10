"""
@author Aaron Tuor (aarontuor@pnnl.gov)


"""
# import os
# # uncomment next line to run cpu when you have gpu tensorflow installed
# os.environ['CUDA_VISIBLE_DEVICES'] = ''
import sys
import tensorflow as tf
from graph_training_utils import EarlyStop, ModelRunner
from tf_ops import bidir_rnn
from util import Parser
from keras.layers import Dense
import numpy as np


def return_parser():
    """
    Defines and returns argparse ArgumentParser object.

    :return: ArgumentParser
    """
    parser = Parser("Simple token based rnn for language modeling from glove vectors.")
    parser.add_argument('-inputdir', type=str, default='input_data/numpy/')
    parser.add_argument('-learnrate', type=float, default=0.00272473811408,
                        help='Step size for gradient descent.')
    parser.add_argument("-lm_layers", nargs='+', type=int, default=[128],
                        help="A list of hidden layer sizes.")
    parser.add_argument('-mb', type=int, default=16,
                        help='The mini batch size for stochastic gradient descent.')
    parser.add_argument('-debug', action='store_true',
                        help='Use this flag to print feed dictionary contents and dimensions.')
    parser.add_argument('-maxbadcount', type=str, default=10,
                        help='Threshold for early stopping.')
    parser.add_argument('-random_seed', type=int, default=5,
                        help='Random seed for reproducible experiments.')
    parser.add_argument('-verbose', type=int, default=1, help='Whether to print loss during training.')
    parser.add_argument('-decay', action='store_true',
                        help='whether to use learnrate decay')
    parser.add_argument('-decay_rate', type=float,
                        help='rate to decay step size')
    parser.add_argument('-decay_steps', type=int,
                        help='how many steps to perform learnrate decay')
    parser.add_argument('-random', action='store_true',
                        help='Whether to initialize embedding vectors to random values')
    parser.add_argument('-fixed', action='store_true')
    parser.add_argument('-epochs', type=float, default=3,
                        help='Maximum epochs to train on. Need not be in whole epochs.')
    parser.add_argument('-outfile', type=str, default='test_make_lstm_classifier.txt')
    parser.add_argument('-l2', type=float, default=0.0)
    parser.add_argument('-partition', type=str, default='both',
                        help='Can be "both", "desc", or "title"')
    parser.add_argument('-modelsave', type=str, default='saved_model/',
                        help='Directory to save trained model in.')
    # 16 0.00272473811408 128 0 0 0.245176650126 0.94024 0.905526 0.983861 0.943069

    return parser


class Batcher:
    """

    """
    def __init__(self, data):
        """

        :param data:
        """
        self.data = data
        self.index = 0
        self.num = data.shape[0]
        self.epoch = 0
        self.perm = range(self.num)
        np.random.shuffle(self.perm)

    def next_batch(self, batchsize):
        """

        :param batchsize:
        :return:
        """
        assert batchsize <= self.num, 'Too large batchsize!!!'
        if self.index + batchsize <= self.num:
            batch = self.data[self.perm[self.index:self.index + batchsize], :]
            self.index += batchsize
            if self.index == self.num:
                np.random.shuffle(self.perm)
                self.index = 0
                self.epoch += 1
            return batch
        else:
            np.random.shuffle(self.perm)
            self.index = 0
            batch = self.data[self.perm[self.index: self.index + batchsize], :]
            self.epoch += 1
            self.index = batchsize
            return batch


if __name__ == '__main__':
    args = return_parser().parse_args()
    tf.set_random_seed(args.random_seed)
    np.random.seed(args.random_seed)
    if not args.inputdir.endswith('/'):
        args.inputdir += '/'

    desc_traindata = np.load(args.input_dir + 'train_desc.npy')
    desc_devdata = np.load(args.input_dir + 'dev_desc.npy')
    desc_testdata = np.load(args.input_dir + 'test_desc.npy')

    title_traindata = np.load(args.input_dir + 'train_title.npy')
    title_devdata = np.load(args.input_dir + 'dev_title.npy')
    title_testdata = np.load(args.input_dir + 'test_title.npy')

    x_title = tf.placeholder(tf.int32, [None, 36])
    x_desc = tf.placeholder(tf.int32, [None, 64])
    ph_dict = {'x_title': x_title, 'x_desc': x_desc}
    ph_dict['title_lengths'] = tf.placeholder(tf.int32, [None])
    ph_dict['desc_lengths'] = tf.placeholder(tf.int32, [None])
    classes = tf.placeholder(tf.int64, [None])
    ph_dict['classes'] = classes
    ph_dict['id'] = tf.placeholder(tf.int64, [None])
    biovecs = np.load(args.input_dir + 'desc_event_biovecs.npy').astype(np.float32)
    if args.random:
        token_embed = 0.01*tf.truncated_normal(biovecs.shape)
    else:
        token_embed = tf.constant(biovecs)
    if not args.fixed:
        token_embed = tf.Variable(token_embed, name='embeddings')
    token_embed = tf.concat([token_embed, 0.01*tf.truncated_normal([2, biovecs.shape[1]])], 0)
    print(token_embed.shape)

    with tf.variable_scope('title'):
        prediction_states_title, hidden_states_title, final_hidden_title = bidir_rnn(x_title, token_embed,
                                                args.lm_layers, seq_len=ph_dict['title_lengths'],
                                                cell=tf.nn.rnn_cell.BasicLSTMCell)
    with tf.variable_scope('description'):
        prediction_states_desc, hidden_states_desc, final_hidden_desc = bidir_rnn(x_desc, token_embed,
                                                                   args.lm_layers, seq_len=ph_dict['desc_lengths'],
                                                                   cell=tf.nn.rnn_cell.BasicLSTMCell)

    sequence_lengths_title = tf.reshape(tf.cast(ph_dict['title_lengths'], tf.float32), (-1, 1))
    sequence_lengths_desc = tf.reshape(tf.cast(ph_dict['desc_lengths'], tf.float32), (-1, 1))

    mean_hidden_title = tf.reduce_sum(tf.stack(hidden_states_title, axis=0), axis=0) / sequence_lengths_title
    mean_hidden_desc = tf.reduce_sum(tf.stack(hidden_states_desc, axis=0), axis=0) / sequence_lengths_desc
    title_rep = tf.concat([mean_hidden_title, final_hidden_title], 1)
    desc_rep = tf.concat([mean_hidden_desc, final_hidden_desc], 1)

    if args.partition == 'both':
        text_embedding = tf.concat([mean_hidden_desc, final_hidden_desc, mean_hidden_title, final_hidden_title], 1)
    elif args.partition == 'desc':
        text_embedding = tf.concat([mean_hidden_desc, final_hidden_desc], 1)
    elif args.partition == 'title':
        text_embedding = tf.concat([mean_hidden_title, final_hidden_title], 1)

    h1 = Dense(256, activation='relu', name='h1')(text_embedding)
    logits = Dense(2, name='logits')(h1)
    tf.add_to_collection('logits', logits)
    for ph in ['id', 'x_title', 'title_lengths', 'x_desc', 'desc_lengths', 'classes']:
        tf.add_to_collection('placeholderdict', ph_dict[ph])
    ce = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=classes, logits=logits))
    predictions = tf.argmax(logits, axis=1)
    true_positives = tf.reduce_sum(predictions*classes)
    number_of_positives = tf.reduce_sum(classes)
    number_of_positive_predictions = tf.reduce_sum(predictions)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, classes), tf.float32))
    precision = tf.cast(true_positives, tf.float32)/tf.cast(number_of_positive_predictions, tf.float32)
    recall = tf.cast(true_positives, tf.float32)/tf.cast(number_of_positives, tf.float32)
    fscore = 2*((precision * recall)/(precision + recall))
    weights = tf.trainable_variables()
    weights = [w for w in weights if w.name in ['h1/kernel:0', 'h1/bias:0', 'logits/kernel:0', 'logits/bias:0']]
    l2 = [tf.nn.l2_loss(w) for w in weights]
    l2_loss = 0.0
    for l in l2:
        l2_loss += l
    l2_loss /= float(len(l2))
    model = ModelRunner(args.l2*l2_loss + ce, ph_dict, learnrate=args.learnrate, debug=args.debug,
                        decay=True,
                        decay_rate=0.99, decay_steps=20)
    # # training loop
    data = Batcher(np.concatenate([title_traindata, desc_traindata], axis=1))
    dev_datadict = {'id': title_devdata[:, 0],
                     'x_title': title_devdata[:, 4:],
                     'title_lengths': title_devdata[:, 3],
                    'x_desc': desc_devdata[:, 4:],
                    'desc_lengths': desc_devdata[:, 3],
                     'classes': title_devdata[:, 2]}
    test_datadict = {'id': title_testdata[:, 0],
                    'x_title': title_testdata[:, 4:],
                    'title_lengths': title_testdata[:, 3],
                     'x_desc': desc_testdata[:, 4:],
                     'desc_lengths': desc_testdata[:, 3],
                    'classes': title_testdata[:, 2]}
    batch_num = 0
    raw_batch = data.next_batch(args.mb)
    raw_batch_title, raw_batch_desc = raw_batch[:, :40], raw_batch[:, 40:]
    current_loss = sys.float_info.max
    not_early_stop = EarlyStop(args.maxbadcount)
    continue_training = not_early_stop(raw_batch, current_loss)
    best_acc = 0.0
    best_results = {'dev': [0.0, 0.0, 0.0, 0.0], 'test': [0.0, 0.0, 0.0, 0.0]}
    while data.epoch + float(data.index)/float(data.num) < args.epochs:
        train_datadict = {'id': raw_batch[:, 0],
                          'x_title': raw_batch_title[:, 4:],
                          'title_lengths': raw_batch_title[:, 3],
                          'x_desc': raw_batch_desc[:, 4:],
                          'desc_lengths': raw_batch_desc[:, 3],
                          'classes': raw_batch[:, 2]}
        _, current_loss, = model.train_step(train_datadict, eval_tensors=[accuracy], update=True)
        _, dev_acc, dev_prec, dev_rec, dev_fscore, np_h1, np_text_embedding, np_titlerep, np_descrep = model.train_step(dev_datadict,
                                                                                               eval_tensors=[accuracy, precision, recall, fscore, h1, text_embedding,
                                                                                                             title_rep, desc_rep], update=False)
        _, test_acc, test_prec, test_rec, test_fscore = model.train_step(test_datadict, eval_tensors=[accuracy, precision, recall, fscore], update=False)

        if dev_acc > best_acc:
            best_results = {'dev': [dev_acc, dev_prec, dev_rec, dev_fscore],
                            'test': [test_acc, test_prec, test_rec, test_fscore]}
            best_acc = dev_acc
            model.saver.save(model.sess, args.modelsave + 'lstm_event_classifier')

        batch_num += 1
        raw_batch = data.next_batch(args.mb)
        raw_batch_title, raw_batch_desc = raw_batch[:, :40], raw_batch[:, 40:]
        print('epoch: %s\tbatchnum: %s\tbacc: %s\tdev_acc: %s\tdev_prec: %s\tdev_recall: %s\tdev_fscore: %s' % (data.epoch + float(data.index)/float(data.num),
                                                                                                                batch_num,
                                                                                                                current_loss,
                                                                                                                dev_acc,
                                                                                                                dev_prec,
                                                                                                                dev_rec,
                                                                                                                dev_fscore))
    print(best_results['dev'])
    with open(args.outfile + '_dev.txt', 'a') as dev_outfile:
        dev_outfile.write('%s %s %s %s %s %s %s %s %s %s\n' % (args.lm_layers[0],
                                                               args.learnrate,
                                                               args.mb,
                                                               int(args.fixed),
                                                               int(args.random),
                                                               args.l2,
                                                               best_results['dev'][0],
                                                               best_results['dev'][1],
                                                               best_results['dev'][2],
                                                               best_results['dev'][3]))

    with open(args.outfile + '_test.txt', 'a') as test_outfile:
        test_outfile.write('%s %s %s %s %s %s %s %s %s %s\n' % (args.lm_layers[0],
                                                                args.learnrate,
                                                                args.mb,
                                                                int(args.fixed),
                                                                int(args.random),
                                                                args.l2,
                                                                best_results['test'][0],
                                                                best_results['test'][1],
                                                                best_results['test'][2],
                                                                best_results['test'][3]))


