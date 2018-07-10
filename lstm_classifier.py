"""
author: Aaron Tuor
lstm_classifer.py: Class definition for pre-trained lstm event text classifier
"""
import os
# uncomment next line to run cpu when you have gpu tensorflow installed
# os.environ['CUDA_VISIBLE_DEVICES'] = ''
import tensorflow as tf
import numpy as np
from graph_training_utils import get_feed_dict


class EventClassifier:

    def __init__(self):
        """
        Restores pre-trained LSTM text classifier for prediction.
        """
        saver = tf.train.import_meta_graph('saved_model/lstm_event_classifier.meta')
        self.sess = tf.Session()
        saver.restore(self.sess, tf.train.latest_checkpoint('saved_model/'))

    def predict(self, title, desc):
        """

        :param np_input: (np.array, docs X max_title_len) Matrix of integer rep of multiple text inputs. Rows are for individual texts.
                         Columns are structured so that the first 40 columns have the zero padded title text and
                         meta data. The final columns have the zero padded description text and meta data.
                         The first 4 columns of title and desc segments of the vectors have:
                         text_id, blank_spot, length_of_text, class_of_text. These positions can be filled with
                         zeros for prediction
        :return: A vector of 1's and 0's of length np_input.shape[0] (1 for is an event zero for is not an event)
        """
        datadict = {'x_title': title[:, 4:],
                    'title_lengths': title[:, 3],
                    'x_desc': desc[:, 4:],
                    'desc_lengths': desc[:, 3]}
        placeholders = tf.get_collection('placeholderdict')
        ph_dict = {'x_title': placeholders[1],
                   'title_lengths': placeholders[2],
                   'x_desc': placeholders[3],
                   'desc_lengths': placeholders[4]}
        fd = get_feed_dict(datadict, ph_dict, train=0)
        np_logits = self.sess.run(tf.get_collection('logits')[0], feed_dict=fd)
        return np.argmax(np_logits, axis=1)

if __name__ == '__main__':
    # Test run
    cl = EventClassifier()
    desc = np.load('input_data/all_numpy/desc.npy')
    title = np.load('input_data/all_numpy/title.npy')
    predictions = cl.predict(title, desc)
    num_correct = title[:, 2] == predictions
    accuracy = np.sum(title[:, 2] == predictions).astype(float)/float(predictions.shape[0])
    print(accuracy)
