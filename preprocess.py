# There was a bug for this code for python 3 but not for python 2
import os
# uncomment next line to run cpu when you have gpu tensorflow installed
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import sys
reload(sys)
sys.setdefaultencoding('utf8')
import re, patterns, string
from unidecode import unidecode
import numpy as np
from lstm_classifier import EventClassifier
import time


def has_number(inputString):
    """

    :param inputString:
    :return:
    """
    return any(char.isdigit() for char in inputString)


def has_letter(inputString):
    """

    :param inputString:
    :return:
    """
    return any(char.isalpha() for char in inputString)


def punc_perc(text):
    """
    Get punctuation frequency for removing html content.
    From analysis of representative documents html is typically above 15% and English below 5%
    :param text: text to get punctuation frequency from
    :return: Percentage of characters that are punctuation
    """
    return float(sum([1 for c in text if c in string.punctuation]))/float(len(text))


def omit_false_patterns(text):
    """
    :param text: Text to be cleaned.
    :return: Cleaned text
    """
    # =====================Anubhav CLEANUP, all uses REGEX defined in patterns.py=======
    # remove url from the text
    text = text.strip()
    text = re.sub(patterns.URL_REGEX, ' ', text,  flags=re.MULTILINE)
    # remove 4XX and 5XX error from the text
    text = re.sub(patterns.HTTP_ERR_REGEX, ' ', text,  flags=re.MULTILINE)
    # remove disclaimers
    text = re.sub(patterns.DISCLAIMER_REGEX, ' ', text,  flags=re.MULTILINE)
    # remove emails
    text = re.sub(patterns.EMAIL_REGEX, ' ', text,  flags=re.MULTILINE)
    # remove Phone_numbers
    text = re.sub(patterns.PH_NUM_REGEX, ' ', text,  flags=re.MULTILINE)
    # remove multiple occurrences of punctuations.
    text = re.sub(patterns.MUL_PUNC_REGEX, ' ', text,  flags=re.MULTILINE)

    # =====================AARON CLEANUP===============================================
    text = unidecode(text)  # change encoding to ascii
    text = text.strip().replace('\n', ' ')  # omit all occurrence of newline characters.
    text = text.replace('\x1e', ' ')  # information separator two (ascii 30) replaced with space (ascii 32)
    text = text.replace('\x1f', ' ')  # information separator one (ascii 31) replaced with space (ascii 32)
    text = text.replace('&#', ' ')  # replace unicode translation artifact of unknown origin
    text = ''.join([' ' + c + ' ' if c in string.punctuation else c for c in text])  # tokenize punctuation
    text = (' '.join(text.split())).lower()  # remove extra spaces between tokens and lowercase all alphabet text
    return text


class PreProcess:
    """

    """
    def __init__(self, dictfile):
        with open(dictfile, 'r') as dfile:
            words = dfile.read().strip().split('\n')
        self.dictionary = dict(zip(words, range(len(words))))
        self.dictionary['<unk>'] = len(words)
        self.maxlen = {'title': 36, 'desc': 64}

    def tokenize(self, text, text_type):
        """

        :param text:
        :param type:
        :return:
        """
        tokens = text.split(' ')
        tokens = ['<NUMERIC>' if has_number(t) and not has_letter(t) else t for t in tokens]  # special number token
        tokens = ['<ALPHANUMERIC>' if has_number(t) and has_letter(t) else t for t in tokens]  # special alpha-numeric token
        tokens = (['<SOD>'] + tokens + ['<EOD>'])  # special tokens to delimit start and end of document
        assert '\n' not in tokens
        assert '\r' not in tokens
        assert len(tokens) >= 2
        tokens = [self.dictionary['<unk>'] if t not in self.dictionary else self.dictionary[t] for t in tokens]
        size = len(tokens)
        if len(tokens) >= self.maxlen[text_type]:
            # truncate
            tokens = [0, 0, 0] + [self.maxlen[text_type]] + tokens[:self.maxlen[text_type]]
        else:
            # pad with zeros
            tokens = [0, 0, 0] + [size] + tokens + [0] * (self.maxlen[text_type] - size)
        return np.array(tokens).reshape((1, -1))

    def __call__(self, text, text_type):
        text = omit_false_patterns(text)
        return self.tokenize(text, text_type)


def make_dict(dictname, textfile, min_occ=2):
    with open(textfile, 'r') as infile:


if __name__ == '__main__':
    cl = EventClassifier()
    preproc = PreProcess()
    start = time.time()
    predictions = []
    truth = np.loadtxt('input_data/labels.txt')
    with open('input_data/desc.txt', 'r') as desc_txt, open('input_data/title.txt', 'r') as title_txt:
        for idx, d in enumerate(desc_txt):
            desc = preproc(d, 'desc')
            title = preproc(title_txt.readline(), 'title')
            prediction = cl.predict(title, desc)
            print(prediction[0])
            predictions.append(prediction[0])
            print(idx)
    pred = np.array(predictions)
    print(pred.shape[0])
    print(np.sum(np.equal(pred, truth).astype(float))/float(pred.shape[0]))
    print(time.time() - start)
