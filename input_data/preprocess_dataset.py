import sys

reload(sys)
sys.setdefaultencoding('utf8')
import re, patterns, string
from unidecode import unidecode


def has_number(inputString):
    return any(char.isdigit() for char in inputString)


def has_letter(inputString):
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

titles = open('raw/title.txt', 'r').read().strip().split('\n')
descs = open('raw/desc.txt', 'r').read().strip().split('\n')
labels = open('raw/labels.txt', 'r').read().strip().split('\n')

print(len(labels))
print(len(descs))
print(len(labels))
assert len(labels) == len(descs) and len(descs) == len(titles)

with open('normalized/title_processed.txt', 'w') as tof,\
        open('normalized/desc_processed.txt', 'w') as dof,\
        open('normalized/label_processed.txt', 'w') as lof:
    for idx in range(len(labels)):
        titleline = omit_false_patterns(titles[idx])
        descline = omit_false_patterns(descs[idx])

        titletokens = titleline.split(' ')
        titletokens = ['<NUMERIC>' if has_number(t) and not has_letter(t) else t for t in titletokens]  # special number token
        titletokens = ['<ALPHANUMERIC>' if has_number(t) and has_letter(t) else t for t in titletokens]  # special alpha-numeric token
        titletokens = (['<SOD>'] + titletokens + ['<EOD>'])  # special tokens to delimit start and end of document
        assert '\n' not in titletokens
        assert '\r' not in titletokens

        desctokens = descline.split(' ')
        desctokens = ['<NUMERIC>' if has_number(t) and not has_letter(t) else t for t in desctokens]  # special number token
        desctokens = ['<ALPHANUMERIC>' if has_number(t) and has_letter(t) else t for t in desctokens]  # special alpha-numeric token
        desctokens = (['<SOD>'] + desctokens + ['<EOD>'])  # special tokens to delimit start and end of document
        assert '\n' not in desctokens
        assert '\r' not in desctokens
        if len(desctokens) >= 4 and len(titletokens) >= 4:
            descline = ' '.join(desctokens)
            titleline = ' '.join(titletokens)
            assert '\n' not in descline
            assert '\n' not in titleline
            tof.write(titleline + '\n')
            dof.write(descline + '\n')
            lof.write(labels[idx] + '\n')
        if idx % 1000 == 0:
            print(descline)
            print(titleline)
