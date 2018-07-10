from sklearn.cross_validation import StratifiedKFold
import argparse
import numpy as np
import random

parser = argparse.ArgumentParser()
parser.add_argument("-infolder", type=str, default="normalized",
                    help="folder with data files for split")
parser.add_argument("-outfolder", type=str, default="split",
                    help="folder to write split data files")

args = parser.parse_args()
if not args.infolder.endswith('/'):
    args.infolder += '/'
if not args.outfolder.endswith('/'):
    args.outfolder += '/'

labels = np.loadtxt(args.infolder + 'label_processed.txt', delimiter=' ')
with open(args.infolder + 'title_processed.txt', 'r') as tfile:
    titles = tfile.read().strip().split('\n')
with open(args.infolder + 'desc_processed.txt', 'r') as dfile:
    desc = dfile.read().strip().split('\n')
print(len(titles))
print(len(desc))
permutation = range(labels.shape[0])

# shuffle the data
random.shuffle(permutation)
labels = labels[permutation]
titles = np.array(titles)[permutation]
desc = np.array(desc)[permutation]


# perform stratified dev/train/test split
folds = StratifiedKFold(labels, n_folds=10, random_state=5)
folds = [f[1] for f in folds]
print([len(f) for f in folds])
train_idxs = np.concatenate(folds[:8])
dev_idxs = folds[8]
test_idxs = folds[9]

# record split to file
train_desc = open(args.outfolder + 'train_desc.txt', 'w')
train_desc.write('\n'.join(desc[train_idxs].tolist()))
train_desc.close()

dev_desc = open(args.outfolder + 'dev_desc.txt', 'w')
dev_desc.write('\n'.join(desc[dev_idxs].tolist()))
dev_desc.close()

test_desc = open(args.outfolder + 'test_desc.txt', 'w')
test_desc.write('\n'.join(desc[test_idxs].tolist()))
test_desc.close()

train_title = open(args.outfolder + 'train_title.txt', 'w')
train_title.write('\n'.join(titles[train_idxs].tolist()))
train_title.close()

dev_title = open(args.outfolder + 'dev_title.txt', 'w')
dev_title.write('\n'.join(titles[dev_idxs].tolist()))
dev_title.close()

test_title = open(args.outfolder + 'test_title.txt', 'w')
test_title.write('\n'.join(titles[test_idxs].tolist()))
test_title.close()


train_labels = np.savetxt(args.outfolder + 'train_labels.txt', labels[train_idxs])
dev_labels = np.savetxt(args.outfolder + 'dev_labels.txt', labels[dev_idxs])
test_labels = np.savetxt(args.outfolder + 'test_labels.txt', labels[test_idxs])