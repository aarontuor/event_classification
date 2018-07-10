import numpy as np

with open('event_desc_dict.txt', 'r') as dfile:
    words = dfile.read().strip().split('\n')
    desc_dictionary = dict(zip(words, range(len(words))))
    desc_dictionary['<unk>'] = len(words)

for f in ['dev', 'train', 'test']:
    desc_doclist = []
    title_doclist = []
    desc_docs = open('split/' + f + '_desc.txt', 'r').readlines()
    title_docs = open('split/' + f + '_title.txt', 'r').readlines()
    labelfile = open('split/' + f + '_labels.txt', 'r')
    labels = labelfile.read().split('\n')

    for idx in range(len(desc_docs)):
        desc_tokens = desc_docs[idx].strip().split()
        title_tokens = title_docs[idx].strip().split()

        if len(title_tokens) >= 4 and len(desc_tokens) >= 4:
            desc_tokens = [desc_dictionary['<unk>'] if t not in desc_dictionary else desc_dictionary[t] for t in desc_tokens]
            title_tokens = [desc_dictionary['<unk>'] if t not in desc_dictionary else desc_dictionary[t] for t in title_tokens]
            desc_size = len(desc_tokens)
            if len(desc_tokens) >= 64:
                desc_tokens = [idx] + [0] + [int(float(labels[idx]))] + [64] + desc_tokens[:64]
            else:
                desc_tokens = [idx] + [0] + [int(float(labels[idx]))] + [desc_size] + desc_tokens + [0] * (64 - desc_size)
            desc_doclist.append(desc_tokens)

            title_size = len(title_tokens)
            if len(title_tokens) >= 36:
                title_tokens = [idx] + [0] + [int(float(labels[idx]))] + [36] + title_tokens[:36]
            else:
                title_tokens = [idx] + [0] + [int(float(labels[idx]))] + [title_size] + title_tokens + [0] * (36 - title_size)
            title_doclist.append(title_tokens)

    print(np.array(desc_doclist).shape)
    print(np.array(desc_doclist).dtype)
    print(np.array(title_doclist).shape)
    print(np.array(title_doclist).dtype)

    np.save('numpy/' + f + '_desc.npy', np.array(desc_doclist))
    np.save('numpy/' + f + '_title.npy', np.array(title_doclist))