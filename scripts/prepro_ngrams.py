"""
Preprocess a raw json dataset into hdf5/json files for use in data_loader.py
This will preprocess the dataset and get the cache for calculating cider score.

This script reads this json, does some basic preprocessing on the captions
(e.g. lowercase, etc.), creates a special UNK token, and encodes everything to arrays

Input: json file that has the form
[{ file_path: 'path/img.jpg', captions: ['a caption', ...] }, ...]
example element in this list would look like
{'captions': [u'A man with a red helmet on a small moped on a dirt road. ', u'Man riding a motor bike on a dirt road on
the countryside.', u'A man riding on the back of a motorcycle.', u'A dirt path with a young person on a motor bike rests
to the foreground of a verdant area with a bridge and a background of cloud-wreathed mountains. ', u'A man in a red
shirt and a red hat is on a motorcycle on a hill side.'],
'file_path': u'val2014/COCO_val2014_000000391895.jpg', 'id': 391895}

Output: a json file and an hdf5 file
The hdf5 file contains several fields:
/images is (N,3,256,256) uint8 array of raw image data in RGB format
/labels is (M,max_length) uint32 array of encoded labels, zero padded
/label_start_ix and /label_end_ix are (N,) uint32 arrays of pointers to the 
  first and last indices (in range 1..M) of labels for each image
/label_length stores the length of the sequence for each of the M sequences

The json file has a dict that contains:
- an 'ix_to_word' field storing the vocab in form {ix:'word'}, where ix is 1-indexed
- an 'images' field that is a list holding auxiliary information for each image, 
  such as in particular the 'split' it was assigned to.
"""

import os
import json
import argparse
from six.moves import cPickle
from collections import defaultdict
import tensorflow as tf  # there is no this tensorflow import in original version.
import math  # there is no this math import in original version.


def precook(s, n=4, out=False):
    """
    Takes a string as input and returns an object that can be given to
    either cook_refs or cook_test. This is optional: cook_refs and cook_test
    can take string arguments as well.
    :param s: string : sentence to be converted into ngrams
    :param n: int    : number of ngrams for which representation is calculated
    :return: term frequency vector for occuring ngrams
    """
    words = s.split()
    counts = defaultdict(int)
    for k in range(1, n+1):
        for i in range(len(words)-k+1):
            ngram = tuple(words[i:i+k])
            counts[ngram] += 1
    return counts


def cook_refs(refs, n=4):  # # lhuang: oracle will call with "average"
    '''Takes a list of reference sentences for a single segment
    and returns an object that encapsulates everything that BLEU
    needs to know about them.
    :param refs: list of string : reference sentences for some image
    :param n: int : number of ngrams for which (ngram) representation is calculated
    :return: result (list of dict)
    '''
    return [precook(ref, n) for ref in refs]


def create_crefs(refs):
    crefs = []
    for ref in refs:
        # ref is a list of 5 captions
        crefs.append(cook_refs(ref))
    return crefs


def compute_doc_freq(crefs):
    '''
    Compute term frequency for reference data.
    This will be used to compute idf (inverse document frequency later)
    The term frequency is stored in the object
    :return: None
    '''
    document_frequency = defaultdict(float)
    for refs in crefs:
        # refs, k ref captions of one image
        # for ngram in set([ngram for ref in refs for (ngram,count) in ref.iteritems()]):
        for ngram in set([ngram for ref in refs for (ngram, count) in ref.items()]):
            document_frequency[ngram] += 1
    return document_frequency


def build_dict(imgs, w2i, params):
    w2i['<eos>'] = 0

    count_imgs = 0

    refs_words = []
    refs_idxs = []
    for img in imgs:
        if (params['split'] == img['split']) or (params['split'] == 'train' and img['split'] == 'restval') or \
                (params['split'] == 'all'):
            ref_words = []
            ref_idxs = []
            for sent in img['sentences']:
                tmp_tokens = sent['tokens'] + ['<eos>']
                tmp_tokens = [_ if _ in w2i else 'UNK' for _ in tmp_tokens]
                ref_words.append(' '.join(tmp_tokens))
                ref_idxs.append(' '.join([str(w2i[_]) for _ in tmp_tokens]))
            refs_words.append(ref_words)
            refs_idxs.append(ref_idxs)
            count_imgs += 1
    print('total imgs:', count_imgs)

    ngram_words = compute_doc_freq(create_crefs(refs_words))
    ngram_idxs = compute_doc_freq(create_crefs(refs_idxs))
    return ngram_words, ngram_idxs, count_imgs


def get_timing_signal_1d(length, channels, min_timescale=1.0, max_timescale=1.0e4, start_index=0):
    """Gets a bunch of sinusoids of different frequencies.
    Each channel of the input Tensor is incremented by a sinusoid of a different
    frequency and phase.
    This allows attention to learn to use absolute and relative positions.
    Timing signals should be added to some precursors of both the query and the
    memory inputs to attention.
    The use of relative position is possible because sin(x+y) and cos(x+y) can be
    expressed in terms of y, sin(x) and cos(x).
    In particular, we use a geometric sequence of timescales starting with
    min_timescale and ending with max_timescale.  The number of different
    timescales is equal to channels / 2. For each timescale, we
    generate the two sinusoidal signals sin(timestep/timescale) and
    cos(timestep/timescale).  All of these sinusoids are concatenated in
    the channels dimension.
    Args:
      length: scalar, length of timing signal sequence.
      channels: scalar, size of timing embeddings to create. The number of
          different timescales is equal to channels / 2.
      min_timescale: a float
      max_timescale: a float
      start_index: index of first position
    Returns:
      a Tensor of timing signals [1, length, channels]
    """
    position = tf.to_float(tf.range(length) + start_index)
    num_timescales = channels // 2
    log_timescale_increment = (
        math.log(float(max_timescale) / float(min_timescale)) /
        (tf.to_float(num_timescales) - 1))
    inv_timescales = min_timescale * tf.exp(
        tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
    scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
    signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
    signal = tf.pad(signal, [[0, 0], [0, tf.mod(channels, 2)]])
    signal = tf.reshape(signal, [1, length, channels])
    return signal


def main(params):
    imgs = json.load(open(params['input_json'], 'r'))
    i2w = json.load(open(params['dict_json'], 'r'))['ix_to_word']
    w2i = {w: i for i, w in i2w.items()}

    imgs = imgs['images']

    ngram_words, ngram_idxs, ref_len = build_dict(imgs, w2i, params)

    cPickle.dump({'document_frequency': ngram_words, 'ref_len': ref_len}, open(params['output_pkl']+'-words.p', 'wb'), protocol=cPickle.HIGHEST_PROTOCOL)
    cPickle.dump({'document_frequency': ngram_idxs, 'ref_len': ref_len}, open(params['output_pkl']+'-idxs.p', 'wb'), protocol=cPickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # input json
    parser.add_argument('--input_json', default='/wangbo/Dataset/mini_coco/preprocess/mini_karpathy_coco.json', help='input json file to process into hdf5')
    parser.add_argument('--dict_json', default='/wangbo/Dataset/mini_coco/preprocess/cocotalk.json', help='output json file')
    parser.add_argument('--output_pkl', default='/wangbo/Dataset/mini_coco/preprocess/coco-train', help='output pickle file')
    parser.add_argument('--split', default='train', help='test, val, train, all')
    args = parser.parse_args()
    params = vars(args)  # convert to ordinary dict

    main(params)
