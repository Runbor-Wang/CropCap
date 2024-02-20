"""
Preprocess a raw json dataset into hdf5/json files for use in data_loader.lua

Input: json file that has the form
[{ file_path: 'path/img.jpg', captions: ['a caption', ...] }, ...]
example element in this list would look like
{'captions': [u'A man with a red helmet on a small moped on a dirt road. ',
u'Man riding a motor bike on a dirt road on the countryside.', u'A man riding on the back of a motorcycle.',
u'A dirt path with a young person on a motor bike rests to the foreground of a verdant area with a bridge and
a background of cloud-wreathed mountains. ', u'A man in a red shirt and a red hat is on a motorcycle on a hill side.'],
'file_path': u'val2014/COCO_val2014_000000391895.jpg', 'id': 391895}

This script reads this json, does some basic preprocessing on the captions
(e.g. lowercase, etc.), creates a special UNK token, and encodes everything to arrays

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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys


sys.path.append(os.path.split(sys.path[0])[0])

import json
import argparse
from random import shuffle, seed
import numpy as np
import torch
from PIL import Image

from torchvision import transforms

from misc.swin_utils import mySwin
import misc.swinencoder as swinencoder


def main(params):
    net = swinencoder.setup(params["pretrain_model"])
    my_preencoder = mySwin(net)
    my_preencoder.cuda()
    my_preencoder.eval()

    IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)])

    imgs = json.load(open(params['input_json'], 'r'))
    imgs = imgs['images']
    N = len(imgs)

    seed(42)  # make reproducible

    dir_swinfeats = params['output_dir'] + '/swinfeats'

    for i, img in enumerate(imgs):
        # load the image
        img_coco_root = os.path.join(dir_swinfeats, str(img['cocoid']))
        if not os.path.isdir(img_coco_root):
            os.mkdir(img_coco_root)
        image_path = os.path.join(params['images_root'], img['filepath'], img['filename'])
        image = Image.open(image_path).resize((384, 384), Image.LANCZOS).convert('RGB')
        if transform is not None:
            image = transform(image).cuda()
        with torch.no_grad():
            swin_0, swin_1, swin_2, swin_3 = my_preencoder(image)
        # write to npz
        np.savez_compressed(os.path.join(img_coco_root, '0'), feat=swin_0.data.cpu().float().numpy())  # compressed
        np.savez_compressed(os.path.join(img_coco_root, '1'), feat=swin_1.data.cpu().float().numpy())  # compressed
        np.savez_compressed(os.path.join(img_coco_root, '2'), feat=swin_2.data.cpu().float().numpy())  # compressed
        np.savez_compressed(os.path.join(img_coco_root, '3'), feat=swin_3.data.cpu().float().numpy())  # compressed

        if i % 1000 == 0:
            print('processing %d/%d (%.2f%% done)' % (i, N, i*100.0/N))
    print('wrote ', params['output_dir'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # input json
    parser.add_argument('--input_json', default='/shenyiming/wangbo/Region_features_30_data/data/dataset_coco.json', help='input json file to process into npz')
    parser.add_argument('--output_dir', default='/shenyiming/wangbo/MSCOCO', help='output npz file')

    # options
    parser.add_argument('--images_root', default='/shenyiming/wangbo/Region_features_30_data/data/IMAGE_ROOT',
                        help='root location in which images are stored, to be prepended to file_path in input json')
    parser.add_argument('--pretrain_model', default='swin-L', type=str,
                        help='resnert18, resnet34, resnet101, resnet152, swin-T, swin-S, swin-B, swin-L')

    args = parser.parse_args()
    params = vars(args)  # convert to ordinary dict
    print('parsed input parameters:')
    print(json.dumps(params, indent=2))
    main(params)
