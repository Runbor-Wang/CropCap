from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np
import json
import os
import sys
sys.path.append("..")
import misc.utils as utils
from misc.report import ReportData

REPORT_DATA_PKL_FILE_TEMPLATE = '%s_%s_report_data.pkl'


def language_eval(dataset, preds, model_id, image_root, split):
    annFile = '/shenyiming/wangbo/Dataset/MS_COCO/captions_val2014.json'
    from pycocotools.coco import COCO
    from misc.correct_coco_eval_cap import CorrectCOCOEvalCap

    results_dir = 'eval_results'
    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)
    cache_path = os.path.join(results_dir, model_id + '_' + split + '.json')

    coco = COCO(annFile)
    valids = coco.getImgIds()

    # filter results to only those in MSCOCO validation set (will be about a third)
    preds_filt = [p for p in preds if p['image_id'] in valids]
    print('using %d/%d predictions' % (len(preds_filt), len(preds)))
    json.dump(preds_filt, open(cache_path, 'w'))  # serialize to temporary json file. Sigh, COCO API...

    cocoRes = coco.loadRes(cache_path)
    cocoEval = CorrectCOCOEvalCap(coco, cocoRes)
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    cocoEval.evaluate()

    if image_root:
        # Save cocoEval and any other relevant information into a pickle to be used
        # later for generating a report and visualizing results.
        report_data = ReportData(cocoEval, preds, image_root, model_id, split)
        pickle_file_name = REPORT_DATA_PKL_FILE_TEMPLATE % (model_id, split)
        pickle_path = os.path.join(results_dir, pickle_file_name)
        report_data.save_to_pickle(pickle_path)

    # create output dictionary
    out = {}
    for metric, score in cocoEval.eval.items():
        out[metric] = score

    imgToEval = cocoEval.imgToEval
    for p in preds_filt:
        image_id, caption = p['image_id'], p['caption']
        imgToEval[image_id]['caption'] = caption
    with open(cache_path, 'w') as outfile:
        json.dump({'overall': out, 'imgToEval': imgToEval}, outfile)

    return out


def eval_split(my_swin, model, crit, loader, eval_kwargs={}):
    verbose = eval_kwargs.get('verbose', True)
    verbose_beam = eval_kwargs.get('verbose_beam', 1)
    verbose_loss = eval_kwargs.get('verbose_loss', 1)
    num_images = eval_kwargs.get('num_images', eval_kwargs.get('val_images_use', -1))  # 5000
    split = eval_kwargs.get('split', 'val')
    lang_eval = eval_kwargs.get('language_eval', 1)
    dataset = eval_kwargs.get('dataset', 'coco')
    beam_size = eval_kwargs.get('beam_size', 1)

    # Make sure in the evaluation mode
    model.eval()

    loader.reset_iterator(split)

    n = 0
    loss = 0
    loss_sum = 0
    loss_evals = 1e-8
    predictions = []
    while True:
        data = loader.get_batch(split)
        n = n + loader.batch_size

        if data.get('labels', None) is not None and verbose_loss:
            # forward the model to get loss
            tmp = [data['input_images'], data['labels'], data['labels_masks']]
            tmp = [torch.from_numpy(_).cuda() if _ is not None else _ for _ in tmp]
            input_image, labels, labels_masks = tmp

            x_0, x_1, x_2, x_3 = my_swin(input_image)

            with torch.no_grad():
                loss = crit(model(x_0, x_1, x_2, x_3, labels), labels[:, 1:], labels_masks[:, 1:]).item()
            loss_sum = loss_sum + loss
            loss_evals = loss_evals + 1

        # forward the model to also get generated samples for each image
        # Only leave one feature for each image, in case duplicate sample
        input_image = data['input_images'][np.arange(loader.batch_size) * loader.seq_per_img]
        input_image = torch.from_numpy(input_image).cuda()

        x_0, x_1, x_2, x_3 = my_swin(input_image)

        # forward the model to also get generated samples for each image
        with torch.no_grad():
            seq = model(x_0, x_1, x_2, x_3, opt=eval_kwargs, mode='sample')[0].data

        # Print beam search
        if beam_size > 1 and verbose_beam:
            for i in range(loader.batch_size):
                print('\n'.join([utils.decode_sequence(loader.get_vocab(), _['seq'].unsqueeze(0))[0] for _ in model.done_beams[i]]))
                print('\n'.join([utils.decode_sequence(loader.get_vocab(), _['seq'].unsqueeze(0))[0] for _ in model.done_beams[i]]))
                print('--' * 10)
        sents = utils.decode_sequence(loader.get_vocab(), seq)

        for k, sent in enumerate(sents):
            image_id = data['infos'][k]['id']
            entry = {'image_id': image_id, 'caption': sent, 'file_path': data['infos'][k]['file_path']}
            if eval_kwargs.get('dump_path', 0) == 1:
                entry['file_name'] = data['infos'][k]['file_path']
            predictions.append(entry)
            if eval_kwargs.get('dump_images', 0) == 1:
                # dump the raw image to vis/ folder
                cmd = 'cp "' + os.path.join(eval_kwargs['image_root'], data['infos'][k]['file_path']) + '" vis/imgs/' + str(image_id) + '.jpg'  # still gross
                print(cmd)
                os.system(cmd)

            if verbose:
                print('image %s: %s' % (entry['image_id'], entry['caption']))

        # if we wrapped around the split or used up val imgs budget then bail
        ix0 = data['bounds']['it_pos_now']
        ix1 = data['bounds']['it_max']
        if num_images != -1:
            ix1 = min(ix1, num_images)
        for i in range(n - ix1):
            predictions.pop()

        if verbose:
            print('evaluating validation preformance... %d/%d (%f)' % (ix0 - 1, ix1, loss))

        if data['bounds']['wrapped']:
            break
        if num_images >= 0 and n >= num_images:
            break

    lang_stats = None
    if lang_eval == 1:
        lang_stats = language_eval(dataset, predictions, eval_kwargs.get('id'), eval_kwargs.get('image_root'), split)

    # Switch back to training mode
    model.train()
    return loss_sum/loss_evals, predictions, lang_stats
