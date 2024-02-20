import argparse

def parse_opt():
    parser = argparse.ArgumentParser()
    # the primary input data information for one stage model
    parser.add_argument('--input_json', type=str, default="/shenyiming/wangbo/Dataset/MS_COCO/cocotalk.json",
                        help='path to the json file containing information info and vocab')
    parser.add_argument('--input_images', type=str, default='/shenyiming/wangbo/Dataset/MS_COCO/MS_COCO_384',
                        help='path to the directory containing the images')
    parser.add_argument('--input_label_h5', type=str,
                        default='/shenyiming/wangbo/Dataset/MS_COCO/cocotalk_label.h5',
                        help='path to the h5file containing the preprocessed dataset')
    parser.add_argument('--cached_tokens', type=str,
                        default='/shenyiming/wangbo/Dataset/MS_COCO/coco-train-idxs.p',
                        help='Cached token file for calculating cider score during self critical training.')
    parser.add_argument('--start_from', type=str, default='/shenyiming/wangbo/MM_Experiments/CropCap/HFormer_cascade_segatt_E2/save/4e-4/xe',  # #############################################
                        help="""continue training from saved checkpoint at this path. Path must contain files saved by previous training process:
                            'infos.pkl'         : configuration;
                            'checkpoint'        : paths to model file(s) (created by tf).
                            Note: this file contains absolute paths, be careful when moving files around;
                            'model.ckpt-*'      : file(s) with model definition (created by tf)""")
    parser.add_argument('--pretrain_swin', default='swin-L', type=str,
                        help='resnert18, resnet34, resnet101, resnet152, swin-T, swin-S, swin-B, swin-L')
    parser.add_argument('--swin_pretrain_path', type=str, default='/shenyiming/wangbo/MM_Experiments/CropCap/HFormer_cascade_segatt_E2/models/swin/swin/swin_large_patch4_window12_384_22kto1k_no_head.pth',
                        help='the path of pretrained swin transformer')
    parser.add_argument('--feats_0_total', type=list, default=[2304, 384, 884736], help='2304 * 384')
    parser.add_argument('--feats_1_total', type=list, default=[576, 768, 1327104], help='2304 * 384 + 576 * 768')
    parser.add_argument('--feats_2_total', type=int, default=[144, 1536, 1548288], help='2304 * 384 + 576 * 768 + 144 * 1536')
    parser.add_argument('--feats_3_total', type=int, default=[144, 1536, 1769472], help='2304 * 384 + 576 * 768 + 144 * 1536 + 144 * 1536')

    # Model settings
    parser.add_argument('--caption_model', type=str, default="hformer",
                        help='transformer, osic, hformer')

    parser.add_argument('--rnn_size', type=int, default=2048,
                        help='size of the rnn in number of hidden nodes in each layer')
    parser.add_argument('--num_layers', type=int, default=1, help='number of layers of the encoder and decoder')
    parser.add_argument('--rnn_type', type=str, default='lstm', help='rnn, gru, lstm, or multi-self attention')
    parser.add_argument('--input_encoding_size', type=int, default=512,  # 2048
                        help='the embedding size of each token in the vocabulary, and the image.')
    parser.add_argument('--word_embed_dimension', type=int, default=512,  # 1000
                        help='the embedding size of each token in the vocabulary')
    parser.add_argument('--att_hidden_size', type=int, default=512,
                        help='')

    parser.add_argument('--transformer_pe_max_len', type=int, default=5000,
                        help='')

    # model settings for the lstm, or rnn model
    parser.add_argument('--att_feat_size', type=int, default=2048, help='2048 for resnet, 512 for vgg')

    # parse settings for training the models

    parser.add_argument('--use_bn', type=int, default=0,
                        help='If 1, then do batch_normalization first in att_embed, '
                             'if 2 then do bn both in the beginning and the end of att_embed')

    parser.add_argument('--use_feats_channel_att', type=bool, default=True,
                        help='if True, then fusing the features channel attention besides spatial position attention; '
                             'if False, then only using the spatial position attention in msa module')

    # Optimization: General
    parser.add_argument('--max_epochs', type=int, default=25, help='number of epochs')  # #############################
    parser.add_argument('--max_sentence_length', type=int, default=17, help='max number words of sentence')
    parser.add_argument('--vocab_size', type=int, default=9487, help='number of words in the built vocabulary')
    parser.add_argument('--batch_size', type=int, default=8, help='minibatch size')
    parser.add_argument('--grad_clip', type=float, default=0.1, help='clip gradients at this value')
    parser.add_argument('--drop_prob_lm', type=float, default=0.5, help='strength of dropout in the Language Model RNN')
    parser.add_argument('--self_critical_after', type=int, default=11,  # #############################################
                        help='After what epoch do we start finetuning the CNN? '
                             '(-1 = disable; never finetune, 0 = finetune from start)')
    parser.add_argument('--seq_per_img', type=int, default=5,
                        help='number of captions to sample for each image during training. Done for efficiency '
                             'since CNN forward pass is expensive. E.g. coco has 5 sents/image')
    parser.add_argument('--beam_size', type=int, default=1,
                        help='used when sample_max = 1, indicates number of beams in beam search. Usually 2 or 3 works '
                             'well. More is not better. Set this to 1 for faster runtime but a bit worse performance.')

    # Optimization: for the Language Model
    parser.add_argument('--optim', type=str, default='adam', help='what update to use? rmsprop|sgd|sgdmom|adagrad|adam')
    parser.add_argument('--learning_rate', type=float, default=4e-4, help='learning rate')
    parser.add_argument('--learning_rate_decay_start', type=int, default=7,
                        help='at what iteration to start decaying learning rate? (-1 = dont) (in epoch)')
    parser.add_argument('--learning_rate_decay_every', type=int, default=2,
                        help='every how many iterations thereafter to drop LR?(in epoch)')
    parser.add_argument('--learning_rate_decay_xl_every', type=int, default=5,  # ################# 3 #################
                        help='every how many iterations thereafter to drop LR?(in epoch)')
    parser.add_argument('--learning_rate_decay_rate', type=float, default=0.1,
                        help='every how many iterations thereafter to drop LR?(in epoch)')
    parser.add_argument('--optim_alpha', type=float, default=0.9, help='alpha for adam')
    parser.add_argument('--optim_beta', type=float, default=0.999, help='beta used for adam')
    parser.add_argument('--optim_epsilon', type=float, default=1e-8,
                        help='epsilon that goes into denominator for smoothing')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight_decay')  # 0.0005

    parser.add_argument('--scheduled_sampling_start', type=int, default=0,
                        help='at what iteration to start decay gt probability')
    parser.add_argument('--scheduled_sampling_increase_every', type=int, default=3,
                        help='every how many iterations thereafter to gt probability')
    parser.add_argument('--scheduled_sampling_increase_prob', type=float, default=0.05,
                        help='How much to update the prob')
    parser.add_argument('--scheduled_sampling_max_prob', type=float, default=0.25,
                        help='Maximum scheduled sampling prob.')

    # Evaluation/Checkpointing
    parser.add_argument('--val_images_use', type=int, default=-1,  # ######### -1 ########################
                        help='how many images to use when periodically evaluating the validation loss? (-1 = all)')
    parser.add_argument('--save_checkpoint_every', type=int, default=2500,  # ################## 2500 ################
                        help='how often to save a model checkpoint (in iterations)?')
    parser.add_argument('--save_checkpoint_start', type=int, default=0,
                        help='at what iteration to start save the trained model, only used when the parameters best decided')
    parser.add_argument('--checkpoint_path', type=str, default='/shenyiming/wangbo/MM_Experiments/CropCap/HFormer_cascade_segatt_E2/save/4e-4/xe_rl_4_4e-5',  # ################################################
                        help='directory to store checkpointed models')
    parser.add_argument('--language_eval', type=int, default=1,
                        help='Evaluate language as well (1 = yes, 0 = no)? BLEU/CIDEr/METEOR/ROUGE_L? '
                             'requires coco-caption code from Github.')
    parser.add_argument('--losses_log_every', type=int, default=100,
                        help='How often do we snapshot losses, for inclusion in the progress dump? (0 = disable)')
    parser.add_argument('--losses_print_every', type=int, default=50,
                        help='How often do we snapshot losses, for inclusion in the progress dump? (0 = disable)')
    parser.add_argument('--load_best_score', type=int, default=1,
                        help='Do we load previous best score when resuming training.')

    # misc
    parser.add_argument('--id', type=str, default='hformer_xe',
                        help='an id identifying this run/job. used in cross-val '
                             'and appended when writing progress files')
    parser.add_argument('--train_only', type=int, default=0, help='if true then use 80k, else use 110k')

    # Reward
    parser.add_argument('--cider_reward_weight', type=float, default=1, help='The reward weight from cider')
    parser.add_argument('--bleu_reward_weight', type=float, default=0, help='The reward weight from bleu4')

    # Transformer
    parser.add_argument('--label_smoothing', type=float, default=0, help='')
    parser.add_argument('--noamopt_warmup', type=int, default=20000, help='')

    # unconcerned
    parser.add_argument('--noamopt', action='store_true', help='')
    parser.add_argument('--noamopt_factor', type=float, default=1, help='')
    parser.add_argument('--reduce_on_plateau', action='store_true', help='')

    parser.add_argument("--legacy_extra_skip", type=str2bool, default=False,
                        help=("true only for certain legacy models that have an extra skip connection"))

    args = parser.parse_args()

    # Check if args are valid
    assert args.rnn_size > 0, "rnn_size should be greater than 0"
    assert args.num_layers > 0, "num_layers should be greater than 0"
    assert args.input_encoding_size > 0, "input_encoding_size should be greater than 0"
    assert args.batch_size > 0, "batch_size should be greater than 0"
    assert args.drop_prob_lm >= 0 and args.drop_prob_lm < 1, "drop_prob_lm should be between 0 and 1"
    assert args.seq_per_img > 0, "seq_per_img should be greater than 0"
    assert args.beam_size > 0, "beam_size should be greater than 0"
    assert args.save_checkpoint_every > 0, "save_checkpoint_every should be greater than 0"
    assert args.losses_log_every > 0, "losses_log_every should be greater than 0"
    assert args.language_eval == 0 or args.language_eval == 1, "language_eval should be 0 or 1"
    assert args.load_best_score == 0 or args.load_best_score == 1, "language_eval should be 0 or 1"
    assert args.train_only == 0 or args.train_only == 1, "language_eval should be 0 or 1"

    return args


def str2bool(v):
    if v.lower() in ('true', '1'):
        return True
    elif v.lower() in ('false', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
