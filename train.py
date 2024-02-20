from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
import models
from models.Swin import mySwin
from dataloader_image import *
import eval_utils
import misc.utils as utils
from misc.rewards import init_scorer, get_self_critical_reward
import time


try:
    import tensorboardX as tb
except ImportError:
    print("tensorboardX is not installed")
    tb = None


def setupseed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True  # cudnn


# visualization during training
def add_summary_value(writer, key, value, iteration):
    if writer:
        writer.add_scalar(key, value, iteration)


def train(opt):
    # loader = DataLoader(opt)
    loader = DataLoaderX(opt)

    opt.vocab_size = loader.vocab_size
    opt.seq_length = loader.seq_length

    print("vocab_size :", opt.vocab_size)
    print("seq_length :", opt.seq_length)

    tb_summary_writer = tb and tb.SummaryWriter(opt.checkpoint_path)
    infos = {}
    histories = {}
    if opt.start_from is not None:  # skipped
        # open old infos and check if models are compatible
        with open(os.path.join(opt.start_from, 'infos_'+opt.id+'.pkl'), "rb") as f:
            infos = pickle.load(f)
            saved_model_opt = infos['opt']
            need_be_same = ["caption_model", "rnn_type", "rnn_size", "num_layers"]
            for checkme in need_be_same:
                assert vars(saved_model_opt)[checkme] == vars(opt)[checkme], \
                    "Command line argument and saved model disagree on '%s' " % checkme

        if os.path.isfile(os.path.join(opt.start_from, 'histories_'+opt.id+'.pkl')):
            with open(os.path.join(opt.start_from, 'histories_'+opt.id+'.pkl'), "rb") as f:
                histories = pickle.load(f)

    iteration = infos.get('iter', 0)
    epoch = infos.get('epoch', 0)

    val_result_history = histories.get('val_result_history', {})
    loss_history = histories.get('loss_history', {})
    lr_history = histories.get('lr_history', {})
    ss_prob_history = histories.get('ss_prob_history', {})

    loader.iterators = infos.get('iterators', loader.iterators)
    loader.split_ix = infos.get('split_ix', loader.split_ix)
    if opt.load_best_score == 1:
        best_val_score = infos.get('best_val_score', None)

    my_swin = mySwin().cuda()
    my_swin = torch.nn.DataParallel(my_swin)
    my_swin.eval()

    model = models.setup(opt).cuda()
    dp_model = torch.nn.DataParallel(model)

    epoch_done = True
    # Assure in training mode
    dp_model.train()

    if opt.label_smoothing > 0:  # skipped
        crit = utils.LabelSmoothing(smoothing=opt.label_smoothing)
    else:
        crit = utils.LanguageModelCriterion()
    rl_crit = utils.RewardCriterion()

    if opt.noamopt:  # skipped
        assert opt.caption_model == 'transformer' or opt.caption_model == 'rainformer', 'noamopt can only work with transformer'
        optimizer = utils.get_std_opt(model, factor=opt.noamopt_factor, warmup=opt.noamopt_warmup)
        optimizer._step = iteration
    elif opt.reduce_on_plateau:  # skipped
        optimizer = utils.build_optimizer(model.parameters(), opt)
        optimizer = utils.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)
    else:
        optimizer = utils.build_optimizer(model.parameters(), opt)
    # Load the optimizer
    if vars(opt).get('start_from', None) is not None and os.path.isfile(os.path.join(opt.start_from, "optimizer.pth")):
        optimizer.load_state_dict(torch.load(os.path.join(opt.start_from, 'optimizer.pth')))

    while True:
        if epoch_done:
            # Assign the scheduled sampling prob
            if epoch > opt.scheduled_sampling_start and opt.scheduled_sampling_start >= 0:
                frac = (epoch - opt.scheduled_sampling_start) // opt.scheduled_sampling_increase_every
                opt.ss_prob = min(opt.scheduled_sampling_increase_prob * frac, opt.scheduled_sampling_max_prob)
                model.ss_prob = opt.ss_prob
            if opt.self_critical_after != -1 and epoch >= opt.self_critical_after:
                sc_flag = True
                init_scorer(opt.cached_tokens)
            else:
                sc_flag = False

            epoch_done = False

        if not opt.noamopt and not opt.reduce_on_plateau:
            # Assign the learning rate
            if opt.noamopt_warmup and iteration < opt.noamopt_warmup:
                # print("168 iteration is :", iteration)
                warmup_percent = iteration / opt.noamopt_warmup
                warmup_lr = opt.learning_rate * warmup_percent
                opt.current_lr = warmup_lr
            else:
                if epoch > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0:
                    frac = ((epoch - opt.learning_rate_decay_start - 1) // opt.learning_rate_decay_every) + 1
                    decay_factor = opt.learning_rate_decay_rate ** frac
                    opt.current_lr = opt.learning_rate * decay_factor
                    # if epoch > 11:
                    #     decay_factor = opt.learning_rate_decay_rate ** 4
                    #     opt.current_lr = opt.learning_rate * decay_factor
                    if epoch > 10:
                        frac = ((epoch - 11) // 4) + 1
                        decay_factor = opt.learning_rate_decay_rate ** frac
                        opt.current_lr = opt.learning_rate * decay_factor

                        # ##############################################
                    # if epoch > 16:
                    #     frac = ((epoch - 17) // 2) + 3
                    #     decay_factor = opt.learning_rate_decay_rate ** frac
                    #     opt.current_lr = opt.learning_rate * decay_factor
                    else:
                        pass

                else:
                    opt.current_lr = opt.learning_rate
            # set the decayed rate
            utils.set_lr(optimizer, opt.current_lr)  # set the decayed rate

        # Load data from train split (0)
        data = loader.get_batch('train')

        torch.cuda.synchronize()
        start = time.time()
        tmp = [data['input_images'], data['labels'], data['labels_masks']]
        tmp = [_ if _ is None else torch.from_numpy(_).cuda() for _ in tmp]
        input_images, labels, labels_masks = tmp

        optimizer.zero_grad()
        with torch.no_grad():  # ################### with torch.no_grad(): ##################################
            x_0, x_1, x_2, x_3 = my_swin(input_images)
        # print("x_0, x_1, x_2, x_3 size are :{}, {}, {}, {}".format(x_0.size(), x_1.size(), x_2.size(), x_3.size()))
        # print("156 x_0, x_1, x_2, x_3 device are :{}, {}, {}, {}".format(x_0.device, x_1.device, x_2.device, x_3.device))
        """[40, 2304, 384], [40, 576, 768], [40, 144, 1536], [40, 144, 1536]"""

        if not sc_flag:
            loss = crit(dp_model(x_0, x_1, x_2, x_3, labels), labels[:, 1:], labels_masks[:, 1:])
        else:
            gen_result, sample_logprobs = dp_model(x_0, x_1, x_2, x_3, opt={'sample_max': 0}, mode='sample')
            reward = get_self_critical_reward(dp_model, x_0, x_1, x_2, x_3, None, data, gen_result, opt)

            loss = rl_crit(sample_logprobs, gen_result.data, torch.from_numpy(reward).float().cuda())

        loss.backward()

        # variable gradient clipping
        utils.clip_gradient(optimizer, opt.grad_clip)

        optimizer.step()
        train_loss = loss.item()
        torch.cuda.synchronize()
        end = time.time()
        if not sc_flag:
            if (iteration % opt.losses_print_every == 0):
                print("iter {} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}"
                      .format(iteration, epoch, train_loss, end - start))
        else:
            if (iteration % opt.losses_print_every == 0):
                print("iter {} (epoch {}), avg_reward = {:.3f}, time/batch = {:.3f}"
                      .format(iteration, epoch, np.mean(reward[:, 0]), end - start))

        # Update the iteration and epoch
        iteration += 1
        if data['bounds']['wrapped']:
            epoch += 1
            epoch_done = True

        if (iteration % opt.losses_log_every == 0):
            add_summary_value(tb_summary_writer, 'train_loss', train_loss, iteration)
            if opt.noamopt:
                opt.current_lr = optimizer.rate()
            elif opt.reduce_on_plateau:
                opt.current_lr = optimizer.current_lr
            add_summary_value(tb_summary_writer, 'learning_rate', opt.current_lr, iteration)

            add_summary_value(tb_summary_writer, 'scheduled_sampling_prob', model.ss_prob, iteration)
            if sc_flag:
                add_summary_value(tb_summary_writer, 'avg_reward', np.mean(reward[:, 0]), iteration)

            loss_history[iteration] = train_loss if not sc_flag else np.mean(reward[:, 0])
            lr_history[iteration] = opt.current_lr

            ss_prob_history[iteration] = model.ss_prob

        # make evaluation on validation set, and save model
        if (iteration % opt.save_checkpoint_every == 0) and iteration > opt.save_checkpoint_start:
            # eval model
            eval_kwargs = {'split': 'val', 'dataset': opt.input_json}
            eval_kwargs.update(vars(opt))
            val_loss, predictions, lang_stats = eval_utils.eval_split(my_swin, dp_model, crit, loader, eval_kwargs)

            if opt.reduce_on_plateau:
                if 'CIDEr' in lang_stats:
                    optimizer.scheduler_step(-lang_stats['CIDEr'])
                else:
                    optimizer.scheduler_step(val_loss)

            # Write validation result into summary
            add_summary_value(tb_summary_writer, 'validation loss', val_loss, iteration)
            if lang_stats:
                for k, v in lang_stats.items():
                    add_summary_value(tb_summary_writer, k, v, iteration)
            val_result_history[iteration] = {'loss': val_loss, 'lang_stats': lang_stats, 'predictions': predictions}

            # Save model if is improving on validation result
            if opt.language_eval == 1:
                current_score = lang_stats['CIDEr']
            else:
                current_score = - val_loss

            best_flag = False
            if True:  # if true
                if best_val_score is None or current_score > best_val_score:
                    best_val_score = current_score
                    best_flag = True

                if not os.path.isdir(opt.checkpoint_path):
                    os.makedirs(opt.checkpoint_path)
                checkpoint_path = os.path.join(opt.checkpoint_path, 'model.pth')
                torch.save(model.state_dict(), checkpoint_path)
                print("model saved to {}".format(checkpoint_path))
                optimizer_path = os.path.join(opt.checkpoint_path, 'optimizer.pth')
                torch.save(optimizer.state_dict(), optimizer_path)

                # Dump miscalleous informations
                infos['iter'] = iteration
                infos['epoch'] = epoch
                infos['iterators'] = loader.iterators
                infos['split_ix'] = loader.split_ix
                infos['best_val_score'] = best_val_score
                infos['opt'] = opt
                infos['vocab'] = loader.get_vocab()

                histories['val_result_history'] = val_result_history
                histories['loss_history'] = loss_history
                """
                histories['lr_history_base'] = lr_history_base
                histories['lr_history_new'] = lr_history_new
                """
                histories['lr_history'] = lr_history
                histories['ss_prob_history'] = ss_prob_history
                with open(os.path.join(opt.checkpoint_path, 'infos_'+opt.id+'.pkl'), 'wb') as f:
                    pickle.dump(infos, f)
                with open(os.path.join(opt.checkpoint_path, 'histories_'+opt.id+'.pkl'), 'wb') as f:
                    pickle.dump(histories, f)

                if best_flag:
                    checkpoint_path = os.path.join(opt.checkpoint_path, 'model-best.pth')
                    torch.save(model.state_dict(), checkpoint_path)
                    print("model saved to {}".format(checkpoint_path))
                    with open(os.path.join(opt.checkpoint_path, 'infos_'+opt.id+'-best.pkl'), 'wb') as f:
                        pickle.dump(infos, f)

        # Stop if reaching max epochs
        if epoch >= opt.max_epochs and opt.max_epochs != -1:
            break


opt = opts.parse_opt()
setupseed(42)
train(opt)
