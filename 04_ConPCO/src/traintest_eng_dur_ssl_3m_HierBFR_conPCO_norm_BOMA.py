# -*- coding: utf-8 -*-
# Archived training entry point. Loss, split, and checkpoint rules are retained.
# See docs/REPRODUCIBILITY.md and use tools/run_experiment.py for a fresh output path.
# @Author  : Bi-Cheng Yan
# @Affiliation  : National Taiwan Normal University
# @Email   : bicheng@ntnu.edu.tw
# @File    : traintest_eng_dur_ssl_3m_HierBFR_conPCO_norm.py

# train and test the models
import sys
import os
import json
import numpy as np
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler # Weighted sampler import; usage depends on the saved experiment.
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
import random
import argparse

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from models.conPCO_norm import ContrastivePhonemicOrdinalRegularizer

# ==============================================================
# BALANCED LOSS FUNCTIONS FOR REGRESSION AND CLASSIFICATION
# ==============================================================
class BalancedMSELoss(nn.Module):
    """Weight squared errors inversely to their target-bin frequency within the current batch."""
    def __init__(self, bins=10):
        super().__init__()
        self.bins = bins

    def forward(self, pred, target, mask=None):
        if mask is not None:
            pred = pred[mask]
            target = target[mask]
            
        if target.numel() == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        mse = (pred - target) ** 2
        
        with torch.no_grad():
            min_val, max_val = target.min(), target.max()
            if min_val == max_val:
                return mse.mean()
            
            # Place dynamic bin boundaries across the target range in this batch.
            bin_edges = torch.linspace(min_val, max_val + 1e-5, self.bins + 1, device=target.device)
            indices = torch.bucketize(target, bin_edges) - 1
            indices = torch.clamp(indices, 0, self.bins - 1)
            
            # Count observations and compute inverse-frequency weights.
            counts = torch.bincount(indices.flatten(), minlength=self.bins).float()
            counts = torch.clamp(counts, min=1.0) # Avoid division by zero.
            weights = 1.0 / counts
            
            # Assign the corresponding weight to each observation.
            sample_weights = weights[indices]
            
            # Normalize weights to preserve the average scale of the loss.
            sample_weights = sample_weights / sample_weights.mean()
            
        return torch.mean(mse * sample_weights)

# ==============================================================

def calc_macro_mse(preds, targets, edges):
    """Compute the mean squared error averaged over the selected target-score intervals."""
    indices = np.digitize(targets, edges)
    num_bins = len(edges) + 1
    
    mse_por_rango = []
    
    for k in range(num_bins):
        mask = (indices == k)
        if np.any(mask):
            mse_k = np.mean((preds[mask] - targets[mask]) ** 2)
            mse_por_rango.append(mse_k)
            
    if not mse_por_rango:
        return 0.0
    return float(np.mean(mse_por_rango))

def ccc_score(y_true, y_pred):
    if np.std(y_true) <= 1e-6 or np.std(y_pred) <= 1e-6:
        return np.nan
        
    cor = np.corrcoef(y_true, y_pred)[0][1]
    if np.isnan(cor):
        return np.nan
        
    mean_true, mean_pred = np.mean(y_true), np.mean(y_pred)
    var_true, var_pred = np.var(y_true), np.var(y_pred)
    sd_true, sd_pred = np.std(y_true), np.std(y_pred)
    
    denominator = var_true + var_pred + (mean_true - mean_pred)**2
    if denominator == 0:
        return np.nan
        
    numerator = 2 * cor * sd_true * sd_pred
    return numerator / denominator

def bootstrap_adaptive_balanced_ccc(gt, pred, bins, n_iterations=1000, min_power=10, multiplier=5.0):
    pred = pred * multiplier 
    gt = gt * multiplier 

    valid_bins = [b for b in bins if not (b[0] == 0.0 and b[1] == bins[-1][1])]
    indices_por_bin = []
    tamanos_reales = []
    
    for min_g, max_g in valid_bins:
        if min_g == valid_bins[0][0]:
            mask = (gt >= min_g) & (gt <= max_g)
        else:
            mask = (gt > min_g) & (gt <= max_g)
            
        idx = np.where(mask)[0]
        if len(idx) > 0:
            indices_por_bin.append(idx)
            tamanos_reales.append(len(idx))
            
    if len(indices_por_bin) < 2:
        return 0, 0

    n_target_per_interval = int(np.median(tamanos_reales))
    n_target_per_interval = max(min_power, n_target_per_interval)

    ccc_scores = []
    
    for _ in range(n_iterations):
        muestra_gt = []
        muestra_pred = []
        
        for idx_array in indices_por_bin:
            idx_sampleados = np.random.choice(idx_array, size=n_target_per_interval, replace=True)
            muestra_gt.extend(gt[idx_sampleados])
            muestra_pred.extend(pred[idx_sampleados])
            
        muestra_gt = np.array(muestra_gt)
        muestra_pred = np.array(muestra_pred)
        
        if np.std(muestra_gt) > 1e-6 and np.std(muestra_pred) > 1e-6:
            ccc_val = ccc_score(muestra_gt, muestra_pred)
            if not np.isnan(ccc_val):
                ccc_scores.append(ccc_val)
                
    if not ccc_scores:
        return np.nan, n_target_per_interval
        
    return np.mean(ccc_scores), n_target_per_interval


print("I am process %s, running on %s: starting (%s)" % (os.getpid(), os.uname()[1], time.asctime()))

def get_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--exp-dir", type=str, default="./exp/", help="directory to dump experiments")
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument("--n-epochs", type=int, default=100, help="number of maximum training epochs")
    parser.add_argument("--p_depth", type=int, default=1, help="depth of hiercb model")
    parser.add_argument("--w_depth", type=int, default=1, help="depth of hiercb model")
    parser.add_argument("--u_depth", type=int, default=1, help="depth of hiercb models")
    parser.add_argument("--hiercbheads", type=int, default=1, help="heads of hiercb model")
    parser.add_argument("--batch_size", type=int, default=25, help="training batch size")
    parser.add_argument("--embed_dim", type=int, default=12, help="hiercb transformer embedding dimension")
    parser.add_argument("--ssl_drop", type=float, default=0.1, help="hiercb transformer embedding dimension")
    parser.add_argument("--loss_w_phn", type=float, default=1, help="weight for phoneme-level loss")
    parser.add_argument("--loss_w_pco", type=float, default=1, help="weight for phoneme-level loss")
    parser.add_argument("--loss_w_clap", type=float, default=1, help="weight for phoneme-level loss")
    parser.add_argument("--pco_ld", type=float, default=5.0, help="weight for phoneme-level loss")
    parser.add_argument("--pco_lt", type=float, default=0.1, help="weight for phoneme-level loss")
    parser.add_argument("--pco_mg", type=float, default=1.0, help="weight for phoneme-level loss")
    parser.add_argument("--clap_t2a", type=float, default=0.1, help="weight for phoneme-level loss")
    parser.add_argument("--loss_w_word", type=float, default=1, help="weight for word-level loss")
    parser.add_argument("--loss_w_utt", type=float, default=1, help="weight for utterance-level loss")
    parser.add_argument("--model", type=str, default='gopt', help="name of the model")
    parser.add_argument("--am", type=str, default='librispeech', help="name of the acoustic models")
    parser.add_argument("--noise", type=float, default=0., help="the scale of random noise added on the input GoP feature")
    parser.add_argument("--conpco", action='store_true', help="whether to use the ConPCO regularizer")

    args = parser.parse_args()

    return args

# just to generate the header for the result.csv
def gen_result_header():
    phn_header = ['epoch', 'phone_train_mse', 'phone_train_pcc', 'phone_test_mse', 'phone_test_pcc', 'learning rate']
    utt_header_set = ['utt_train_mse', 'utt_train_pcc', 'utt_test_mse', 'utt_test_pcc']
    utt_header_score = ['accuracy', 'completeness', 'fluency', 'prosodic', 'total']
    word_header_set = ['word_train_pcc', 'word_test_pcc']
    word_header_score = ['accuracy', 'stress', 'total']
    utt_header, word_header = [], []
    for dset in utt_header_set:
        utt_header = utt_header + [dset+'_'+x for x in utt_header_score]
    for dset in word_header_set:
        word_header = word_header + [dset+'_'+x for x in word_header_score]
    header = phn_header + utt_header + word_header
    return header

def train(audio_model, train_loader, val_loader, test_loader, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('running on ' + str(device))

    # best_cum_mAP is checkpoint ensemble from the first epoch to the best epoch
    best_epoch, best_mse = 0, 999
    global_step, epoch = 0, 0
    exp_dir = getattr(args, 'current_exp_dir', args.exp_dir)

    if not isinstance(audio_model, nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)

    audio_model = audio_model.to(device)
    # Set up the optimizer
    trainables = [p for p in audio_model.parameters() if p.requires_grad]
    print('Total parameter number is : {:.3f} k'.format(sum(p.numel() for p in audio_model.parameters()) / 1e3))
    print('Total trainable parameter number is : {:.3f} k'.format(sum(p.numel() for p in trainables) / 1e3))
    optimizer = torch.optim.Adam(trainables, args.lr, weight_decay=5e-7, betas=(0.95, 0.999))

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)

    loss_fn = BalancedMSELoss(bins=10).to(device)
    loss_pco = ContrastivePhonemicOrdinalRegularizer(args.pco_ld, args.pco_lt, args.clap_t2a, args.pco_mg)

    print("current #steps=%s, #epochs=%s" % (global_step, epoch))
    print("start training...")
    result = np.zeros([args.n_epochs, 32])

    while epoch < args.n_epochs:
        audio_model.train()
        s_t = int(round(time.time()))
        for i, (audio_input, audio_input_ssl1, audio_input_ssl2, audio_input_ssl3, audio_input_eng, audio_input_dur, phn_label, phns, utt_label, word_label, word_id) in enumerate(train_loader):

            audio_input = audio_input.to(device, non_blocking=True)
            audio_input_ssl1 = audio_input_ssl1.to(device, non_blocking=True)
            audio_input_ssl2 = audio_input_ssl2.to(device, non_blocking=True)
            audio_input_ssl3 = audio_input_ssl3.to(device, non_blocking=True)
            audio_input_eng = audio_input_eng.to(device, non_blocking=True)
            audio_input_dur = audio_input_dur.to(device, non_blocking=True)
            phns = phns.to(device, non_blocking=True)
            word_id = word_id.to(device, non_blocking=True)

            audio_input_ssl = torch.cat([audio_input_ssl2, audio_input_ssl1, audio_input_ssl3], dim=-1)

            phn_label = phn_label.to(device, non_blocking=True)
            utt_label = utt_label.to(device, non_blocking=True)
            word_label = word_label.to(device, non_blocking=True)

            # warmup
            warm_up_step = 100
            if global_step <= warm_up_step and global_step % 5 == 0:
                warm_lr = (global_step / warm_up_step) * args.lr
                for param_group in optimizer.param_groups:
                    param_group['lr'] = warm_lr
                print('warm-up learning rate is {:f}'.format(optimizer.param_groups[0]['lr']))

            # add random noise for augmentation.
            noise = (torch.rand([audio_input.shape[0], audio_input.shape[1], audio_input.shape[2]]) - 1) * args.noise
            noise = noise.to(device, non_blocking=True)
            audio_input = audio_input + noise

            #print(phns.shape)
            u1, u2, u3, u4, u5, p, w1, w2, w3, phn_audio_feats, phn_text_feats = audio_model(audio_input, audio_input_eng, audio_input_dur, audio_input_ssl, phns, word_label[:, :, -1], word_id)
            
            mask_phn = (phn_label >= 0)
            p = p.squeeze(2)
            loss_phn = loss_fn(p, phn_label, mask=mask_phn)
                            
            # performs PCO-loss
            if args.conpco:
                loss_phn_pco, loss_center_clap = loss_pco(phn_audio_feats, phn_text_feats, phn_label, phns)

            # utterance level loss, also mse
            utt_preds = torch.cat((u1, u2, u3, u4, u5), dim=1)
            mask_utt = (utt_label >= 0)
            loss_utt = loss_fn(utt_preds, utt_label, mask=mask_utt)
                                        
            word_label_targets = word_label[:, :, 0:3]
            mask_word = (word_label_targets >= 0)
            word_pred = torch.cat((w1, w2, w3), dim=2)
            loss_word = loss_fn(word_pred, word_label_targets, mask=mask_word)                  

            if args.conpco:
                loss = (args.loss_w_phn * loss_phn +
                        args.loss_w_utt * loss_utt +
                        args.loss_w_word * loss_word +
                        args.loss_w_pco * loss_phn_pco +
                        args.loss_w_clap * loss_center_clap)
            else:
                loss = (args.loss_w_phn * loss_phn +
                        args.loss_w_utt * loss_utt +
                        args.loss_w_word * loss_word)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step += 1

        print('start validation')

        # ensemble results
        # don't save prediction for the training set
        tr_mse, tr_corr, tr_utt_mse, tr_utt_corr, tr_word_mse, tr_word_corr= validate(audio_model, train_loader, args, -1)
        val_mse, val_corr, val_utt_mse, val_utt_corr, val_word_mse, val_word_corr = validate(audio_model, val_loader, args, best_mse)

        print('Phone: Train MSE: {:.3f}, CORR: {:.3f}'.format(tr_mse, tr_corr))
        print('Phone: Val MSE: {:.3f}, CORR: {:.3f}'.format(val_mse, val_corr))
        print('Utterance:, ACC: {:.3f}, COM: {:.3f}, FLU: {:.3f}, PRO: {:.3f}, Total: {:.3f}'.format(val_utt_corr[0], val_utt_corr[1], val_utt_corr[2], val_utt_corr[3], val_utt_corr[4]))
        print('Word:, ACC: {:.3f}, Stress: {:.3f}, Total: {:.3f}'.format(val_word_corr[0], val_word_corr[1], val_word_corr[2]))
        print('-------------------Epoch finished-------------------')
        print()
        sys.stdout.flush()

        if val_mse < best_mse:
            best_mse = val_mse
            best_epoch = epoch
            best_lr = optimizer.param_groups[0]['lr']
             
            result[0, :6] = [epoch, tr_mse, tr_corr, val_mse, val_corr, best_lr]
            result[0, 6:26] = np.concatenate([tr_utt_mse, tr_utt_corr, val_utt_mse, val_utt_corr])
            result[0, 26:32] = np.concatenate([tr_word_corr, val_word_corr])
            
            header = ','.join(gen_result_header())
            np.savetxt(exp_dir + '/result.csv', result, delimiter=',', header=header, comments='')

        if best_epoch == epoch:
            if os.path.exists("%s/models/" % (exp_dir)) == False:
                os.mkdir("%s/models" % (exp_dir))
            torch.save(audio_model.state_dict(), "%s/models/best_audio_model.pth" % (exp_dir))

        if global_step > warm_up_step:
            scheduler.step(val_mse)
        
        epoch += 1
        
    print('-------------------training finished-------------------')
    
    if os.path.exists("%s/models/best_audio_model.pth" % (exp_dir)):
        audio_model.load_state_dict(torch.load("%s/models/best_audio_model.pth" % (exp_dir)))
        
    te_mse, te_corr, te_utt_mse, te_utt_corr, te_word_mse, te_word_corr = validate(audio_model, test_loader, args, 9999.0)

    print('FINAL TEST Phone: Test MSE: {:.3f}, CORR: {:.3f}'.format(te_mse, te_corr))
    print('FINAL TEST Utterance:, ACC: {:.3f}, COM: {:.3f}, FLU: {:.3f}, PRO: {:.3f}, Total: {:.3f}'.format(te_utt_corr[0], te_utt_corr[1], te_utt_corr[2], te_utt_corr[3], te_utt_corr[4]))
    print('FINAL TEST Word:, ACC: {:.3f}, Stress: {:.3f}, Total: {:.3f}'.format(te_word_corr[0], te_word_corr[1], te_word_corr[2]))
    
    tr_mse, tr_corr, tr_utt_mse, tr_utt_corr, tr_word_mse, tr_word_corr= validate(audio_model, train_loader, args, -1)

    result[0, :6] = [best_epoch, tr_mse, tr_corr, te_mse, te_corr, best_lr]
    result[0, 6:26] = np.concatenate([tr_utt_mse, tr_utt_corr, te_utt_mse, te_utt_corr])
    result[0, 26:32] = np.concatenate([tr_word_corr, te_word_corr])
    

    header = ','.join(gen_result_header())
    exp_dir = args.exp_dir
    np.savetxt(exp_dir + '/result.csv', result, delimiter=',', header=header, comments='')

    return te_mse, te_corr, te_utt_corr[0], te_utt_corr[1], te_utt_corr[2], te_utt_corr[3], te_utt_corr[4], te_word_corr[0], te_word_corr[1], te_word_corr[2]

def validate(audio_model, val_loader, args, best_mse):
    exp_dir = getattr(args, 'current_exp_dir', args.exp_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not isinstance(audio_model, nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)
    audio_model = audio_model.to(device)
    audio_model.eval()

    A_phn, A_phn_target = [], []
    A_u1, A_u2, A_u3, A_u4, A_u5, A_utt_target = [], [], [], [], [], []
    A_w1, A_w2, A_w3, A_word_target = [], [], [], []

    with torch.no_grad():
        for i, (audio_input, audio_input_ssl1, audio_input_ssl2, audio_input_ssl3, audio_input_eng, audio_input_dur, phn_label, phns, utt_label, word_label, word_id) in enumerate(val_loader):
            audio_input = audio_input.to(device)
            audio_input_ssl1 = audio_input_ssl1.to(device)
            audio_input_ssl2 = audio_input_ssl2.to(device)
            audio_input_ssl3 = audio_input_ssl3.to(device)
            audio_input_eng = audio_input_eng.to(device)
            audio_input_dur = audio_input_dur.to(device)
            word_id = word_id.to(device)
            word_label = word_label.to(device)

            audio_input_ssl = torch.cat([audio_input_ssl2, audio_input_ssl1, audio_input_ssl3], dim=-1)

            # compute output
            u1, u2, u3, u4, u5, p, w1, w2, w3, _, _ = audio_model(audio_input, audio_input_eng, audio_input_dur, audio_input_ssl, phns, word_label[:, :, -1], word_id)

            word_label = word_label.cpu()

            p = p.cpu().detach()
            u1, u2, u3, u4, u5 = u1.cpu().detach(), u2.cpu().detach(), u3.cpu().detach(), u4.cpu().detach(), u5.cpu().detach()
            w1, w2, w3 = w1.cpu().detach(), w2.cpu().detach(), w3.cpu().detach()

            A_phn.append(p)
            A_phn_target.append(phn_label)

            A_u1.append(u1)
            A_u2.append(u2)
            A_u3.append(u3)
            A_u4.append(u4)
            A_u5.append(u5)
            A_utt_target.append(utt_label)

            A_w1.append(w1)
            A_w2.append(w2)
            A_w3.append(w3)
            A_word_target.append(word_label)

        # phone level
        A_phn, A_phn_target = torch.cat(A_phn), torch.cat(A_phn_target)

        # utterance level
        A_u1, A_u2, A_u3, A_u4, A_u5, A_utt_target = torch.cat(A_u1), torch.cat(A_u2), torch.cat(A_u3), torch.cat(A_u4), torch.cat(A_u5), torch.cat(A_utt_target)

        # word level
        A_w1, A_w2, A_w3, A_word_target = torch.cat(A_w1), torch.cat(A_w2), torch.cat(A_w3), torch.cat(A_word_target)

        # get the scores
        phn_mse, phn_corr = valid_phn(A_phn, A_phn_target)

        A_utt = torch.cat((A_u1, A_u2, A_u3, A_u4, A_u5), dim=1)
        utt_mse, utt_corr = valid_utt(A_utt, A_utt_target)

        A_word = torch.cat((A_w1, A_w2, A_w3), dim=2)
        word_mse, word_corr, valid_word_pred, valid_word_target = valid_word(A_word, A_word_target)

        if phn_mse < best_mse:
            print('new best phn mse {:.3f}, now saving predictions.'.format(phn_mse))

            # create the directory
            if os.path.exists(exp_dir + '/preds') == False:
                os.mkdir(exp_dir + '/preds')

            # saving the phn target, only do once
            np.save(exp_dir + '/preds/phn_target.npy', A_phn_target)
            np.save(exp_dir + '/preds/word_target.npy', valid_word_target)
            np.save(exp_dir + '/preds/utt_target.npy', A_utt_target)

            np.save(exp_dir + '/preds/phn_pred.npy', A_phn)
            np.save(exp_dir + '/preds/word_pred.npy', valid_word_pred)
            np.save(exp_dir + '/preds/utt_pred.npy', A_utt)

    return phn_mse, phn_corr, utt_mse, utt_corr, word_mse, word_corr


def valid_phn(audio_output, target):
    valid_token_pred, valid_token_target = [], []
    audio_output = audio_output.squeeze(2)
    for i in range(audio_output.shape[0]):
        for j in range(audio_output.shape[1]):
            if target[i, j] >= 0:
                valid_token_pred.append(audio_output[i, j])
                valid_token_target.append(target[i, j])
    valid_token_target = np.array(valid_token_target)
    valid_token_pred = np.array(valid_token_pred)

    edges_phn = [0.5, 1.5]
    valid_token_mse = calc_macro_mse(valid_token_pred, valid_token_target, edges_phn)
    corr, _ = bootstrap_adaptive_balanced_ccc(valid_token_target, valid_token_pred, bins=[(0.0, 2.0), (2.0, 4.0), (4.0, 6.0), (6.0, 8.0), (8.0, 10.0)])
    return valid_token_mse, corr

def valid_utt(audio_output, target):
    mse, corr = [], []

    # Utterance score boundaries.
    utt_edges = [
        [2.0, 4.0, 6.0, 8.0], 
        [7.5],                
        [2.0, 4.0, 6.0, 8.0], 
        [2.0, 4.0, 6.0, 8.0], 
        [2.0, 4.0, 6.0, 8.0]  
    ]

    for i in range(5):
        cur_pred = audio_output[:, i].numpy()
        cur_target = target[:, i].numpy()

        cur_mse = calc_macro_mse(cur_pred, cur_target, utt_edges[i])
        cur_corr, _ = bootstrap_adaptive_balanced_ccc(target[:, i].numpy(), audio_output[:, i].numpy(), bins=[(0.0, 2.0), (2.0, 4.0), (4.0, 6.0), (6.0, 8.0), (8.0, 10.0)])
        mse.append(cur_mse)
        corr.append(cur_corr)
    return mse, corr

def valid_word(audio_output, target):
    word_id = target[:, :, -1]
    target = target[:, :, 0:3]

    valid_token_pred, valid_token_target = [], []

    for i in range(target.shape[0]):
        prev_w_id, start_id = 0, 0
        for j in range(target.shape[1]):
            cur_w_id = word_id[i, j].int()
            if cur_w_id != prev_w_id:
                valid_token_pred.append(np.mean(audio_output[i, start_id: j, :].numpy(), axis=0))
                valid_token_target.append(np.mean(target[i, start_id: j, :].numpy(), axis=0))
                if cur_w_id == -1: break
                else: prev_w_id = cur_w_id; start_id = j

    valid_token_pred = np.array(valid_token_pred)
    valid_token_target = np.array(valid_token_target).round(2)

    mse_list, corr_list = [], []

    word_edges = [
        [2.0, 4.0, 6.0, 8.0], 
        [7.5],                
        [2.0, 4.0, 6.0, 8.0]  
    ]

    for i in range(3):
        cur_pred = valid_token_pred[:, i]
        cur_target = valid_token_target[:, i]

        valid_token_mse = calc_macro_mse(cur_pred, cur_target, word_edges[i])
        corr, _ = bootstrap_adaptive_balanced_ccc(valid_token_target[:, i], valid_token_pred[:, i], bins=[(0.0, 2.0), (2.0, 4.0), (4.0, 6.0), (6.0, 8.0), (8.0, 10.0)])
        mse_list.append(valid_token_mse)
        corr_list.append(corr)
    return mse_list, corr_list, valid_token_pred, valid_token_target

class GoPDataset(Dataset):
    def __init__(self, set, am='librispeech'):
        # normalize the input to 0 mean and unit std.
        if am == 'librispeech':
            dir = 'seq_data_librispeech_v4'
            norm_mean, norm_std = 3.203, 4.045
            #fbank_mean, fbank_std = 0.0783, 0.5718
            energy_mean, energy_std = 0.1697, 0.4824
            dur_mean, dur_std = 0.1392, 0.0993
        else:
            raise ValueError('Acoustic Model Unrecognized.')

        if set == 'train':
            self.feat = torch.tensor(np.load('../data/'+dir+'/tr_feat.npy'), dtype=torch.float)
            self.feat_energy = torch.tensor(np.load('../data/'+dir+'/tr_energy_feat.npy'), dtype=torch.float)
            self.feat_dur = torch.tensor(np.load('../data/'+dir+'/tr_dur_feat.npy'), dtype=torch.float)
            self.feat_ssl1 = torch.tensor(np.load('../data/'+dir+'/tr_hubert_feat_v2.npy'), dtype=torch.float)
            self.feat_ssl2 = torch.tensor(np.load('../data/'+dir+'/tr_w2v_300m_feat_v2.npy'), dtype=torch.float)
            self.feat_ssl3 = torch.tensor(np.load('../data/'+dir+'/tr_wavlm_feat_v2.npy'), dtype=torch.float)
            self.phn_label = torch.tensor(np.load('../data/'+dir+'/tr_label_phn.npy'), dtype=torch.float)
            self.utt_label = torch.tensor(np.load('../data/'+dir+'/tr_label_utt.npy'), dtype=torch.float)
            self.word_label = torch.tensor(np.load('../data/'+dir+'/tr_label_word.npy'), dtype=torch.float)
            self.word_id = torch.tensor(np.load('../data/'+dir+'/tr_word_id.npy'), dtype=torch.float)
        elif set == 'test':
            self.feat = torch.tensor(np.load('../data/'+dir+'/te_feat.npy'), dtype=torch.float)
            self.feat_energy = torch.tensor(np.load('../data/'+dir+'/te_energy_feat.npy'), dtype=torch.float)
            self.feat_dur = torch.tensor(np.load('../data/'+dir+'/te_dur_feat.npy'), dtype=torch.float)
            self.feat_ssl1 = torch.tensor(np.load('../data/'+dir+'/te_hubert_feat_v2.npy'), dtype=torch.float)
            self.feat_ssl2 = torch.tensor(np.load('../data/'+dir+'/te_w2v_300m_feat_v2.npy'), dtype=torch.float)
            self.feat_ssl3 = torch.tensor(np.load('../data/'+dir+'/te_wavlm_feat_v2.npy'), dtype=torch.float)
            self.phn_label = torch.tensor(np.load('../data/'+dir+'/te_label_phn.npy'), dtype=torch.float)
            self.utt_label = torch.tensor(np.load('../data/'+dir+'/te_label_utt.npy'), dtype=torch.float)
            self.word_label = torch.tensor(np.load('../data/'+dir+'/te_label_word.npy'), dtype=torch.float)
            self.word_id = torch.tensor(np.load('../data/'+dir+'/te_word_id.npy'), dtype=torch.float)

        # normalize the GOP feature using the training set mean and std (only count the valid token features, exclude the padded tokens).
        self.feat = self.norm_valid(self.feat, norm_mean, norm_std)

        # normalize the utt_label to 0-2 (same with phn score range)
        self.utt_label = self.utt_label / 5
        # the last dim is word_id, so not normalizing
        self.word_label[:, :, 0:3] = self.word_label[:, :, 0:3] / 5
        self.phn_label[:, :, 1] = self.phn_label[:, :, 1]

    # only normalize valid tokens, not padded token
    def norm_valid(self, feat, norm_mean, norm_std):
        norm_feat = torch.zeros_like(feat)
        for i in range(feat.shape[0]):
            for j in range(feat.shape[1]):
                if feat[i, j, 0] != 0:
                    norm_feat[i, j, :] = (feat[i, j, :] - norm_mean) / norm_std
                else:
                    break
        return norm_feat

    def __len__(self):
        return self.feat.shape[0]

    def __getitem__(self, idx):
        # feat, phn_label, phn_id, utt_label, word_label
        return self.feat[idx, :], self.feat_ssl1[idx, :], self.feat_ssl2[idx, :], self.feat_ssl3[idx, :], self.feat_energy[idx, :], self.feat_dur[idx, :], self.phn_label[idx, :, 1], self.phn_label[idx, :, 0], self.utt_label[idx, :], self.word_label[idx, :], self.word_id[idx,:]


if __name__ == '__main__':
    args = get_arguments()
    # NOTE: set seed
    print("setting seed %d" %(args.seed))
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    am = args.am
    print('now train with {:s} acoustic models'.format(am))
    input_dim = 7 + 1 + 84 # 7 energy feature + 1 dur features + 84 GOP features

    # nowa is the best models used in this work
    def build_model(args):
        if args.model == 'hiercb':
            print('now train a HierCB models')
            from models.gopt_ssl_3m_bfr_cat_utt_clap import HierCB
            audio_mdl = HierCB(embed_dim=args.embed_dim, num_heads=args.hiercbheads, p_depth=args.p_depth, w_depth=args.w_depth, u_depth=args.u_depth, ssl_drop=args.ssl_drop, input_dim=input_dim)
            return audio_mdl

    tr_dataset = GoPDataset('train', am=am)
    tr_dataloader = DataLoader(tr_dataset, batch_size=args.batch_size, shuffle=True)
    te_dataset = GoPDataset('test', am=am)
    te_dataloader = DataLoader(te_dataset, batch_size=2500, shuffle=False)

    def get_multilabel_stratification_matrix(dataset):
        print("Construyendo matriz indicadora (Fonemas + Terciles Utterance)...")
        num_samples = len(dataset.utt_label)
        
        valid_ids = dataset.phn_label[:, :, 0]
        max_phn_id = int(torch.max(valid_ids).item())
        
        multilabel_matrix = np.zeros((num_samples, max_phn_id + 1), dtype=int)
        utt_scores = np.zeros(num_samples)
        
        for i in range(num_samples):
            phn_ids = dataset.phn_label[i, :, 0].numpy()
            valid_phn_ids = phn_ids[phn_ids >= 0].astype(int)
            for pid in valid_phn_ids:
                multilabel_matrix[i, pid] = 1

            u_score = dataset.utt_label[i].numpy() 
            valid_u_scores = u_score[u_score >= 0] if u_score.ndim > 0 else (np.array([u_score]) if u_score >= 0 else np.array([]))
            utt_scores[i] = np.mean(valid_u_scores) if len(valid_u_scores) > 0 else 0.0

        valid_utt_scores = utt_scores[utt_scores > 0]
        
        if len(valid_utt_scores) > 0:
            edges_utt = np.quantile(valid_utt_scores, [0.33, 0.66])
        else:
            edges_utt = np.array([0.5, 1.5]) 
            
        utt_classes = np.digitize(utt_scores, edges_utt)
        
        utt_onehot = np.zeros((num_samples, 3), dtype=int)
        utt_onehot[np.arange(num_samples), utt_classes] = 1
        
        final_strat_matrix = np.hstack((multilabel_matrix, utt_onehot))
        
        return final_strat_matrix, multilabel_matrix, utt_scores, edges_utt

    def save_distribution_plots(multilabel_matrix, utt_scores, edges_utt, output_dir, prefix):
        os.makedirs(output_dir, exist_ok=True)
        
        # ==========================================
        # 1. DICCIONARIOS PROPORCIONADOS
        # ==========================================

        with open('dicts/pureLabel_to_(0-41).json', 'r') as file:
            dict_phn_to_orig = json.load(file)
                
        with open('dicts/(3-41)_to_(0,numPhonesUsed).json', 'r') as file:
            dict_orig_to_new = json.load(file)
        

        # ==========================================
        # 2. INVERT THE PHONEME DICTIONARIES
        # ==========================================
        # Map the new ID (0-38) to the original ID ("3"-"41").
        dict_new_to_orig = {v: k for k, v in dict_orig_to_new.items()}
        
        # Map the original ID ("0"-"41") to its label ("W", "IY", etc.).
        dict_orig_to_phn = {v: k for k, v in dict_phn_to_orig.items()}

        phoneme_counts = np.sum(multilabel_matrix, axis=0)
        
        # ==========================================
        # 3. CONVERT INDICES TO LABELS
        # ==========================================
        phn_labels = []
        for i in range(len(phoneme_counts)):
            # Use .get() so unexpected IDs retain their original numeric value.
            orig_id = dict_new_to_orig.get(i, str(i))
            phn_char = dict_orig_to_phn.get(orig_id, str(orig_id))
            phn_labels.append(phn_char)

        # ==========================================
        # PLOT 1: PHONEME DISTRIBUTION
        # ==========================================
        # Increase the figure width from 10 to 12 to fit the 39 phoneme labels.
        plt.figure(figsize=(12, 5))
        plt.bar(range(len(phoneme_counts)), phoneme_counts, color='#1f77b4', edgecolor='black', alpha=0.8)
        
        # Rotate the text labels by 90 degrees to prevent overlap on the x axis.
        plt.xticks(ticks=range(len(phoneme_counts)), labels=phn_labels, rotation=90, fontsize=9)
        
        plt.title(f'{prefix} - Phoneme Distribution', fontsize=12, fontweight='bold')
        plt.xlabel('Phonemes', fontsize=10) # Use phoneme names as axis labels.
        plt.ylabel('Absolute Frequency', fontsize=10)
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{prefix.lower()}_phoneme_dist.png"), dpi=300)
        plt.close()
        
        # ==========================================
        # PLOT 2: UTTERANCE SCORE DISTRIBUTION
        # Retain the original plotting procedure.
        # ==========================================
        plt.figure(figsize=(7, 4))
        plt.hist(utt_scores, bins=np.linspace(0, 2.0, 21), color='#d62728', edgecolor='black', alpha=0.8)
        
        plt.axvline(edges_utt[0], color='black', linestyle='dashed', linewidth=1.5, alpha=0.8, label=f'P33: {edges_utt[0]:.2f}')
        plt.axvline(edges_utt[1], color='black', linestyle='dashed', linewidth=1.5, alpha=0.8, label=f'P66: {edges_utt[1]:.2f}')

        plt.title(f'{prefix} -  Utterance Total Score Distribution', fontsize=12, fontweight='bold')
        plt.xlabel('Score Value', fontsize=10)
        plt.ylabel('Count', fontsize=10)
        plt.xlim(-0.1, 2.1)
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.legend(loc='upper left', fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{prefix.lower()}_utt_score_dist.png"), dpi=300)
        plt.close()

    if __name__ == "__main__":
        args = get_arguments()
        am = args.am
        
        tr_dataset_full = GoPDataset('train', am=am)
        te_dataset = GoPDataset('test', am=am)
        te_dataloader = DataLoader(te_dataset, batch_size=2500, shuffle=False)

        stratify_matrix, full_multilabel, full_utt, e_utt = get_multilabel_stratification_matrix(tr_dataset_full)

        print("\nGenerando gráficos de distribución globales...")
        save_distribution_plots(full_multilabel, full_utt, e_utt, args.exp_dir, prefix="Global_Train")

        # LOAD SPEAKER IDS TO KEEP SPEAKERS SEPARATE ACROSS FOLDS
        speaker_groups = []
        with open('utt_ids_train_orden_mi_train', 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    speaker_id = line.split('\t')[0][:5] # Extract the first five characters of the utterance ID.
                    speaker_groups.append(speaker_id)
        speaker_groups = np.array(speaker_groups)

        # Check that speaker IDs align with the dataset.
        if len(speaker_groups) != len(stratify_matrix):
            raise ValueError(f"Fallo de integridad: {len(speaker_groups)} locutores frente a {len(stratify_matrix)} muestras de audio.")

        gkf = GroupKFold(n_splits=5)
        base_exp_dir = args.exp_dir

        cv_mse, cv_phn_corr, cv_acc_utt_corr, cv_pro_utt_corr, cv_flu_utt_corr, cv_com_utt_corr, cv_tot_utt_corr, cv_acc_word_corr, cv_stress_word_corr, cv_tot_word_corr = [], [], [], [], [], [], [], [], [], [] 

        for fold, (train_idx, val_idx) in enumerate(gkf.split(np.zeros(len(stratify_matrix)), stratify_matrix, groups=speaker_groups)):
            print(f"\n{'='*30}\nINICIANDO MULTI-LABEL STRATIFIED: FOLD {fold+1}/5\n{'='*30}")
            
            tr_dataset = Subset(tr_dataset_full, train_idx)
            val_dataset = Subset(tr_dataset_full, val_idx)
            
            tr_dataloader_apa = DataLoader(tr_dataset, batch_size=args.batch_size, shuffle=True) 
            # =========================================================================
            
            val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
            
            audio_mdl = build_model(args)
            
            args.current_exp_dir = os.path.join(base_exp_dir, f"fold_{fold+1}")
            os.makedirs(args.current_exp_dir, exist_ok=True)
                
            print(f"Guardando gráficos de validación I/O para el Fold {fold+1}...")
            save_distribution_plots(
                full_multilabel[train_idx], full_utt[train_idx], e_utt, 
                args.current_exp_dir, prefix=f"Fold_{fold+1}_Train"
            )
            save_distribution_plots(
                full_multilabel[val_idx], full_utt[val_idx], e_utt, 
                args.current_exp_dir, prefix=f"Fold_{fold+1}_Val"
            )
                
            te_mse, te_phn_corr, te_utt_acc_corr, te_utt_com_corr, te_utt_flu_corr, te_utt_pro_corr, te_utt_tot_corr, te_word_acc_corr, te_word_str_corr, te_word_tot_corr = train(audio_mdl, tr_dataloader_apa, val_dataloader, te_dataloader, args)
            
            cv_mse.append(te_mse)
            cv_phn_corr.append(te_phn_corr)
            cv_acc_utt_corr.append(te_utt_acc_corr)
            cv_pro_utt_corr.append(te_utt_pro_corr)
            cv_flu_utt_corr.append(te_utt_flu_corr)
            cv_com_utt_corr.append(te_utt_com_corr)
            cv_tot_utt_corr.append(te_utt_tot_corr)
            cv_acc_word_corr.append(te_word_acc_corr)
            cv_stress_word_corr.append(te_word_str_corr)
            cv_tot_word_corr.append(te_word_tot_corr)
    

        args.exp_dir = base_exp_dir
        print()
        print("---------------------------\nRESULTADOS FINALES DE LA VALIDACIÓN MULTI-LABEL STRATIFIED\n---------------------")
        print()

        print("\nResults:")
        print("-" * 50)

        metrics_map = {
            "phone_test_mse": cv_mse,
            "phone_test_pcc": cv_phn_corr,
            "utt_test_pcc_accuracy": cv_acc_utt_corr,
            "utt_test_pcc_completeness": cv_com_utt_corr,
            "utt_test_pcc_fluency": cv_flu_utt_corr,
            "utt_test_pcc_prosodic": cv_pro_utt_corr,
            "utt_test_pcc_total": cv_tot_utt_corr,
            "word_test_pcc_accuracy": cv_acc_word_corr,
            "word_test_pcc_stress": cv_stress_word_corr,
            "word_test_pcc_total": cv_tot_word_corr,

        }

        for metric_name, values in metrics_map.items():
            mean_val = np.mean(values)
            std_val = np.std(values)
            print(f"{metric_name}: {mean_val:.3f}")
            print(f"std: {std_val:.3f}")
            print("-" * 50)

        print("\n--- EJECUCIÓN MULTI-LABEL CROSS-VALIDATION COMPLETADA ---")

        result = np.zeros([1, 16])
        result[0, :2] = [np.mean(cv_mse), np.mean(cv_phn_corr)]
        result[0, 2:10] = [np.mean(cv_acc_utt_corr), np.mean(cv_pro_utt_corr), np.mean(cv_flu_utt_corr), np.mean(cv_com_utt_corr), np.mean(cv_tot_utt_corr), np.mean(cv_acc_word_corr), np.mean(cv_stress_word_corr), np.mean(cv_tot_word_corr)]
        result[0, 10:13] = [np.mean(cv_acc_word_corr), np.mean(cv_stress_word_corr), np.mean(cv_tot_word_corr)]
    
        header = ','.join(gen_result_header())
        exp_dir = args.exp_dir
        np.savetxt(exp_dir + '/result.csv', result, delimiter=',', header=header, comments='')
