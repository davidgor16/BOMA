# -*- coding: utf-8 -*-
# Archived training entry point. Loss, split, and checkpoint rules are retained.
# See docs/REPRODUCIBILITY.md and use tools/run_experiment.py for a fresh output path.
# @Time    : 9/20/21 12:02 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : traintest.py

import sys
import os
import numpy as np
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler # Weighted sampler import; usage depends on the saved experiment.
import torch.nn.functional as F
import matplotlib.pyplot as plt

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

from sklearn.model_selection import GroupKFold

import argparse

sys.path.append(os.path.dirname(os.path.dirname(sys.path[0])))
from models import *

print("I am process %s, running on %s: starting (%s)" % (os.getpid(), os.uname()[1], time.asctime()))
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# GOPT PARAMETERS
parser.add_argument("--goptdepth", type=int, default=2, help="number of layers of the GOPT")
parser.add_argument("--goptheads", type=int, default=2, help="number of heads of the GOPT")
parser.add_argument("--goptmlp", type=int, default=64, help="hidden dimension of the MLP in the GOPT")

# GENERAL PARAMETERS
parser.add_argument("--exp-dir", type=str, default="./exp/", help="directory to dump experiments")
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, metavar='LR', help='initial learning rate')
parser.add_argument("--n-epochs", type=int, default=3, help="number of maximum training epochs")
parser.add_argument("--batch_size", type=int, default=25, help="training batch size")
parser.add_argument("--embed_dim", type=int, default=12, help="M3C transformer embedding dimension")
parser.add_argument("--loss_w_phn", type=float, default=1, help="weight for phoneme-level loss")
parser.add_argument("--loss_w_word", type=float, default=1, help="weight for word-level loss")
parser.add_argument("--loss_w_utt", type=float, default=1, help="weight for utterance-level loss")
parser.add_argument("--model", type=str, default='gopt', help="name of the model")
parser.add_argument("--am", type=str, default='librispeech', help="name of the acoustic models")
parser.add_argument("--noise", type=float, default=0., help="the scale of random noise added on the input GoP feature")
parser.add_argument("--alpha_mdd", type=float, default=0.3, help="weight for MDD loss")

# V/C FEATURE EXTRACTION PARAMETERS
parser.add_argument("--num_convs_vc", type=int, default=32)
parser.add_argument("--input_dim_vowels", type=int, default=6)
parser.add_argument("--input_dim_consonants", type=int, default=6)
parser.add_argument("--output_dim_vc", type=int, default=6)
parser.add_argument("--dropout_cnn_vc", type=float, default=0.)
parser.add_argument("--dropout_mlp_vc", type=float, default=0.)

# SSL FEATURE EXTRACTION PARAMETERS
parser.add_argument("--num_convs_ssl", type=int, default=32)
parser.add_argument("--input_dim_ssl", type=int, default=6)
parser.add_argument("--output_dim_ssl", type=int, default=6)
parser.add_argument("--dropout_cnn_ssl", type=float, default=0.)
parser.add_argument("--dropout_mlp_ssl", type=float, default=0.)

# FEATURE FUSION PARAMETERS
parser.add_argument("--fusion_dim", type=int, default=10)
parser.add_argument("--dropout_mlp_fusion", type=float, default=0.)

# PHONEME-LEVEL PARAMETERS
parser.add_argument("--num_convs_phn", type=int, default=32)
parser.add_argument("--output_dim_phn", type=int, default=6)
parser.add_argument("--dropout_cnn_phn", type=float, default=0.)
parser.add_argument("--dropout_mlp_phn", type=float, default=0.)

# WORD-LEVEL PARAMETERS
parser.add_argument("--num_convs_word", type=int, default=32)
parser.add_argument("--output_dim_word", type=int, default=6)
parser.add_argument("--dropout_cnn_word", type=float, default=0.)
parser.add_argument("--dropout_mlp_word", type=float, default=0.)

# UTTERANCE-LEVEL PARAMETERS
parser.add_argument("--num_convs_utt", type=int, default=32)
parser.add_argument("--output_dim_utt", type=int, default=6)
parser.add_argument("--dropout_cnn_utt", type=float, default=0.)
parser.add_argument("--dropout_mlp_utt", type=float, default=0.)

def gen_result_header():
    phn_header = ['epoch', 'phone_train_mse', 'phone_train_pcc', 'phone_test_mse', 'phone_test_pcc', 'learning rate']
    utt_header_set = ['utt_train_mse', 'utt_train_pcc', 'utt_test_mse', 'utt_test_pcc']
    utt_header_score = ['accuracy', 'completeness', 'fluency', 'prosodic', 'total']
    word_header_set = ['word_train_pcc', 'word_test_pcc']
    word_header_score = ['accuracy', 'stress', 'total']
    mdd_header_score = ['f1_mdd', 'precision_mdd', 'recall_mdd']
    
    utt_header, word_header = [], []
    for dset in utt_header_set:
        utt_header = utt_header + [dset+'_'+x for x in utt_header_score]
    for dset in word_header_set:
        word_header = word_header + [dset+'_'+x for x in word_header_score]
    header = phn_header + utt_header + word_header  + mdd_header_score
    return header

def train(audio_model, train_loader_apa, val_loader, test_loader, args, p_vals, word_vals, utt_vals):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('running on ' + str(device))
    print(f"The value of alpha_mdd is: {args.alpha_mdd}")

    best_epoch, best_mse = 0, 999
    best_lr = args.lr
    global_step, epoch = 0, 0
    exp_dir = getattr(args, 'current_exp_dir', args.exp_dir)

    if not isinstance(audio_model, nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)

    audio_model = audio_model.to(device)
    
    trainables = [p for p in audio_model.parameters() if p.requires_grad]
    print('Total parameter number is : {:.3f} k'.format(sum(p.numel() for p in audio_model.parameters()) / 1e3))
    print('Total trainable parameter number is : {:.3f} k'.format(sum(p.numel() for p in trainables) / 1e3))
    
    optimizer = torch.optim.Adam(trainables, args.lr, weight_decay=5e-7, betas=(0.95, 0.999))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, list(range(20, 100, 5)), gamma=0.5, last_epoch=-1)
    
    loss_fn_mdd = nn.CrossEntropyLoss()

    result = np.zeros([1, 35])

    while epoch < args.n_epochs:
        train_loader = train_loader_apa
        print(f'Epoch: {epoch} | Number of batches: {len(train_loader)}')
            
        audio_model.train()
        
        for i, (audio_input, phn_label, dur_feat, ener_feat, w2v_feat, hubert_feat, wavlm_feat, cano_phns, real_phns, utt_label, word_label) in enumerate(train_loader):
                    
            wavlm_feat = wavlm_feat.to(device, non_blocking=True)
            hubert_feat = hubert_feat.to(device, non_blocking=True)
            w2v_feat = w2v_feat.to(device, non_blocking=True)
            ener_feat = ener_feat.to(device, non_blocking=True)
            dur_feat = dur_feat.to(device, non_blocking=True)
            audio_input = audio_input.to(device, non_blocking=True)
            phn_label = phn_label.to(device, non_blocking=True)
            utt_label = utt_label.to(device, non_blocking=True)
            word_label = word_label.to(device, non_blocking=True)
                        
            warm_up_step = 100
            if global_step <= warm_up_step and global_step % 5 == 0:
                warm_lr = (global_step / warm_up_step) * args.lr
                for param_group in optimizer.param_groups:
                    param_group['lr'] = warm_lr
    
            words = word_label[:,:,3]

            if "noSSL" in args.model:
                u1, u2, u3, u4, u5, p, w1, w2, w3, x_phn, mdd = audio_model(audio_input, cano_phns, words, dur_feat, ener_feat)
            else:
                u1, u2, u3, u4, u5, p, w1, w2, w3, x_phn, mdd = audio_model(audio_input, cano_phns, words, dur_feat, ener_feat, w2v_feat, hubert_feat, wavlm_feat)

            
            phn_mask = (phn_label>=0)
            p = p.squeeze(2)
            p = p * phn_mask
            phn_label = phn_label * phn_mask
            
            loss_phn = torch.pow(phn_label - p, 2) # MSE loss for phn
            p_num = torch.abs((p>0) * (p<=2.0) * p) # apply only for valid predictions  
            p_num = torch.div(p_num, 0.2, rounding_mode = 'floor') # find score class for the phn predictions 
            p_num = p_vals[p_num.long()] # pick up the value that corresponds to the prediction's class
            
            b = 0.9       
                  
            loss_phn = torch.where(p_num != 0, (1-b)/(1-torch.pow(b, p_num)), 1.) * phn_mask * loss_phn # SB factor * MSE loss
            loss_phn = torch.sum(loss_phn) / torch.sum(phn_mask)

            # utterance level loss, also mse
            utt_preds = torch.cat((u1, u2, u3, u4, u5), dim=1)
            loss_utt = torch.pow(utt_label - utt_preds, 2) # MSE loss for utt
            utt_num = torch.abs((utt_preds > 0) * (utt_preds <= 2.0) * utt_preds) # apply only for valid predictions
            utt_num = torch.div(utt_num, 0.2, rounding_mode = 'floor') # find score class for the utt predictions 
            
            for i in range(5):
                utt_num[:, i] = utt_vals[:,i][utt_num[:,i].long()] # pick up the value that corresponds to the prediction's class
            
            loss_utt = torch.where(utt_num != 0, (1-b)/(1-torch.pow(b, utt_num)), 1.) * loss_utt # SB factor * MSE loss
            loss_utt = torch.mean(loss_utt)
            
            # word level loss
            word_label = word_label[:, :, 0:3]
            word_mask = (word_label>=0)
            word_pred = torch.cat((w1,w2,w3), dim=2)
            word_pred = word_pred * word_mask
            word_label = word_label * word_mask
            
            loss_word = torch.pow(word_label - word_pred, 2) # MSE loss for utt
            word_num = torch.abs((word_pred>0) * (word_pred<=2.0) * word_pred) # apply only for valid predictions
            word_num = torch.div(word_num, 0.2, rounding_mode = 'floor') # find score class for the word predictions 
            
            for i in range(3):
                word_num[:,:,i] = word_vals[:,i][word_num[:,:,i].long()] # pick up the value that corresponds to the prediction's class
            
            loss_word = torch.where(word_num != 0, (1-b)/(1-torch.pow(b, word_num)), 1.) * word_mask * loss_word # SB factor * MSE loss
            loss_word = torch.sum(loss_word) / torch.sum(word_mask)
            
            loss_apa = args.loss_w_phn * loss_phn + args.loss_w_utt * loss_utt + args.loss_w_word * loss_word
            
            mask = real_phns != -1
            real_phns = real_phns[mask].to(device)
            mdd = mdd[mask]
                     
            loss_mdd = loss_fn_mdd(mdd, real_phns.long())
            loss = loss_apa + args.alpha_mdd * loss_mdd 

            optimizer.zero_grad()
            loss.backward()  
            optimizer.step()
            global_step += 1
                
        tr_mse, tr_corr, tr_utt_mse, tr_utt_corr, tr_word_mse, tr_word_corr, tr_precision, tr_recall, tr_f1_score = validate(audio_model, train_loader, args, -1)
        val_mse, val_corr, val_utt_mse, val_utt_corr, val_word_mse, val_word_corr, val_precision, val_recall, val_f1_score = validate(audio_model, val_loader, args, best_mse)

        print('Phone: Train MSE: {:.3f}, CORR: {:.3f}'.format(tr_mse, tr_corr))
        print('Phone: Val MSE: {:.3f}, CORR: {:.3f}'.format(val_mse, val_corr))
        print('Utterance:, ACC: {:.3f}, COM: {:.3f}, FLU: {:.3f}, PRO: {:.3f}, Total: {:.3f}'.format(val_utt_corr[0], val_utt_corr[1], val_utt_corr[2], val_utt_corr[3], val_utt_corr[4]))
        print('Word:, ACC: {:.3f}, Stress: {:.3f}, Total: {:.3f}'.format(val_word_corr[0], val_word_corr[1], val_word_corr[2]))
        print('MDD:, F1: {:.3f}, Precision: {:.3f}, Recall: {:.3f}'.format(val_f1_score, val_precision, val_recall))
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
            result[0,32:35] = [val_f1_score, val_precision, val_recall]

            header = ','.join(gen_result_header())
            np.savetxt(exp_dir + '/result.csv', result, delimiter=',', header=header, comments='')

        if best_epoch == epoch:
            if os.path.exists("%s/models/" % (exp_dir)) == False:
                os.mkdir("%s/models" % (exp_dir))
            torch.save(audio_model.state_dict(), "%s/models/best_audio_model.pth" % (exp_dir))

        if global_step > warm_up_step:
            scheduler.step()
        
        epoch += 1
        
    print('-------------------training finished-------------------')
    
    if os.path.exists("%s/models/best_audio_model.pth" % (exp_dir)):
        audio_model.load_state_dict(torch.load("%s/models/best_audio_model.pth" % (exp_dir)))
        
    te_mse, te_corr, te_utt_mse, te_utt_corr, te_word_mse, te_word_corr, te_precision, te_recall, te_f1_score = validate(audio_model, test_loader, args, 9999.0)

    print('FINAL TEST Phone: Test MSE: {:.3f}, CORR: {:.3f}'.format(te_mse, te_corr))
    print('FINAL TEST Utterance:, ACC: {:.3f}, COM: {:.3f}, FLU: {:.3f}, PRO: {:.3f}, Total: {:.3f}'.format(te_utt_corr[0], te_utt_corr[1], te_utt_corr[2], te_utt_corr[3], te_utt_corr[4]))
    print('FINAL TEST Word:, ACC: {:.3f}, Stress: {:.3f}, Total: {:.3f}'.format(te_word_corr[0], te_word_corr[1], te_word_corr[2]))
    print('FINAL TEST MDD:, F1: {:.3f}, Precision: {:.3f}, Recall: {:.3f}'.format(te_f1_score, te_precision, te_recall))
    
    tr_mse, tr_corr, tr_utt_mse, tr_utt_corr, tr_word_mse, tr_word_corr, tr_precision, tr_recall, tr_f1_score = validate(audio_model, train_loader_apa, args, -1)

    result[0, :6] = [best_epoch, tr_mse, tr_corr, te_mse, te_corr, best_lr]
    result[0, 6:26] = np.concatenate([tr_utt_mse, tr_utt_corr, te_utt_mse, te_utt_corr])
    result[0, 26:32] = np.concatenate([tr_word_corr, te_word_corr])
    result[0,32:35] = [te_f1_score, te_precision, te_recall]

    header = ','.join(gen_result_header())
    np.savetxt(exp_dir + '/result.csv', result, delimiter=',', header=header, comments='')

    return te_mse, te_corr, te_utt_corr[0], te_utt_corr[1], te_utt_corr[2], te_utt_corr[3], te_utt_corr[4], te_word_corr[0], te_word_corr[1], te_word_corr[2], te_f1_score, te_precision, te_recall


def validate(audio_model, val_loader, args, best_mse):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not isinstance(audio_model, nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)
    audio_model = audio_model.to(device)
    audio_model.eval()

    exp_dir = getattr(args, 'current_exp_dir', args.exp_dir)

    A_phn, A_phn_target = [], []
    A_u1, A_u2, A_u3, A_u4, A_u5, A_utt_target = [], [], [], [], [], []
    A_w1, A_w2, A_w3, A_word_target = [], [], [], []
    
    with torch.no_grad():
        for i, (audio_input, phn_label, dur_feat, ener_feat, w2v_feat, hubert_feat, wavlm_feat, cano_phns, real_phns, utt_label, word_label) in enumerate(val_loader):
            wavlm_feat = wavlm_feat.to(device)
            hubert_feat = hubert_feat.to(device)
            w2v_feat = w2v_feat.to(device)
            audio_input = audio_input.to(device)
            dur_feat = dur_feat.to(device)
            ener_feat = ener_feat.to(device)
            
            words = word_label[:,:,3]
            if "noSSL" in args.model:
                u1, u2, u3, u4, u5, p, w1, w2, w3, x_phn, mdd = audio_model(audio_input, cano_phns, words, dur_feat, ener_feat)
            else:
                u1, u2, u3, u4, u5, p, w1, w2, w3, x_phn, mdd = audio_model(audio_input, cano_phns, words, dur_feat, ener_feat, w2v_feat, hubert_feat, wavlm_feat)      
                          
            if best_mse != -1:
                mask = cano_phns != -1
                cano_phns = cano_phns[mask]
                real_phns = real_phns[mask]
                x_phn = x_phn[mask]
                mdd = mdd[mask]
                
                from sklearn.metrics import precision_recall_fscore_support
                mdd = mdd.cpu().numpy()
                mdd = np.argmax(mdd, axis=1)
                real_phns = real_phns.cpu().numpy()
                precision, recall, f1_score, _ = precision_recall_fscore_support(real_phns, mdd, average='macro', zero_division=0)
            else:
                precision = recall = f1_score = 0
            
            p = p.to('cpu').detach()
            u1, u2, u3, u4, u5 = u1.to('cpu').detach(), u2.to('cpu').detach(), u3.to('cpu').detach(), u4.to('cpu').detach(), u5.to('cpu').detach()
            w1, w2, w3 = w1.to('cpu').detach(), w2.to('cpu').detach(), w3.to('cpu').detach()

            A_phn.append(p)
            A_phn_target.append(phn_label)
            A_u1.append(u1); A_u2.append(u2); A_u3.append(u3); A_u4.append(u4); A_u5.append(u5)
            A_utt_target.append(utt_label)
            A_w1.append(w1); A_w2.append(w2); A_w3.append(w3)
            A_word_target.append(word_label)

        A_phn, A_phn_target  = torch.cat(A_phn), torch.cat(A_phn_target)
        A_u1, A_u2, A_u3, A_u4, A_u5, A_utt_target = torch.cat(A_u1), torch.cat(A_u2), torch.cat(A_u3), torch.cat(A_u4), torch.cat(A_u5), torch.cat(A_utt_target)
        A_w1, A_w2, A_w3, A_word_target = torch.cat(A_w1), torch.cat(A_w2), torch.cat(A_w3), torch.cat(A_word_target)

        phn_mse, phn_corr = valid_phn(A_phn, A_phn_target)
        A_utt = torch.cat((A_u1, A_u2, A_u3, A_u4, A_u5), dim=1)
        utt_mse, utt_corr = valid_utt(A_utt, A_utt_target)
        A_word = torch.cat((A_w1, A_w2, A_w3), dim=2)
        word_mse, word_corr, valid_word_pred, valid_word_target = valid_word(A_word, A_word_target)

        if phn_mse < best_mse:
            print(f"[INFO] New best model found during validation with phn_mse {phn_mse:.3f} (previous best: {best_mse:.3f}). Saving predictions...")
            if os.path.exists(exp_dir + '/preds') == False:
                os.mkdir(exp_dir + '/preds')
  
            np.save(exp_dir + '/preds/phn_target.npy', A_phn_target)
            np.save(exp_dir + '/preds/word_target.npy', valid_word_target)
            np.save(exp_dir + '/preds/utt_target.npy', A_utt_target)
            if 'cano_phns' in locals() and best_mse != -1:
                np.save(exp_dir + '/preds/cano_phn.npy', cano_phns.cpu().numpy())  

            np.save(exp_dir + '/preds/phn_pred.npy', A_phn)
            np.save(exp_dir + '/preds/word_pred.npy', valid_word_pred)
            np.save(exp_dir + '/preds/utt_pred.npy', A_utt)
            if best_mse != -1:
                np.save(exp_dir + '/preds/mdd_pred.npy', mdd)

    return phn_mse, phn_corr, utt_mse, utt_corr, word_mse, word_corr, precision, recall, f1_score

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
        self.am = am
        if am=="features_vc_clean_embbedings_norm":
            dir='GoP_VC_Features_Embbeding_norm'
        else:
            raise ValueError('Acoustic Model Unrecognized.')

        if set == 'train':
            self.feat = torch.tensor(np.load('../data/'+dir+'/tr_feat.npy'), dtype=torch.float)
            self.phn_label = torch.tensor(np.load('../data/'+dir+'/tr_label_phn.npy'), dtype=torch.float)
            self.utt_label = torch.tensor(np.load('../data/'+dir+'/tr_label_utt.npy'), dtype=torch.float)
            self.word_label = torch.tensor(np.load('../data/'+dir+'/tr_label_word.npy'), dtype=torch.float)
            self.real_phn_label = torch.tensor(np.load('../data/'+dir+'/tr_real_phn.npy'), dtype=torch.float)
            self.wavlm_feat = torch.tensor(np.load('../data/'+dir+'/tr_wavlm_features.npy'), dtype=torch.float)
            self.hubert_feat = torch.tensor(np.load('../data/'+dir+'/tr_hub_features.npy'), dtype=torch.float)
            self.w2v_feat = torch.tensor(np.load('../data/'+dir+'/tr_w2v_features.npy'), dtype=torch.float)
            self.ener_feat = torch.tensor(np.load('../data/'+dir+'/tr_energy_features.npy'), dtype=torch.float)
            self.dur_feat = torch.tensor(np.load('../data/'+dir+'/tr_duration_features.npy'), dtype=torch.float)

        elif set == 'test':
            self.feat = torch.tensor(np.load('../data/'+dir+'/te_feat.npy'), dtype=torch.float)
            self.phn_label = torch.tensor(np.load('../data/'+dir+'/te_label_phn.npy'), dtype=torch.float)
            self.utt_label = torch.tensor(np.load('../data/'+dir+'/te_label_utt.npy'), dtype=torch.float)
            self.word_label = torch.tensor(np.load('../data/'+dir+'/te_label_word.npy'), dtype=torch.float)
            self.real_phn_label = torch.tensor(np.load('../data/'+dir+'/te_real_phn.npy'), dtype=torch.float)
            self.wavlm_feat = torch.tensor(np.load('../data/'+dir+'/te_wavlm_features.npy'), dtype=torch.float)
            self.hubert_feat = torch.tensor(np.load('../data/'+dir+'/te_hub_features.npy'), dtype=torch.float)
            self.w2v_feat = torch.tensor(np.load('../data/'+dir+'/te_w2v_features.npy'), dtype=torch.float)
            self.ener_feat = torch.tensor(np.load('../data/'+dir+'/te_energy_features.npy'), dtype=torch.float)
            self.dur_feat = torch.tensor(np.load('../data/'+dir+'/te_duration_features.npy'), dtype=torch.float)
        self.utt_label = self.utt_label / 5
        self.word_label[:, :, 0:3] = self.word_label[:, :, 0:3] / 5
        self.phn_label[:, :, 1] = self.phn_label[:, :, 1]
        
        self.utt_vals = torch.zeros(11, 5).cuda()
        for i in range(5):
            utt_uniq, utt_cnt = torch.unique(self.utt_label[:,i], return_counts=True)
            utt_uniq = utt_uniq/0.2
            self.utt_vals[utt_uniq.long(), i] = utt_cnt.cuda().float()
                            
        self.p_vals = torch.zeros(11).cuda()
        p_uniq, p_cnt = torch.unique(self.phn_label[:, :, 1], return_counts=True)
        p_uniq, p_cnt = p_uniq[1:]/0.2, p_cnt[1:]
        self.p_vals[p_uniq.long()] = p_cnt.cuda().float()
                        
        self.word_vals = torch.zeros(11, 3).cuda()
        for i in range(3):
            word_uniq, word_cnt = torch.unique(self.word_label[:, :, i], return_counts=True)
            word_uniq, word_cnt = word_uniq[1:]/0.2, word_cnt[1:]
            self.word_vals[word_uniq.long(), i] = word_cnt.cuda().float()
        
    def __len__(self):
        return self.feat.shape[0]

    def __getitem__(self, idx):
        return self.feat[idx, :], self.phn_label[idx, :, 1], self.dur_feat[idx,:], self.ener_feat[idx,:], self.w2v_feat[idx,:], self.hubert_feat[idx,:], self.wavlm_feat[idx,:], self.phn_label[idx, :, 0],  self.real_phn_label[idx, :], self.utt_label[idx, :], self.word_label[idx, :]

def build_model(args):
    if args.model == 'm3c':
        mdl = M3C(num_convs_vc = args.num_convs_vc, input_dim_vowels = args.input_dim_vowels, input_dim_consonants = args.input_dim_consonants, output_dim_vc = args.output_dim_vc, dropout_cnn_vc= args.dropout_cnn_vc, dropout_mlp_vc = args.dropout_mlp_vc,
                        num_convs_ssl = args.num_convs_ssl, input_dim_ssl = args.input_dim_ssl, output_dim_ssl = args.output_dim_ssl, dropout_cnn_ssl= args.dropout_cnn_ssl, dropout_mlp_ssl = args.dropout_mlp_ssl,
                        fusion_dim=args.fusion_dim, dropout_mlp_fusion=args.dropout_mlp_fusion,
                        num_convs_phn=args.num_convs_phn, output_dim_phn=args.output_dim_phn, dropout_cnn_phn=args.dropout_cnn_phn, dropout_mlp_phn=args.dropout_cnn_phn, 
                        num_convs_word=args.num_convs_word, output_dim_word=args.output_dim_word, dropout_cnn_word=args.dropout_cnn_word, dropout_mlp_word=args.dropout_cnn_word, 
                        num_convs_utt=args.num_convs_utt, output_dim_utt=args.output_dim_utt, dropout_cnn_utt=args.dropout_cnn_utt, dropout_mlp_utt=args.dropout_cnn_utt) 
        return mdl
    elif args.model == "m3c_noSSL":
        mdl = M3C_noSSL(num_convs_vc = args.num_convs_vc, input_dim_vowels = args.input_dim_vowels, input_dim_consonants = args.input_dim_consonants, output_dim_vc = args.output_dim_vc, dropout_cnn_vc= args.dropout_cnn_vc, dropout_mlp_vc = args.dropout_mlp_vc,
                        fusion_dim=args.fusion_dim, dropout_mlp_fusion=args.dropout_mlp_fusion,
                        num_convs_phn=args.num_convs_phn, output_dim_phn=args.output_dim_phn, dropout_cnn_phn=args.dropout_cnn_phn, dropout_mlp_phn=args.dropout_cnn_phn, 
                        num_convs_word=args.num_convs_word, output_dim_word=args.output_dim_word, dropout_cnn_word=args.dropout_cnn_word, dropout_mlp_word=args.dropout_cnn_word, 
                        num_convs_utt=args.num_convs_utt, output_dim_utt=args.output_dim_utt, dropout_cnn_utt=args.dropout_cnn_utt, dropout_mlp_utt=args.dropout_cnn_utt)
        return mdl
    raise ValueError(f"Model {args.model} no reconocido.")

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
    args = parser.parse_args()
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

    cv_mse, cv_phn_corr, cv_acc_utt_corr, cv_pro_utt_corr, cv_flu_utt_corr, cv_com_utt_corr, cv_tot_utt_corr, cv_acc_word_corr, cv_stress_word_corr, cv_tot_word_corr, cv_f1_mdd, cv_precision_mdd, cv_recall_mdd = [], [], [], [], [], [], [], [], [], [], [], [], [] 

    for fold, (train_idx, val_idx) in enumerate(gkf.split(np.zeros(len(stratify_matrix)), stratify_matrix, groups=speaker_groups)):
        print(f"\n{'='*30}\nINICIANDO MULTI-LABEL STRATIFIED: FOLD {fold+1}/5\n{'='*30}")
        
        tr_dataset = Subset(tr_dataset_full, train_idx)
        val_dataset = Subset(tr_dataset_full, val_idx)
        
     
        # =========================================================================
        # BALANCE BATCH SAMPLING WITH WeightedRandomSampler
        # =========================================================================
        # Group training utterances into tertiles using e_utt thresholds.
        train_utt_scores = full_utt[train_idx]
        train_classes = np.digitize(train_utt_scores, e_utt)
        
        # Count observations in each class.
        class_counts = np.bincount(train_classes, minlength=len(e_utt)+1)
        class_counts = np.where(class_counts == 0, 1, class_counts) # Avoid division by zero.
        
        # Sampling weights are inversely proportional to class frequency.
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[train_classes]
        
        # Create the sampler.
        batch_sampler = WeightedRandomSampler(
            weights=torch.DoubleTensor(sample_weights), 
            num_samples=len(sample_weights), 
            replacement=True # Sample with replacement to oversample the minority classes.
        )
      
        # The sampler controls random selection, so shuffle=True is omitted.
        tr_dataloader_apa = DataLoader(tr_dataset, batch_size=args.batch_size, sampler=batch_sampler)
        #tr_dataloader_apa = DataLoader(tr_dataset, batch_size=args.batch_size, shuffle=True) 
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
            
        te_mse, te_phn_corr, te_utt_acc_corr, te_utt_com_corr, te_utt_flu_corr, te_utt_pro_corr, te_utt_tot_corr, te_word_acc_corr, te_word_str_corr, te_word_tot_corr, te_f1, te_precision, te_recall = train(audio_mdl, tr_dataloader_apa, val_dataloader, te_dataloader, args, tr_dataset_full.p_vals, tr_dataset_full.word_vals, tr_dataset_full.utt_vals)
        
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
        cv_f1_mdd.append(te_f1)
        cv_precision_mdd.append(te_precision)
        cv_recall_mdd.append(te_recall) 

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
        "f1_score": cv_f1_mdd,
        "precision": cv_precision_mdd,
        "recall": cv_recall_mdd
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
    result[0,13:16] = [np.mean(cv_f1_mdd), np.mean(cv_precision_mdd), np.mean(cv_recall_mdd)]

    header = ','.join(gen_result_header())
    np.savetxt(os.path.join(base_exp_dir, "result.csv"),result,delimiter=',',header=header,comments='')