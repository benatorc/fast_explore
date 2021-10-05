import argparse
import os
import json
from tqdm import tqdm
import pickle as pl
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
import selfies as sf

from mol_explore.dataset import SelfiesFragmentDataset

from mol_explore.models.smiles_vae import SelfiesVGNN
from mol_explore.models.smiles_vqvae import SelfiesVQVGNN
from mol_explore.models.vocab import common_atom_vocab
from mol_explore.utils.model_utils import save_model, load_model
from mol_explore.utils.chem_utils import selfies_frag_to_smiles

import pdb

def read_data(data_path):
    with open(data_path, 'r') as data_file:
        data = json.load(data_file)
        return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/chembl')
    parser.add_argument('--vocab_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument('--pretrain_model', type=str, default=None)

    parser.add_argument('--model_type', type=str,
                        choices=['vae', 'vqvae'], required=True, default='vae')
    parser.add_argument('--autoregressive', action='store_true', default=False)

    parser.add_argument('--atom_vocab', default=common_atom_vocab)

    parser.add_argument('--rnn_type', type=str, default='LSTM')
    parser.add_argument('--hidden_size', type=int, default=200)
    parser.add_argument('--n_embed', type=int, default=20)
    parser.add_argument('--latent_size', type=int, default=50)
    parser.add_argument('--depth', type=int, default=3)
    parser.add_argument('--diter', type=int, default=3)

    parser.add_argument('--mask_p', type=float, default=0.)
    parser.add_argument('--constant_var', action='store_true', default=False)
    parser.add_argument('--z_norm', type=float, default=0.)
    parser.add_argument('--beta', type=float, default=0.1)

    parser.add_argument('--vq_coef', type=float, default=1.)
    parser.add_argument('--commit_coef', type=float, default=1.)

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--grad_norm', type=float, default=10.)

    parser.add_argument('--n_epochs', type=int, default=10)
    parser.add_argument('--patience', type=int, default=20.,
                        help='Stop the pretraining if loss does not improve')
    parser.add_argument('--save_steps', type=int, default=1000)
    args = parser.parse_args()
    # Create the output_dir if not exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    model_dir = args.output_dir + '/models'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    logs_dir = args.output_dir + '/logs'
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    tf_writer = SummaryWriter(logs_dir)

    fragment_data_path = '%s/fragments.json' % args.data_dir
    data_list = read_data(fragment_data_path)
    data_list = [x[2] for x in data_list]

    if args.model_type == 'vae':
        model_class = SelfiesVGNN
    elif args.model_type == 'vqvae':
        model_class = SelfiesVQVGNN
    else:
        print('%s not supported' % args.model_type)
        exit()

    with open(args.vocab_path, 'rb') as vocab_file:
        selfies_stoi = pl.load(vocab_file)
        args.selfies_stoi = selfies_stoi

    model = model_class(args).cuda()
    gen_dataset = lambda: SelfiesFragmentDataset(
        data=data_list, avocab=args.atom_vocab, selfies_stoi=selfies_stoi,
        batch_size=args.batch_size)

    for param in model.parameters():
        if param.dim() == 1:
            nn.init.constant_(param, 0)
        else:
            nn.init.xavier_normal_(param)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Keep track of the lowest loss, and the model with the lowest loss
    best_loss, best_model_idx = float('inf'), 0

    cur_model_idx, step = 0, 0
    stats = defaultdict(list)
    for epoch in range(args.n_epochs):
        dataset = gen_dataset()

        for _, batch in enumerate(tqdm(dataset)):
            step += 1
            if batch is None:
                # Handle the case where the batch is invalid for some reason
                continue

            model.zero_grad()
            graph_tensors, smiles_list, selfies_list, selfies_enc, selfies_mask = batch
            batch = (graph_tensors, selfies_enc, selfies_mask, args.beta)

            loss, batch_stats, selfies_preds = model(*batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
            optimizer.step()

            smiles_acc, selfies_acc = compute_acc(selfies_preds, selfies_list, smiles_list)
            for stat_name, val in batch_stats.items():
                stats[stat_name].append(val)
            stats['pred_smiles_acc'].append(smiles_acc)
            stats['pred_selfies_acc'].append(selfies_acc)

            if step % args.save_steps == 0:
                stats_str = ''
                loss_avg = None
                for name, val_list in stats.items():
                    val_arr = np.array(val_list)
                    stat_avg = np.mean(val_arr)
                    stats_str += '%s: %.3f ' % (name, stat_avg)

                    tf_writer.add_scalar(tag='pretrain/' + name, scalar_value=stat_avg, global_step=step)
                    if name == 'total_loss':
                        loss_avg = stat_avg
                tf_writer.flush()
                print('Epoch: %d, step: %d ' % (epoch, step) + stats_str)
                stats = defaultdict(list)

                if loss_avg < best_loss:
                    best_loss = loss_avg
                    best_model_idx = cur_model_idx
                    model_save_path = '%s/model_%d' % (model_dir, cur_model_idx)
                    save_model(
                        save_path=model_save_path,
                        model=model,
                        model_class=model_class,
                        args=args, verbose=True)
                    cur_model_idx += 1

    best_model_path = '%s/model_%d' % (model_dir, best_model_idx)
    print('Training finished, loading model from %s' % best_model_path)
    model = load_model(best_model_path)
    save_model(
        save_path = '%s/model_best' % model_dir,
        model=model,
        model_class=model_class,
        args=args, verbose=True)

    tf_writer.close()

def compute_acc(selfies_preds, selfies_gold, smiles_gold):
    n = len(selfies_preds)
    smiles_preds = [selfies_frag_to_smiles(x) for x in selfies_preds]

    n_correct = 0
    for idx, pred in enumerate(selfies_preds):
        if pred == selfies_gold[idx]:
            n_correct += 1
    selfies_acc = n_correct / float(n)

    n_correct = 0
    for idx, pred in enumerate(smiles_preds):
        if pred == smiles_gold[idx]:
            n_correct += 1
    smiles_acc = n_correct / float(n)

    return smiles_acc, selfies_acc

if __name__ == '__main__':
    main()
