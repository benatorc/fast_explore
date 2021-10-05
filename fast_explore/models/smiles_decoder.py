import torch
import torch.nn as nn
import rdkit.Chem as Chem
import torch.nn.functional as F
import selfies as sf

import pdb

class SelfiesDecoder(nn.Module):
    def __init__(self, vocab_stoi, hidden_size, latent_size, depth, autoregressive=False):
        super(SelfiesDecoder, self).__init__()
        self.vocab_stoi = vocab_stoi
        self.vocab_idx = {v: k for k, v in vocab_stoi.items()}
        self.n_classes = len(vocab_stoi)
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.depth = depth
        self.autoregressive = autoregressive

        self.rnn = nn.GRU(
            input_size=latent_size,
            hidden_size=hidden_size,
            num_layers=depth,
            batch_first=False)

        self.W_vocab = nn.Linear(self.n_classes, latent_size, bias=False)

        self.W_out = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.n_classes))

        self.criterion = nn.CrossEntropyLoss(reduction='none')

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        return weight.new_zeros(self.depth, batch_size, self.hidden_size)

    def forward(self, z_graph_vecs, targets, targets_mask, decode_selfies=False):
        batch_size, max_seq_len = targets.size()

        hidden = self.init_hidden(batch_size)  # [depth, batch_size, hidden_size]
        z_graph_vecs = z_graph_vecs.unsqueeze(0)  # [1, batch_size, hidden_size]

        targets_input_1h = F.one_hot(
            targets, num_classes=self.n_classes).float().transpose(0, 1) # [seq_len, batch_size, vocab_size]
        targets_input_1h = torch.cat([
            torch.zeros([1, batch_size, self.n_classes]).cuda(),
            targets_input_1h[:-1, :, :]])

        if self.autoregressive:
            z_input = z_graph_vecs.repeat(max_seq_len, 1, 1) + self.W_vocab(targets_input_1h)
        else:
            z_input = z_graph_vecs.repeat(max_seq_len, 1, 1)

        # During training, use gold labels
        rnn_output, hidden = self.rnn(z_input, hidden)
        all_output_preds = self.W_out(rnn_output).transpose(0, 1)

        all_losses = self.criterion(input=all_output_preds.transpose(1, 2), target=targets)
        all_losses_masked = all_losses * targets_mask

        avg_loss = torch.mean(torch.sum(all_losses_masked, dim=1) / batch_size)
        avg_loss = torch.sum(all_losses_masked) / torch.sum(targets_mask)

        selfies_list = None
        if decode_selfies:
            output_indices = torch.argmax(all_output_preds, dim=2).detach().cpu().numpy()
            selfies_list = []
            for bidx in range(batch_size):
                cur_indices = output_indices[bidx]
                selfies_str = self.decode_into_selfie(cur_indices)
                if selfies_str is not None:
                    selfies_list.append(selfies_str)
                else:
                    selfies_list.append('')
        return avg_loss, selfies_list

    def decode_into_selfie(self, output_indices):
        output_str = ''
        pad_token = self.vocab_stoi['[nop]']
        for idx in output_indices:
            if idx == pad_token:
                break
            else:
                output_str += self.vocab_idx[idx]
        return output_str

    def decode(self, z_graph_vecs, max_decode=20):
        batch_size = z_graph_vecs.size()[0]

        hidden = self.init_hidden(batch_size)
        z_graph_vecs = z_graph_vecs.unsqueeze(0)

        cur_pred = torch.zeros([1, batch_size, self.n_classes]).cuda() # Initialize empty vector for start
        all_output_preds = []
        for _ in range(max_decode):
            if self.autoregressive:
                z_input = z_graph_vecs + self.W_vocab(cur_pred)
            else:
                z_input = z_graph_vecs
            rnn_output, hidden = self.rnn(z_input, hidden)
            output_preds = self.W_out(rnn_output)

            pred_probs = torch.softmax(output_preds, dim=2)
            max_idx = torch.argmax(pred_probs, dim=2)
            cur_pred = F.one_hot(max_idx, num_classes=self.n_classes).float()

            all_output_preds.append(output_preds)
        all_output_preds = torch.cat(all_output_preds, dim=0).transpose(0, 1)

        output_indices = torch.argmax(all_output_preds, dim=2).detach().cpu().numpy()
        selfies_list = []
        for bidx in range(batch_size):
            cur_indices = output_indices[bidx]
            selfies_str = self.decode_into_selfie(cur_indices)
            if selfies_str is not None:
                selfies_list.append(selfies_str)
            else:
                selfies_list.append('')
        return selfies_list
