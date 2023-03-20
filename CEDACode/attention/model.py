import torch
import torch.nn as nn
import torch.nn.functional as F


# Define model architecture

class CEDA(nn.Module):
    def __init__(self, sample_feats, predicate_hist_feats, predicate_feats, join_feats, hid_units):
        super(CEDA, self).__init__()
        self.sample_mlp1 = nn.Linear(sample_feats, hid_units)
        self.sample_mlp2 = nn.Linear(hid_units, hid_units)

        self.predicate_hist_mlp1 = nn.Linear(predicate_hist_feats, 4)
        self.predicate_hist_mlp2 = nn.Linear(4, 4)

        self.attention = nn.MultiheadAttention(embed_dim=predicate_feats+4, num_heads=1, batch_first=True)
        self.predicate_mlp1 = nn.Linear(predicate_feats+4, hid_units)
        #self.predicate_mlp2 = nn.Linear(hid_units, hid_units)


        self.join_mlp1 = nn.Linear(join_feats, hid_units)
        self.join_mlp2 = nn.Linear(hid_units, hid_units)
        self.out_mlp1 = nn.Linear(hid_units * 3, hid_units)
        self.out_mlp2 = nn.Linear(hid_units, 1)

    def forward(self, samples, predicates_hist, predicates, joins, sample_mask, predicate_hist_mask, predicate_mask, join_mask):
        # samples has shape [batch_size x num_joins+1 x sample_feats]
        # predicates has shape [batch_size x num_predicates x predicate_feats]
        # joins has shape [batch_size x num_joins x join_feats]

        hid_sample = F.relu(self.sample_mlp1(samples))
        hid_sample = F.relu(self.sample_mlp2(hid_sample))
        hid_sample = hid_sample * sample_mask  # Mask
        hid_sample = torch.sum(hid_sample, dim=1, keepdim=False)
        sample_norm = sample_mask.sum(1, keepdim=False)
        hid_sample = hid_sample / sample_norm  # Calculate average only over non-masked parts

        hid_predicate_hist = F.relu(self.predicate_hist_mlp1(predicates_hist))
        hid_predicate_hist = F.relu(self.predicate_hist_mlp2(hid_predicate_hist))

        predicates = torch.cat((predicates, hid_predicate_hist), 2)

        out_atten, _ = self.attention(predicates, predicates, predicates)
        hid_predicate = F.relu(self.predicate_mlp1(out_atten))
        #hid_predicate = F.relu(self.predicate_mlp2(hid_predicate))
        hid_predicate = hid_predicate * predicate_mask
        hid_predicate = torch.sum(hid_predicate, dim=1, keepdim=False)
        predicate_norm = predicate_mask.sum(1, keepdim=False)
        hid_predicate = hid_predicate / predicate_norm

        hid_join = F.relu(self.join_mlp1(joins))
        hid_join = F.relu(self.join_mlp2(hid_join))
        hid_join = hid_join * join_mask
        hid_join = torch.sum(hid_join, dim=1, keepdim=False)
        join_norm = join_mask.sum(1, keepdim=False)
        hid_join = hid_join / join_norm


        hid = torch.cat((hid_sample, hid_predicate, hid_join), 1)
        hid = F.relu(self.out_mlp1(hid))
        out = torch.sigmoid(self.out_mlp2(hid))
        return out
