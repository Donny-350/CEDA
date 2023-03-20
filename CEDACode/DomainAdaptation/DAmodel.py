import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


# Autograd Function objects are what record operation history on tensors,
# and define formulas for the forward and backprop.
class GradientReversalFn(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        # Store context for backprop
        ctx.alpha = alpha

        # Forward pass is a no-op
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # Backward pass is just to -alpha the gradient
        output = grad_output.neg() * ctx.alpha

        # Must return same number as inputs to forward()
        return output, None


# Define model architecture
class CEDA(nn.Module):
    def __init__(self, sample_feats, predicate_feats, join_feats, hid_units):
        super(CEDA, self).__init__()

        self.sample_mlp1 = nn.Linear(sample_feats, hid_units)
        self.sample_mlp2 = nn.Linear(hid_units, hid_units)

        self.attention_predicate = nn.MultiheadAttention(embed_dim=predicate_feats, num_heads=1, batch_first=True)
        self.predicate_mlp1 = nn.Linear(predicate_feats, hid_units)

        self.join_mlp1 = nn.Linear(join_feats, hid_units)
        self.join_mlp2 = nn.Linear(hid_units, hid_units)

        self.out_mlp1 = nn.Linear(hid_units * 3, hid_units)

        self.regression = nn.Sequential(
            nn.Linear(hid_units, hid_units),
            nn.ReLU(),
            nn.Linear(hid_units, 1),
            nn.Sigmoid()
        )
        self.domain_classifer = nn.Sequential(
            nn.Linear(hid_units, 100),
            nn.ReLU(True),
            nn.Linear(100, 2),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, samples, predicates, joins, sample_mask, predicate_mask, join_mask, grl_lambda=1.0):

        hid_sample = F.relu(self.sample_mlp1(samples))
        hid_sample = F.relu(self.sample_mlp2(hid_sample))
        hid_sample = hid_sample * sample_mask
        hid_sample = torch.sum(hid_sample, dim=1, keepdim=False)
        sample_norm = sample_mask.sum(1, keepdim=False)
        hid_sample = hid_sample / sample_norm

        out_atten, _ = self.attention_predicate(predicates, predicates, predicates)
        hid_predicate = F.relu(self.predicate_mlp1(out_atten))
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
        reverse_hid = GradientReversalFn.apply(hid, grl_lambda)

        out = self.regression(hid)
        domain_pred = self.domain_classifer(reverse_hid)
        return hid, out, domain_pred
