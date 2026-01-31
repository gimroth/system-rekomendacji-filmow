"""Simple gradient-based ANFIS implemented in PyTorch.

- n_inputs: number of input features
- n_mfs: number of membership functions per input (e.g., 3)
- consequents: zero-order (scalar per rule) or first-order (linear: w*x + b per rule)

This implementation focuses on clarity and small-scale training for quick testing.
"""

import itertools
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PyTorchANFIS(nn.Module):
    def __init__(self, n_inputs: int = 3, n_mfs: int = 3, consequent_order: int = 0):
        super().__init__()
        self.n_inputs = n_inputs
        self.n_mfs = n_mfs
        self.consequent_order = consequent_order  # 0 => scalar per rule; 1 => linear per rule

        # MF parameters: for Gaussian MF we learn centers and sigmas per input per mf
        # centers: (n_inputs, n_mfs)
        self.centers = nn.Parameter(torch.linspace(0.0, 1.0, steps=n_mfs).unsqueeze(0).repeat(n_inputs, 1))
        # sigma init to reasonable width
        self.log_sigmas = nn.Parameter(torch.log(torch.ones(n_inputs, n_mfs) * 0.1))

        # rules: every combination of mf indices across inputs -> n_rules = n_mfs ** n_inputs
        self.rule_combinations = list(itertools.product(range(n_mfs), repeat=n_inputs))
        self.n_rules = len(self.rule_combinations)

        if consequent_order == 0:
            # scalar consequent per rule
            self.consequents = nn.Parameter(torch.zeros(self.n_rules))
        else:
            # linear consequents: weight per input + bias per rule
            self.consequents_w = nn.Parameter(torch.zeros(self.n_rules, n_inputs))
            self.consequents_b = nn.Parameter(torch.zeros(self.n_rules))

    def _gauss_mf(self, x):
        # x: (..., n_inputs)
        # returns membership (..., n_inputs, n_mfs)
        # centers: (n_inputs, n_mfs)
        x_exp = x.unsqueeze(-1)  # (..., n_inputs, 1)
        centers = self.centers.unsqueeze(0)  # (1, n_inputs, n_mfs)
        sigmas = torch.exp(self.log_sigmas).unsqueeze(0)  # (1, n_inputs, n_mfs)
        return torch.exp(-0.5 * ((x_exp - centers) / (sigmas + 1e-8)) ** 2)

    def forward(self, x):
        # x: (batch, n_inputs)
        # membership: (batch, n_inputs, n_mfs)
        mf = self._gauss_mf(x)
        batch = x.shape[0]
        # compute rule firing strengths by multiplying selected mfs for each rule
        # Prepare tensor of shape (n_rules, n_inputs) with indices
        idx = torch.tensor(self.rule_combinations, dtype=torch.long, device=x.device)  # (n_rules, n_inputs)
        # Build per-rule membership values by gathering per-input indices
        # mf: (batch, n_inputs, n_mfs)
        # We'll gather for each input separately and stack
        idx_t = idx.t()  # (n_inputs, n_rules)
        mf_per_input = []
        for i in range(self.n_inputs):
            # mf[:, i, :] -> (batch, n_mfs); need indices (batch, n_rules)
            inds = idx_t[i].unsqueeze(0).expand(batch, -1)  # (batch, n_rules)
            vals = mf[:, i, :].gather(1, inds)  # (batch, n_rules)
            mf_per_input.append(vals)
        # stack -> (n_inputs, batch, n_rules) -> permute to (batch, n_rules, n_inputs)
        mf_per_rule = torch.stack(mf_per_input, dim=0).permute(1, 2, 0)
        # product over inputs -> (batch, n_rules)
        firing = torch.prod(mf_per_rule, dim=-1)
        # normalize
        denom = firing.sum(dim=-1, keepdim=True)
        denom = denom + 1e-8
        weights = firing / denom

        if self.consequent_order == 0:
            # consequents: (n_rules,)
            out_rules = self.consequents.unsqueeze(0).expand(batch, -1)  # (batch, n_rules)
        else:
            # linear: out = w*x + b per rule
            # x: (batch, n_inputs); consequents_w: (n_rules, n_inputs)
            out_rules = (x.unsqueeze(1) * self.consequents_w.unsqueeze(0)).sum(-1) + self.consequents_b.unsqueeze(0)

        out = (weights * out_rules).sum(-1)
        return out, {'firing': firing, 'weights': weights}

    def predict_numpy(self, X: np.ndarray):
        self.eval()
        with torch.no_grad():
            t = torch.from_numpy(X.astype(np.float32))
            y, _ = self.forward(t)
            return y.cpu().numpy()

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str, map_location='cpu'):
        sd = torch.load(path, map_location=map_location)
        self.load_state_dict(sd)
