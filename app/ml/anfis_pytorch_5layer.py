import math
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F


class ANFIS5Layer(nn.Module):
    """PyTorch implementation of a 5-layer ANFIS-like model (regression).

    Layers (conceptual):
      1) Fuzzification (Gaussian MFs) -> membership degrees
      2) Rule layer -> compute firing strengths (product of memberships)
      3) Normalization of firing strengths
      4) Consequent functions (first-order Sugeno: linear per rule)
      5) Weighted sum -> output

    This implementation keeps MF centers and log_sigmas learnable.
    Consequents are linear functions per rule: y_k(x) = w_k^T x + b_k
    Output y = sum_k (w_k * y_k(x)), where w_k are normalized firings.
    """

    def __init__(self, n_inputs: int, n_mfs: int = 3, poly_degree: int = 1, full_poly: bool = False):
        super().__init__()
        self.n_inputs = n_inputs
        self.n_mfs = n_mfs
        self.poly_degree = max(1, int(poly_degree))
        self.full_poly = bool(full_poly)

        # Learnable membership function parameters: centers and log_sigma
        # Shape: (n_inputs, n_mfs)
        centers = torch.linspace(0.2, 0.8, steps=n_mfs)
        centers = centers.unsqueeze(0).repeat(n_inputs, 1)
        self.centers = nn.Parameter(centers)
        # log_sigma to ensure positivity
        self.log_sigma = nn.Parameter(torch.zeros(n_inputs, n_mfs) + math.log(0.1))

        # Number of rules = n_mfs ** n_inputs (can be large - use with care)
        self.n_rules = n_mfs ** n_inputs

        # Consequent params: linear coeffs per rule and bias
        # support polynomial degree; optionally include full cross-terms for degree=2
        if self.full_poly and self.poly_degree == 2:
            # features: original features + pairwise products (i<=j)
            self.consequent_dim = n_inputs + (n_inputs * (n_inputs + 1)) // 2
        else:
            # no cross terms: per-feature powers
            self.consequent_dim = n_inputs * self.poly_degree
        self.coeffs = nn.Parameter(torch.randn(self.n_rules, self.consequent_dim) * 0.1)
        self.bias = nn.Parameter(torch.zeros(self.n_rules))

        # Precompute all index combinations for rules
        self.rule_indices = list(itertools.product(range(n_mfs), repeat=n_inputs))

    def gaussian_mf(self, x):
        # x: (batch, n_inputs)
        # centers: (n_inputs, n_mfs) -> broadcast
        # returns: (batch, n_inputs, n_mfs)
        x_exp = x.unsqueeze(2)  # (batch, n_inputs, 1)
        centers = self.centers.unsqueeze(0)  # (1, n_inputs, n_mfs)
        sigma = torch.exp(self.log_sigma).unsqueeze(0)  # (1, n_inputs, n_mfs)
        sq = (x_exp - centers) ** 2
        mf = torch.exp(-sq / (2 * sigma ** 2 + 1e-8))
        return mf

    def set_centers(self, centers_array):
        """Set centers from numpy array of shape (n_inputs, n_mfs)."""
        arr = torch.tensor(centers_array, dtype=self.centers.dtype)
        if arr.shape != self.centers.shape:
            raise ValueError('centers_array must match shape (n_inputs, n_mfs)')
        with torch.no_grad():
            self.centers.copy_(arr)

    def forward(self, x):
        """Forward pass.

        Args:
            x: FloatTensor (batch, n_inputs)

        Returns:
            y: (batch,) predicted values
            info: dict with membership, raw_firing, normalized_firing, consequents
        """
        batch = x.shape[0]
        device = x.device

        # 1) membership degrees
        mem = self.gaussian_mf(x)  # (batch, n_inputs, n_mfs)

        # 2) rule firing strengths: product of selected MF degrees across inputs
        # We'll compute by iterating over rule index combinations (ok for small dims)
        # result: (batch, n_rules)
        firings = []
        for comb in self.rule_indices:
            # for each input i take mem[:, i, comb[i]]
            parts = [mem[:, i, comb[i]] for i in range(self.n_inputs)]
            prod = parts[0]
            for p in parts[1:]:
                prod = prod * p
            firings.append(prod.unsqueeze(1))
        raw_firing = torch.cat(firings, dim=1)  # (batch, n_rules)

        # 3) normalize
        denom = raw_firing.sum(dim=1, keepdim=True) + 1e-8
        norm_firing = raw_firing / denom

        # 4) consequents y_k(x) = coeffs_k^T phi(x) + bias_k
        # phi(x) includes polynomial terms up to poly_degree per input (no cross terms)
        # build consequent features phi
        if self.full_poly and self.poly_degree == 2:
            # x: (batch, d)
            parts = [x]
            d = self.n_inputs
            for i in range(d):
                for j in range(i, d):
                    parts.append((x[:, i] * x[:, j]).unsqueeze(1))
            phi = torch.cat(parts, dim=1)
        else:
            if self.poly_degree == 1:
                phi = x
            else:
                parts = []
                for p in range(1, self.poly_degree + 1):
                    parts.append(x ** p)
                phi = torch.cat(parts, dim=1)

        # phi: (batch, consequent_dim)
        rule_out = F.linear(phi, self.coeffs, self.bias)  # (batch, n_rules)

        # 5) final output
        y = (norm_firing * rule_out).sum(dim=1)

        info = {
            "membership": mem.detach().cpu(),
            "raw_firing": raw_firing.detach().cpu(),
            "norm_firing": norm_firing.detach().cpu(),
            "rule_out": rule_out.detach().cpu(),
            "centers": self.centers.detach().cpu(),
            "sigmas": torch.exp(self.log_sigma).detach().cpu(),
        }

        return y, info


if __name__ == "__main__":
    # quick smoke test
    model = ANFIS5Layer(n_inputs=3, n_mfs=2)
    x = torch.rand(4, 3)
    y, info = model(x)
    print(y.shape)
