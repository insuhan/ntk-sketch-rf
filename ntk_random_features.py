import math
import numpy as np
import pandas as pd
import torch
assert torch.__version__.split('+')[0] >= '1.8.0'
import torch.fft as fft
import torch.nn as nn
from scipy.special import erf


def gibbs_sampling_weighted_normal(input_dim, num_chains, gibbs_iterations=1, inv_resolution=1000):
    assert (num_chains > 0)
    W = np.random.randn(input_dim, num_chains)
    marginal_sum = np.sum(W**2, axis=0)

    x = np.linspace(-9, 9, inv_resolution + 1)
    a0 = erf(x / np.sqrt(2)) / 2 + 0.5
    a1 = x * np.exp(-x**2 / 2) / np.sqrt(2 * np.pi)

    for _ in range(gibbs_iterations):

        for i in range(input_dim):
            marginal_sum = marginal_sum - W[i, :]**2
            cdf = a0 - np.outer(1 / (marginal_sum + 1), a1)
            randv = np.random.rand(num_chains)

            idx1 = np.array([np.searchsorted(cdf[j], randv[j]) for j in range(num_chains)])
            idx0 = idx1 - 1
            frac1 = (randv - cdf[np.arange(num_chains), idx0]) / (cdf[np.arange(num_chains), idx1] -
                                                                  cdf[np.arange(num_chains), idx0])
            W[i, :] = x[idx0] * (1 - frac1) + x[idx1] * frac1
            marginal_sum += W[i, :]**2

    return torch.FloatTensor(W)


class AcosFeatureMap(nn.Module):

    def __init__(self, input_dim, output_dim, do_leverage, dev='cpu'):
        super(AcosFeatureMap, self).__init__()
        if input_dim == 0 or output_dim == 0:
            import pdb
            pdb.set_trace()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.do_leverage = do_leverage
        if not do_leverage:
            self.W = torch.randn(input_dim, output_dim)
            self.norm_const = math.sqrt(2.0 / self.output_dim)
        else:
            self.W = gibbs_sampling_weighted_normal(input_dim, output_dim)
            W_norm = self.W.pow(2).sum(0).sqrt()
            self.norm_const = math.sqrt(2.0 * input_dim / output_dim) / W_norm
            self.norm_const = self.norm_const.to(dev)
        self.W = self.W.to(dev)

    def forward(self, x, order):
        try:
            assert x.shape[1] == self.input_dim
        except:
            import pdb
            pdb.set_trace()
        if self.do_leverage:
            assert order == 1
        xw = x @ self.W
        if order == 0:
            return (xw > 0) * self.norm_const
        elif order == 1:
            return (abs(xw) + xw) * (self.norm_const / 2.0)
        else:
            raise NotImplementedError


class TensorProduct(nn.Module):

    def __init__(self, input_dim1, input_dim2, output_dim):
        super(TensorProduct, self).__init__()
        pass

    def forward(self, x, y):
        assert x.shape[0] == y.shape[0]
        n = x.shape[0]
        return torch.einsum('ij, ik->ijk', x, y).reshape(n, -1)


class TensorSRHT(nn.Module):

    def __init__(self, input_dim1, input_dim2, output_dim_double, dev='cpu'):
        super(TensorSRHT, self).__init__()
        output_dim = output_dim_double // 2  # ouput contains real + imag values
        self.input_dim1 = input_dim1
        self.input_dim2 = input_dim2
        self.output_dim = output_dim
        self.dev = dev
        self.sign1 = (torch.randint(2, (input_dim1,)) * 2 - 1).to(dev)
        self.indx1 = torch.randint(input_dim1, (output_dim,)).to(dev)
        self.sign2 = (torch.randint(2, (input_dim2,)) * 2 - 1).to(dev)
        self.indx2 = torch.randint(input_dim2, (output_dim,)).to(dev)

    def forward(self, x, y):
        assert x.shape[0] == y.shape[0]
        assert (x.shape[1] == self.input_dim1)
        assert (y.shape[1] == self.input_dim2)
        xhat = torch.fft.fftn(x * self.sign1, dim=1)[:, self.indx1]
        yhat = torch.fft.fftn(y * self.sign2, dim=1)[:, self.indx2]

        out_ = math.sqrt(1 / self.output_dim) * (xhat * yhat)
        return torch.cat((out_.real, out_.imag), 1)


class CountSketch2(nn.Module):

    def __init__(self, input_dim1, input_dim2, output_dim):
        super(CountSketch2, self).__init__()
        if input_dim1 == 0 or input_dim2 == 0 or output_dim == 0:
            import pdb
            pdb.set_trace()
        self.input_dim1 = input_dim1
        self.input_dim2 = input_dim2
        self.output_dim = output_dim
        self.sign1 = torch.randint(2, (input_dim1,)) * 2 - 1
        self.indx1 = torch.randint(output_dim, (input_dim1,))
        self.sign2 = torch.randint(2, (input_dim2,)) * 2 - 1
        self.indx2 = torch.randint(output_dim, (input_dim2,))

    # From https://github.com/gdlg/pytorch_compact_bilinear_pooling/blob/master/compact_bilinear_pooling/__init__.py
    def count_sketch_forward(self, x, indx, sign):
        x_size = tuple(x.size())
        s_view = (1,) * (len(x_size) - 1) + (x_size[-1],)
        out_size = x_size[:-1] + (self.output_dim,)
        sign = sign.view(s_view)
        xs = x * sign
        indx = indx.view(s_view).expand(x_size)
        out = x.new(*out_size).zero_()
        return out.scatter_add_(-1, indx, xs)

    def forward(self, x, y):
        assert (x.shape[0] == y.shape[0])
        assert (x.shape[1] == self.input_dim1)
        assert (y.shape[1] == self.input_dim2)
        n = x.shape[0]
        x_cs = self.count_sketch_forward(x, self.indx1, self.sign1)
        y_cs = self.count_sketch_forward(y, self.indx2, self.sign2)
        return fft.ifft(fft.fft(x_cs, dim=-1) * fft.fft(y_cs, dim=-1)).real


class NtkFeatureMapOps(nn.Module):

    def __init__(self, num_layers, input_dim, m1, m0, ms, sketch='srht', do_leverage=False, dev='cpu'):
        super(NtkFeatureMapOps, self).__init__()
        if m0 < 0:
            m0 = m1
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.m1 = m1
        self.m0 = m0
        self.ms = ms
        self.dev = dev

        self.sketch = sketch
        assert sketch in ['exact', 'srht', 'countsketch']
        if sketch == 'srht':
            sketch_func = TensorSRHT
        elif sketch == 'countsketch':
            sketch_func = CountSketch2
        elif sketch == 'exact':
            sketch_funct = TensorProduct
        else:
            raise NotImplementedError

        if ms < 0:
            Warning("For negative ms, we restrict the exact tensor product.")
            sketch_funct = TensorProduct
        elif ms == 0:
            Warning("When ms == 0, it approximates random features of NNGP.")

        self.arccos0 = [AcosFeatureMap(input_dim, m0, False, dev)]
        self.arccos1 = [AcosFeatureMap(input_dim, m1, do_leverage, dev)]
        self.sketches = [sketch_func(input_dim, m0, ms, dev)]
        for _ in range(num_layers - 1):
            self.arccos0.append(AcosFeatureMap(m1, m0, False, dev))
            self.arccos1.append(AcosFeatureMap(m1, m1, do_leverage, dev))
            self.sketches.append(sketch_func((ms + m1), m0, ms, dev))

    def forward(self, z_nngp_orig, z_ntk_orig=None):
        z_nngp = z_nngp_orig
        if z_ntk_orig is not None:
            z_ntk = torch.cat((z_nngp_orig, z_ntk_orig), axis=1)
        else:
            z_ntk = z_nngp

        for i in range(self.num_layers):
            if self.ms == 0:
                z_nngp = self.arccos1[i](z_nngp, order=1)
                z_ntk = z_nngp
            else:
                tmp = self.arccos0[i](z_nngp, order=0)
                z_nngp = self.arccos1[i](z_nngp, order=1)
                mu = self.sketches[i](z_ntk, tmp)
                z_ntk = torch.cat((z_nngp, mu), axis=1)

        return z_nngp, z_ntk
