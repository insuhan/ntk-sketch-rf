import os
import numpy as np
import argparse
import torch
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

from datasets import get_uci_dataset
from ntk_sketch import TensorSketch, OblvFeat, get_poly_approx_ntk
from ntk_random_features import NtkFeatureMapOps


def solve_reg(x_tr, y_tr, x_te, y_te, lam, y_std=1.0):
    n, d = x_tr.shape
    xtype = x_tr.type()

    if x_tr.shape[0] > x_tr.shape[1]:
        b = x_tr.T @ (y_tr[:, None] if y_tr.dim() == 1 else y_tr)
        y_pred = x_te @ torch.linalg.solve(x_tr.T @ x_tr + lam * torch.eye(d).type(xtype), b)
    else:
        b = y_tr[:, None] if y_tr.dim() == 1 else y_tr
        y_pred = x_te @ (x_tr.T @ torch.linalg.solve(x_tr @ x_tr.T + lam * torch.eye(n).type(xtype), b))

    mse = mean_squared_error((y_pred * y_std).cpu(), (y_te * y_std).cpu())
    acc = ((y_pred.argmax(axis=1) == y_te.argmax(axis=1)) * 1.0).mean().item()
    return mse, acc


def hyperparams_search(LAMBDA_LIST, X, y, train_fold, valid_fold, y_std=1.0):

    best_mse = np.inf
    best_lam = np.inf

    for lam in LAMBDA_LIST:
        mse, _ = solve_reg(X[train_fold], y[train_fold], X[valid_fold], y[valid_fold], lam, y_std)

        if mse <= best_mse:
            best_lam = lam
            best_mse = mse

    return {"best_lam": best_lam, "best_mse": best_mse}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_layers', default=1, type=int)
    parser.add_argument('--method', default='ntksketch', type=str)
    parser.add_argument('--dataset', default='ct', type=str)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--ns_deg', default=2, type=int)
    parser.add_argument('--cs_dim', default=2, type=int)
    parser.add_argument('--feat_dim', default=8192, type=int)
    args = parser.parse_args()

    for name_, val_ in args.__dict__.items():
        print(f"{name_:>10} : {val_}")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    X_orig, y_orig = get_uci_dataset(args.dataset)

    for i in range(len(np.where(np.std(X_orig, axis=0) == 0.0)[0])):
        X_orig = np.delete(X_orig, np.std(X_orig, axis=0).tolist().index(0.0), axis=1)

    n_tot, d = X_orig.shape

    mse_all = []
    for i, (tr_fold, te_fold) in enumerate(KFold(n_splits=4).split(np.arange(n_tot))):

        if args.dataset == 'ct':
            X_std = 1
            y_std = 1
        else:
            X_std = np.std(X_orig[tr_fold], axis=0)
            y_std = np.std(y_orig[tr_fold], axis=0)

        X = (X_orig - np.mean(X_orig[tr_fold], axis=0)) / X_std
        y = (y_orig - np.mean(y_orig[tr_fold], axis=0)) / y_std

        if args.method == 'ntksketch':
            X = torch.from_numpy(X).type(torch.DoubleTensor)
            y = torch.from_numpy(y).type(torch.DoubleTensor)[:, None]

            ts = TensorSketch(d=d, m=args.feat_dim, q=args.ns_deg, dev='cpu')
            coeff = get_poly_approx_ntk(args.num_layers, args.ns_deg)
            coeff = torch.from_numpy(coeff).type(torch.DoubleTensor)

            Z = OblvFeat(ts, X, coeff).T

        elif args.method == 'ntkfeat':
            X = torch.from_numpy(X).type(torch.FloatTensor)
            y = torch.from_numpy(y).type(torch.DoubleTensor)[:, None]

            cs_dim = args.cs_dim
            a1_dim = args.feat_dim - cs_dim

            ntkrf = NtkFeatureMapOps(args.num_layers, d, m1=a1_dim, m0=a1_dim, ms=cs_dim, do_leverage=True)
            _, Z = ntkrf(X)
            Z = Z.type(torch.DoubleTensor)

        indices = np.random.permutation(len(tr_fold))
        tr_train_fold = indices[:1000]
        tr_valid_fold = indices[1000:11000]

        LAMBDA_LIST = [(np.exp(1)**i) * len(tr_fold) for i in range(-18, 4)]

        res = hyperparams_search(LAMBDA_LIST, Z[tr_fold], y[tr_fold], tr_train_fold, tr_valid_fold, y_std)
        best_lam = res['best_lam']

        mse, _ = solve_reg(Z[tr_fold], y[tr_fold], Z[te_fold], y[te_fold], best_lam * len(tr_fold), y_std)
        mse_all.append(mse)

    mse = np.mean(mse_all)
    print(f"mse: {mse}")


if __name__ == "__main__":
    main()
