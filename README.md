# Scaling Neural Tangent Kernels via Sketching and Random Features

Python implementation of [Scaling Neural Tangent Kernels via Sketching and Random Features](https://arxiv.org/pdf/2106.07880.pdf)

Structures:
- See `ntk_sketch.py` for NTK Sketch algorithm
- See `cntk_sketch.py` for CNTK Sketch algorithm
- See `ntk_random_features.py` for NTK Random Features algorithm
- See `run_uci_regression.py` for ridge regression problems

Install required Python packages
```console
$  pip install requirements.txt
```
	
To run uci regression (Table 2), execute
```console
$  python run_uci_regression.py --dataset ct --method ntkfeat --num_layers 1 --feat_dim 8192 --cs_dim 7500
$  python run_uci_regression.py --dataset ct --method ntksketch --ns_deg 2 --num_layers 2 --feat_dim 8192
$  python run_uci_regression.py --dataset workloads --method ntkfeat --num_layers 2 --feat_dim 8192 --cs_dim 5000
$  python run_uci_regression.py --dataset workloads --method ntksketch --ns_deg 2 --num_layers 2 --feat_dim 8192
```

To run CNTK Sketch with CIFAR-10 dataset (Table 1), execute
```console
$  python run_cifar10.py
```
