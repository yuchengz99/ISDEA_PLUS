# Inductive Structural Double Equivariance Architecture PLUS (ISDEA+)

A significantly faster double equivariant GNN implementation that achieves up to 120x speedup compared to the original ISDEA implementation.

## Install dependencies

PyTorch and TorchDrug are the two main dependencies. Install all the dependencies by running
```bash
pip install -r requirements.txt
```

## Example commands to train the model

To train a model for the PediaTypes experiments, e.g. on the `EN-FR` dataset, run
```bash
python src/main.py --exp_name PediaTypes --dataset_folder data/PediaTypes --dataset EN2FR-15K-V2 --mode train --epoch 10 --valid_epoch 1 --num_cpus 32
```

This will create a checkpoint folder under `result/PediaTypes/EN2FR-15K-V2/<run_hash>`, where `<run_hash>` is the hash of the arguments to this run.


## Example commands to test the model

To test a model's *node prediction performance $(i, k, ?)$* for the PediaTypes experiments, e.g. on the `EN-FR` dataset, run
```bash
python src/main.py --exp_name PediaTypes-Test --dataset_folder data/PediaTypes --dataset EN2FR-15K-V2 --mode test --negative_sampling node --load_ckpt result/PediaTypes/EN2FR-15K-V2/<run_hash> --num_cpus 32
```
where `<run_hash>` is the run hash of the previous training run. 

To test a model's *relation prediction performance $(i, ?, j)$* on PediaTypes `EN-FR` dataset, run
```bash
python src/main.py --exp_name PediaTypes-Test --dataset_folder data/PediaTypes --dataset EN2FR-15K-V2 --mode test --negative_sampling relation --load_ckpt result/PediaTypes/EN2FR-15K-V2/<run_hash> --num_cpus 32
```


