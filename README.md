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

To train a model for the Meta Learning experiments with the WikiTopics dataset, e.g. on the `Run-1-InfSciSpoTax` scenario where KGs of four domains - Infrastructure, Science, Sport, Taxonomy - are merged, run
```bash
python src/main.py --exp_name MetaLearn --dataset_folder data/WikiTopics-MetaLearn/ --dataset Run-1-InfSciSpoTax --mode train --epoch 10 --valid_epoch 1 --num_cpus 32
```

Similarly, this will create a checkpoint folder under `result/MetaLearn/Run-1-InfSciSpoTax/<run_hash>`. where `<run_hash>` is the hash of the arguments to this run. Note that we don't include test splits inside `data/WikiTopics-MetaLearn/`. This is because the test experiments are done on the individual WikiTopics KGs, which is included in the `data/WikiTopics/` folder.

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

To test a model's *node prediction performance $(i, k, ?)$* for the Meta Learning experiments, e.g. on the Art domain KG of the WikiTopics dataset, run
```bash
python src/main.py --exp_name MetaLearn-Test --dataset_folder data/WikiTopics/ --dataset wikidata_artv2 --mode test --negative_sampling node --load_ckpt result/MetaLearn/Run-1-InfSciSpoTax/<run_hash> --num_cpus 32
```

To test a model's *relation prediction performance $(i, ?, j)$* for the Meta Learning experiments, e.g. on the Art domain KG of the WikiTopics dataset, run
```bash
python src/main.py --exp_name MetaLearn-Test --dataset_folder data/WikiTopics/ --dataset wikidata_artv2 --mode test --negative_sampling relation --load_ckpt result/MetaLearn/Run-1-InfSciSpoTax/<run_hash> --num_cpus 32
```


