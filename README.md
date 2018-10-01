# Where Did My Optimum Go? Experiments.

This is the repository for the experiments seen in the EWRL 2018 Paper "Where Did My Optimum Go?". Largely they are a modification of the code found in:

https://github.com/ikostrikov/pytorch-a2c-ppo-acktr

Modifications are mostly superficial to allow reading of configuration files for easier experiment running and distribution.

## Example 

You can run one of the ablation experiments for example via:

```bash
./bin/indexed_experiment --config_file ./benchmarks/configurations/continuous/a2c/continuous_suite.json --run_index 0 --ablation_config ./benchmarks/configurations/papers/hendersonromoff2018optimizer/ablation/sgd/momentum.json
```

## Possible Issues

You may need to add the current directory to your pythonpath if you have problems running the experiment:

```bash
export PYTHONPATH=${PYTHONPATH}:.
```

## Citation

If you find this useful, please cite our work:


```
@inproceedings{hendersonromoff2018optimizer,
  author    = {Peter Henderson and Joshua Romoff and Joelle Pineau},
  title     = {Where Did My Optimum Go?: An Empirical Analysis of Gradient Descent Optimization in Policy Gradient Methods},
  booktitle = {The 14th European Workshop on Reinforcement Learning (EWRL 2018)},
  year      = {2018}
}
```

Additionally, if you are relying on the codebase heavily please note the original codebase as well:

```
@misc{pytorchrl,
  author = {Kostrikov, Ilya},
  title = {PyTorch Implementations of Reinforcement Learning Algorithms},
  year = {2018},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/ikostrikov/pytorch-a2c-ppo-acktr}},
}
```
