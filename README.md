# Do autoencoders need a bottleneck for anomaly detection?

The code for reproducing the results in the paper "Do autoencoders need a bottleneck for anomaly detection?" 

Runs with baetorch (built on Pytorch), and Neural Tangent Kernel.

## Code

- Code for preprocessing benchmark datasets and industrial datasets are in `benchmark` and `case_study` folders, respectively.
- Code for analysing the results are in `analyse` folder.

Main codes are in the base folder:
- `01-Main-Run.py` executes the training and prediction with deterministic AE, VAE and BAEs, and hyperparameter grids are specified in `Params_{Dataset}.py` (e.g. `Params_ZEMA.py` for ZeMA dataset). 
- `02-Run-InfiniteBAE.py` executes similar evaluation run but with infinitely wide BAE using Neural Tangent Kernel.

## Citation

Yong, Bang Xiang, and Alexandra Brintrup. "Do autoencoders need a bottleneck for anomaly detection?." arXiv preprint arXiv:2202.12637 (2022).

```
@article{yong2022autoencoders,
  title={Do autoencoders need a bottleneck for anomaly detection?},
  author={Yong, Bang Xiang and Brintrup, Alexandra},
  journal={arXiv preprint arXiv:2202.12637},
  year={2022}
}
```

