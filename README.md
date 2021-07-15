# Variational Variance

This repository is the official implementation of
Probabilistically Fortified Optimization of Parameter Maps for Heteroscedastic Gaussian Likelihoods 

We include [code](https://github.com/SkafteNicki/john) files from
[Reliable training and estimation of variance networks](https://arxiv.org/abs/1906.03260)
in the `john-master` sub-directory of our repository. Their methods constitute one of our baselines.
Any file therein denoted with `_orig` is their original code. We had to modify certain files for integration.
We include these originals for easy differencing. Any file without an `_orig` pair is unmodified.

## Requirements

To install requirements:
```setup
pip install -r requirements.txt
```

Our code is based in TensorFlow 2.3, which automatically detects and utilizes any available CUDA acceleration.
However, the baseline code from Detlefsen et al., 2019 (`john-master` sub-directory) is based in pyTorch.
Their code supports enabling and disabling CUDA.
Any of our code that calls theirs does NOT access their CUDA option and consequently defaults with CUDA enabled.
Please be aware of this if running on a machine without access to CUDA acceleration.

To download UCI data sets (performed in `run_regression_uci.sh` as well):
```setup
python regression_data.py
```

Data for our VAE experiments downloads automatically via [TensorFlow Datasets](https://www.tensorflow.org/datasets).

## Reproducing Results
If you downloaded a copy of this repository with a populated `results` sub-directory, then you may already
have all of our original results. Please see the following subsections for details on how to run just the analysis.

### Toy Data
Please see `run_regression_toy.sh`, which runs our toy regression experiments in sequence.
Upon completing the experiments,  this script also runs the analysis code, which automatically
generates the relevant figures in our manuscript.
To run just the analyses:
```
python regression_analysis.py --experiment toy
python regression_analysis.py --experiment toy-sparse
```

### UCI Data
Please see `run_regression_uci.sh`, which runs our UCI regression experiments in parallel.
Upon completing the experiments, this script also runs the analysis code, which automatically
generates the relevant tables in our manuscript.
To run just the analyses:
```
python regression_analysis.py --experiment uci
```

#### Active Learning
Please see `run_active_learning_uci.sh`, which runs our UCI active learning experiments in parallel.
Upon completing the experiments, this script also runs the analysis code, which automatically
generates the relevant tables and figures in our manuscript.
To run just the analyses:
```
python active_learning_analysis.py
```

### VAE Experiments
Please see `run_generative.sh`, which runs our VAE experiments.
Upon completing the experiments, this script also runs the analysis  code, which automatically
generates the relevant tables and figures in our manuscript.
To run just the analyses:
```
python generative_analysis.py
```

## Contributing

We have selected an MIT license for this repository.
