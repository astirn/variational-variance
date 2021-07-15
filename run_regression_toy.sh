#!/bin/sh

# set mode
MODE="resume"

# number of parallel jobs
N=5

# dataset iterators
declare -a Datasets=("toy" "toy-sparse")

# MLE algorithms
declare -a MaximumLikelihoodAlgorithms=("Detlefsen" "Detlefsen (fixed)" "Normal" "Student")

# Bayesian algorithms and priors
declare -a BayesianAlgorithms=("Gamma-Normal")
declare -a PriorTypes=("VAP" "Standard" "VAMP" "VAMP*" "xVAMP" "xVAMP*" "VBEM" "VBEM*")

# loop over datasets
for data in "${Datasets[@]}"; do

  # loop over MLE algorithms
  for alg in "${MaximumLikelihoodAlgorithms[@]}"; do

    # run jobs in parallel if specified
    if [ $N -gt 1 ]; then
      python regression_experiments.py --dataset "$data" --algorithm "$alg" --mode $MODE  --parallel 1 &

    # otherwise, run job in foreground
    else
      python regression_experiments.py --dataset "$data" --algorithm "$alg" --mode $MODE  --parallel 0
    fi

    # check/wait for maximum jobs
    if [[ $(jobs -r -p | wc -l) -ge $N ]]; then
      wait -n
    fi
  done

  # loop over Bayesian algorithms
  for alg in "${BayesianAlgorithms[@]}"; do
    for prior in "${PriorTypes[@]}"; do

      # run jobs in parallel if specified
      if [ $N -gt 1 ]; then
        python regression_experiments.py --dataset "$data" --algorithm $alg --prior_type $prior --mode $MODE \
          --k 20 --parallel 1 &

      # otherwise, run job in foreground
      else
        python regression_experiments.py --dataset "$data" --algorithm $alg --prior_type $prior --mode $MODE \
          --k 20 --parallel 0
      fi

      # check/wait for maximum jobs
      if [[ $(jobs -r -p | wc -l) -ge $N ]]; then
        wait -n
      fi
    done
  done
done

# print all done
echo "toy experiments done!"

# run analysis scripts
python regression_analysis.py --experiment "toy"
python regression_analysis.py --experiment "toy-sparse"
