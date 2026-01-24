# Parameter Learning for Bayesian Networks via Decomposition

This repository provides all code and resources necessary to reproduce the experiments on parameter learning and probabilistic inference for Bayesian networks presented in our paper.  

## Repository Contents

- **Jupyter Notebook Files**: Each experiment is provided as a separate notebook for easy review and reproduction of results.  
- **Decom_Tree.py**: Implements the core algorithms for minimal d-decomposition tree construction, DC TEST, and parameter learning.  
- **bif_file**: Contains all Bayesian network files used in the experiments, sourced from BNlearn ([https://www.bnlearn.com/bnrepository/](https://www.bnlearn.com/bnrepository/)).  
- **README**: This document, detailing experimental setup, key hyperparameters, and hardware configuration.  

## Hardware Configuration

All experiments were conducted on the following system:

- **Processor**: Intel(R) Xeon(R) Silver 4215R CPU @ 3.20GHz (2 processors)  
- **Memory**: 128 GiB  

## Key Hyperparameters and Experimental Settings

- **DC TEST Decay Implementation**: All non-adjacent node pairs are exhaustively checked to ensure convexity constraints. Since the minimal separator nodes usually have very low dimensions, this step incurs minimal computational overhead.  
- **Parallel Parameter Learning Threads**: The number of threads is dynamically allocated according to task size, with a maximum of 9 cores.  
- **Parameter Learning Method**: Maximum likelihood estimation (MLE) is used. Unlike Bayesian estimation, MLE generally does not require prior distributions.  
- **Randomness Control**: All experiments use standardized random seeds to ensure reproducibility by third parties.  

## Reproducibility

By using the provided notebooks and following the above configurations, all experimental results can be reliably reproduced.  

## Contact

For any questions or assistance, please submit an issue in this repository or contact the corresponding author.
