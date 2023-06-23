# BalsubLast - Subset-Sum Problem Solver

This repository contains an improved balanced algorithm, named BalsubLast, for solving the Subset-Sum Problem, a special case of the traditional knapsack problem.

The algorithm was originally described in the article "An improved balanced algorithm for the subset-sum problem" published in the European Journal of Operational Research and the code was published on a public repository that no longer exists. This repository is a backup version I had (not the final version). Please, let me know if it is not working.

## Overview

The Subset-Sum Problem is a well-known computational problem in computer science and mathematics. Given a set of numbers and a target sum, the problem aims to find a subset of the numbers that adds up to the target sum. The BalsubLast algorithm presented in this repository has been designed to efficiently solve residual instances of the Subset-Sum Problem that require an exhaustive search of the solution space. For non-residual instances, simple strategies of brute-force approach can easily solve it.

## Usage

The source code provided in this repository allows you to implement and utilize the BalsubLast algorithm for solving the Subset-Sum Problem. You can incorporate the algorithm into your own projects or adapt it to your specific use case.

The repo has the following:

- *benchmarks*: directory with several instance classes.
- *code/BalsubLast*: directory with the code BalsubLast algorithm;
- *code/CS32*: CUDA code for CS32, another algorithm that we compare in the paper.

In *benchmark* directory, the file 'Optimal Results.xlsx' has the value of the optimal solution for each instance.
The codes were a previous backup, so they are not so organized (sorry).

## How to cite

The BalsubLast algorithm is presented in the following paper:

V.V. Curtis, C.A.A. Sanches. (2019). "*An improved balanced algorithm for the subset-sum problem*". European Journal of Operational Research, 275(2), 460-466. DOI: [10.1016/j.ejor.2018.11.055](https://doi.org/10.1016/j.ejor.2018.11.055).

The parallel algorithm CS32 for GPU is presented in the paper:

V.V. Curtis, C.A.A. Sanches. (2017). "*A low-space algorithm for the subset-sum problem on GPU*". Computers & Operations Research, 83, 120-124. DOI: [10.1016/j.cor.2017.02.006](https://doi.org/10.1016/j.cor.2017.02.006).

