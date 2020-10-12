# Baselines

This directory contains the code for the baselines listed as `Deep+MST`, `Deep+ILP` and `Deep+Greedy` in our paper.

## MST Solver

The MST solver is implemented in C++ (`mst.cpp`). Please build it first:

```bash
g++ mst.cpp -o mst --std=c++11
```

## ILP Solver

Following [irit-stac](https://github.com/irit-melodi/irit-stac), we also use [SCIP](https://www.scipopt.org/) as the ILP solver. Please download it and update the path argument `scip_path` in `main.py`.

## Usage

```bash
python main.py --method=mst|ilp|greedy --train
```
