# Approximate Bayesian Computation

## Introduction
This project is focused on the implementation of two algorithms for likelihood-free approximation of the posterior distribution: the ABC-Rejection and the ABC-MCMC (Approximate Bayesian Computation - Markov Chain Monte Carlo) methods. These algorithms are particularly useful in scenarios where the likelihood is difficult to compute or unknown.

## Project Overview
- **Academic Example**: We have included an academic example with a known true posterior to analyze the performance of both the ABC-Rejection and ABC-MCMC methods.
- **Stochastic Differential Equation Model**: The project also explores a stochastic differential equation that models the level of Theophylline in the blood over time.

## Repository Structure
- `distribution`: This file contains all the distributions used within the examples. The `rvs` method is used for sampling from a distribution, and the `pdf` method for evaluating the density at certain points.
- `approx_bayesian_computation.py`: Defines the ABC-Rejection and ABC-MCMC algorithms.
- `discrepancy_metric.py`: Implements the discrepancy metric, which plays a crucial role in these algorithms.

## Getting Started

### Setting Up
1. **Create a Virtual Environment (Recommended)**
   ```bash
   python -m venv venv
   ```
2. **Activate the Virtual Environment**
    On Windows
    ```bash
   venv\Scripts\activate
   ```
   On Unix or MacOS:
   ```bash
   source venv/bin/activate
   ```
3. **Install Required Packages**
   ```bash
   python -m venv venv
   ```

### Running the Experiments
- **Academic Example with ABC-Rejection**
     ```bash
   python academic_example.py --method=abc_rejection --M=100 --N=500 --plot_distance=True
   ```
- **Academic Example with ABC-MCMC**
     ```bash
   python academic_example.py --method=abc_mcmc --nu=nu --M=100 --N=500
   ```
   Note: Replace `nu` with the desired standard deviation of the proposal distribution.
- **Pharmacokinetic Model**
    ```bash
   python pharmacokinetic_model.py
   ```