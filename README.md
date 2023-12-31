# RL-for-TSP


## Overview

This project compares solutions to the Traveling Salesman Problem (TSP) using both Q-Learning and a Policy Gradient approach with a neural network. The Q-Learning algorithm is based on tabular Q-values, while the Policy Gradient approach utilizes a neural network to learn a policy.

## Table of Contents

- [Getting Started](#getting-started)
   - [Prerequisites](#prerequisites)
   - [Installation](#installation)
- [Usage](#usage)
   - [play-around](#play-around)
      - [Q-Learning Parameters](#Q-Learning-Parameters)
      - [Policy Gradient Parameters](#Policy-Gradient-Parameters)
- [Results](#results)

## Getting Started

### Prerequisites

- Python (>=3.6)
- NumPy
- TensorFlow (for Q-Learning)
- PyTorch (for Policy Gradient)
- Matplotlib

### Installation

 Clone the repository:

     ```bash
     git clone https://github.com/your-username/traveling-salesman.git
     cd traveling-salesman

## Usage

Run the file by  using the folllowing command

    ```bash
    python QL_PG.py

  ### play-around

  Change the parameters in the script and observe the graph.

  #### Q-Learning Parameters

    num_cities, num_episodes, epsilon, alpha, gamma 

  #### Policay Gradient Parameters

    num_cities, input_size, hidden_size, output_size, learning_rate, num_episodes

## Results

View the results for loss over episdoes for Q-Learning and Policy Gradient Respectively.


