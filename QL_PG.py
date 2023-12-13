import sys
sys.path.append("../")


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


#For Q-Learning Algorithm


# Function to calculate Euclidean distance between two cities
def calculate_distance(city1, city2):
    return np.linalg.norm(np.array(city1) - np.array(city2))

# Function to initialize Q-values for state-action pairs
def initialize_q_values(num_cities):
    return np.zeros((num_cities, num_cities))

# Function to select an action using epsilon-greedy strategy
def select_action(q_values, state, epsilon):
    if np.random.rand() < epsilon:
        # Explore: Randomly choose an action
        return np.random.choice(len(q_values[state]))
    else:
        # Exploit: Choose the action with the highest Q-value
        return np.argmax(q_values[state])

# Function to update Q-values based on observed reward
def update_q_values(q_values, state, action, next_state, reward, alpha, gamma):
    best_next_action = np.argmax(q_values[next_state])
    q_values[state, action] += alpha * (reward + gamma * q_values[next_state, best_next_action] - q_values[state, action])

def tsp_environment(num_cities):
    cities = np.random.rand(num_cities, 2)  # Randomly generate city coordinates
    distances = np.zeros((num_cities, num_cities))

    # Populate distance matrix
    for i in range(num_cities):
        for j in range(num_cities):
            distances[i, j] = calculate_distance(cities[i], cities[j])

    return cities, distances


# Function to run Q-Learning for TSP
def run_q_learning(num_cities, num_episodes, epsilon, alpha, gamma):
    q_values = initialize_q_values(num_cities)
    cities, distances = tsp_environment(num_cities)

    episode_losses = []

    for episode in range(num_episodes):
        state = np.random.choice(num_cities)  # Start from a random city
        total_distance = 0

        for _ in range(num_cities - 1):
            action = select_action(q_values, state, epsilon)
            next_state = action
            reward = -distances[state, action]  # Negative distance as we want to minimize it
            total_distance += distances[state, action]

            update_q_values(q_values, state, action, next_state, reward, alpha, gamma)

            state = next_state

        # Return to the starting city to complete the tour
        total_distance += distances[state, np.argmax(q_values[state])]
        episode_losses.append(total_distance)
        #print(f"Episode {episode + 1}, Total Distance: {total_distance:.4f}")

    return episode_losses




# For TSP 


# Define the TSP environment
class TSPEnvironment:
    def __init__(self, num_cities):
        self.num_cities = num_cities
        self.cities = np.random.rand(num_cities, 2)  # Randomly generate city coordinates

    def reset(self):
        return np.zeros(self.num_cities)

    def get_distance(self, state, action):
        current_city = np.where(state == 0)[0][0]
        next_city = action
        distance = np.linalg.norm(self.cities[current_city] - self.cities[next_city])
        return distance

# Define the policy network
class PolicyNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=-1)
        return x

