import numpy as np
import random
import argparse
import time
import multiprocessing
import pygame
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.sparse import csr_matrix
from scipy.spatial import distance

torch.autograd.set_detect_anomaly(True)

class MazeGenerator(nn.Module):
    def __init__(self, size, hidden_size):
        super(MazeGenerator, self).__init__()
        self.size = size
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        return x

def torch_maze(size, model):
    maze = np.zeros((size, size))
    with torch.no_grad():
        for i in range(size):
            row = torch.tensor(maze[i], dtype=torch.float32)
            output = model(row)
            probs = torch.softmax(output, dim=0)
            maze[i] = torch.multinomial(probs, num_samples=1).numpy().reshape(-1)
    return maze

def train_maze_generator(size, hidden_size, num_epochs):
    # Create the model and optimizer
    model = MazeGenerator(size, hidden_size)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Train the model
    for epoch in range(num_epochs):
        # Generate a random maze and convert it to a tensor
        maze = np.random.randint(2, size=(size,size))
        maze_tensor = torch.tensor(maze, dtype=torch.float32)

        # Zero the gradients and perform a forward pass
        optimizer.zero_grad()
        output = model(maze_tensor)
        print("size: ",size)
        print("hidden: ", hidden_size)
        print("output: ",output)
        print("flatten output: ", output.int().flatten())
        print("flatten: ",maze_tensor.int().flatten())

        # Compute the loss and perform backpropagation
        loss = nn.functional.cross_entropy(output.float().flatten(), maze_tensor.float().flatten())
        loss.backward()
        optimizer.step()

        # Print the loss every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

    return model

def main():
    torch.autograd.set_detect_anomaly(True)
    # Set the maze size and number of epochs
    size = 20
    num_epochs = 100

    # Train the maze generator
    hidden_size = 100
    model = train_maze_generator(size, hidden_size, num_epochs)

    # Generate a maze using the trained model
    maze = torch_maze(size, model)

    # Display the maze
    print(maze)

if __name__ == "__main__":
    main()