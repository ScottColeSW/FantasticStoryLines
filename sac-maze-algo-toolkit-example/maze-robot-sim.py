# this is a maze run simulation that demonstrates the speed difference between algorithems
# [TODO] add version for ???

import random
import argparse
import time
import multiprocessing
import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.sparse import csr_matrix
from scipy.spatial import distance

# Define available maze creation algorithms
ALGORITHMS = {
    "dfs": "Depth-First Search",
    "prim": "Prim's Algorithm",
    "kruskal": "Kruskal's Algorithm",
    "eller": "Eller's Algorithm",
    "prim-parallel": "Prim's Algorithm in Parallel",
    "torch": "Torch Implementation"
}

# Define available maze sizes
SIZES = {
    "tiny": 16,
    "small": 32,
    "medium": 64,
    "large": 128,
    "huge": 256
}

# Parse command line arguments
parser = argparse.ArgumentParser(description='Robot Maze Runner Choice Simulation')
parser.add_argument('--algorithm', choices=ALGORITHMS.keys(), default='prim', help=f'Choose maze creation algorithm (default: %(default)s)')
parser.add_argument('--size', choices=SIZES.keys(), default='medium', help=f'Choose maze size (default: %(default)s)')
parser.add_argument('--parallel', action='store_true', help='Use parallelism to speed up simulation')
args = parser.parse_args()

# Define maze creation functions
def dfs_maze(size):
    # Depth-First Search algorithm for maze creation
    maze = np.zeros((size, size), dtype=np.int8)
    visited = np.zeros((size, size), dtype=np.bool_)
    start = (random.randint(0, size-1), random.randint(0, size-1))
    stack = [start]
    while stack:
        current = stack.pop()
        if not visited[current]:
            visited[current] = True
            maze[current] = 1
            neighbors = []
            if current[0] > 0: neighbors.append((current[0]-1, current[1]))
            if current[0] < size-1: neighbors.append((current[0]+1, current[1]))
            if current[1] > 0: neighbors.append((current[0], current[1]-1))
            if current[1] < size-1: neighbors.append((current[0], current[1]+1))
            random.shuffle(neighbors)
            for neighbor in neighbors:
                if not visited[neighbor]:
                    stack.append(neighbor)
                    maze[current[0]+(neighbor[0]-current[0])//2][current[1]+(neighbor[1]-current[1])//2] = 1
    return maze

#This implementation uses a set of visited cells and a set of 
# frontier cells, which are the cells adjacent to visited cells 
# that have not yet been explored. At each step, the algorithm 
# chooses a random cell from the frontier, adds it to the visited 
# set, and adds its unvisited neighbors to the frontier. 
# It then chooses the closest neighbor to the start cell and adds 
# the corresponding wall to the maze.
def prim_maze(size):
    # Prim's algorithm for maze creation
    maze = np.zeros((size, size), dtype=np.int8)
    frontier = set()
    visited = set()
    start = (random.randint(0, size-1), random.randint(0, size-1))
    frontier.add(start)
    while frontier:
        current = random.choice(list(frontier))
        frontier.remove(current)
        visited.add(current)
        neighbors = []
        if current[0] > 0 and (current[0]-1, current[1]) not in visited:
            neighbors.append((current[0]-1, current[1]))
        if current[0] < size-1 and (current[0]+1, current[1]) not in visited:
            neighbors.append((current[0]+1, current[1]))
        if current[1] > 0 and (current[0], current[1]-1) not in visited:
            neighbors.append((current[0], current[1]-1))
        if current[1] < size-1 and (current[0], current[1]+1) not in visited:
            neighbors.append((current[0], current[1]+1))
        if neighbors:
            chosen = min(neighbors, key=lambda x: distance.euclidean(x, start))
            frontier.add(chosen)
            try:
                maze[current[0]+chosen[0]+1][current[1]+chosen[1]+1] = 1
            except IndexError as msg:
                continue
                #print(msg)
    return maze

def kruskal_maze(size):
    # Kruskal's algorithm for maze creation
    maze = np.zeros((size, size), dtype=np.int8)
    sets = {(i, j): [(i, j)] for i in range(size) for j in range(size)}
    edges = []
    for i in range(size):
        for j in range(size):
            if i > 0:
                edges.append(((i, j), (i-1, j)))
            if j > 0:
                edges.append(((i, j), (i, j-1)))
    random.shuffle(edges)
    for edge in edges:
        (i1, j1), (i2, j2) = edge
        set1 = sets[(i1, j1)]
        set2 = sets[(i2, j2)]
        if set1 != set2:
            maze[max(i1, i2)][max(j1, j2)] = 1
            set1.extend(set2)
            for x, y in set2:
                sets[(x, y)] = set1
    return maze

def eller_maze(size):
    # Eller's algorithm for maze creation
    
    # Create an empty maze of size (size, size) using a numpy array
    maze = np.zeros((size, size), dtype=np.int8)
    
    # Create a numpy array of indices representing the current row
    row = np.arange(size)
    
    # Loop over each row of the maze, except for the last row
    for i in range(size-1):
        
        # Generate a random mask for the current row, with values 0 or 1
        mask = np.random.randint(2, size=(size,))
        
        # Compute the sets of connected components in the current row using bitwise operations
        sets = np.unique(row * mask)
        
        # Assign a set number to each connected component in the current row
        for j, s in enumerate(sets):
            maze[i, s] = j+1
        
        # If we are at the last row, exit the loop
        if i == size-2:
            break
        
        # Iterate over the sets in the current row
        for j, s in enumerate(sets):
            
            # Break the connection to the next set with a 50% probability, except for the last set
            if j == len(sets)-1 or np.random.rand() < 0.5:
                mask[s] = 0
            
            # If the connection is not broken, randomly choose a cell from the current set that is connected to the next row
            # and assign it to the same set as the corresponding cell in the next row
            if j != len(sets)-1 and (mask[s] == 0 or s == sets[-1] or np.random.rand() < 0.5):
                choices = np.where(mask == 1)[0]
                chosen = np.random.choice(choices)
                maze[i+1, chosen] = maze[i, s]
                row[chosen] = row[s]
    
    # Return the completed maze
    return maze

from multiprocessing import Pool

def generate_submaze(args):
    start, end, size = args
    submaze = prim_maze(size, start[0], start[1], end[0], end[1])
    return submaze

def parallel_prim_maze(size):
    p = Pool(4)
    submaze_size = size // 4
    submazes = []
    
    for i in range(4):
        start_row = i * submaze_size
        end_row = start_row + submaze_size
        start_col = 0
        end_col = size
        
        if i == 3:
            end_row = size
        
        submaze_args = [(j, k, submaze_size) for j in range(start_row, end_row) for k in range(start_col, end_col)]
        submaze = p.map(generate_submaze, submaze_args)
        submaze = np.array(submaze).reshape(submaze_size, size)
        submazes.append(submaze)
    
    maze = np.vstack(submazes)
    maze = np.insert(maze, 0, 1, axis=0)
    maze = np.insert(maze, 0, 1, axis=1)
    maze[-1, -2] = 0
    maze[-2, -1] = 0
    
    return maze

class MazeGenerator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MazeGenerator, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

def torch_maze(size, model):
    input_size = size * size
    output_size = size * size
    
    maze_input = torch.zeros((1, input_size))
    maze_output = torch.zeros((1, output_size))
    
    for i in range(size):
        for j in range(size):
            idx = i * size + j
            maze_input[0, idx] = 1
            
            if i == 0 or i == size-1 or j == 0 or j == size-1:
                maze_output[0, idx] = 1
            else:
                prediction = model(maze_input)
                maze_output[0, idx] = prediction[0, idx]
            
            maze_input[0, idx] = maze_output[0, idx]
    
    maze = maze_output.view(size, size)
    maze = maze.detach().numpy()
    maze = np.round(maze).astype(int)
    
    return maze

def train_maze_generator(size, hidden_size, num_epochs):
    input_size = size * size
    output_size = size * size
    model = MazeGenerator(input_size, hidden_size, output_size)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        input_tensor = torch.zeros((1, input_size))
        output_tensor = torch.zeros((1, output_size))
        
        for i in range(size):
            for j in range(size):
                idx = i * size + j
                input_tensor[0, idx] = 1
                
                if i == 0 or i == size-1 or j == 0 or j == size-1:
                    output_tensor[0, idx] = 1
                else:
                    prediction = model(input_tensor)
                    output_tensor[0, idx] = prediction[0, idx]
                
                input_tensor[0, idx] = output_tensor[0, idx]
        
        loss = criterion(output_tensor, input_tensor)
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
    
    return model

def main():
    # Initialize Pygame
    pygame.init()
    
    # Prompt the user for the maze size and algorithm
    size = int(input("Enter maze size (): "))
    # Define available maze sizes
        # SIZES = {
        #     "tiny": 16,
        #     "small": 32,
        #     "medium": 64,
        #     "large": 128,
        #     "huge": 256
        # }
    algorithm = input("Enter maze generation algorithm (kruskal, prim-parallel, prim, eller, dfs, torch): ")
    
    # Generate the maze using the selected algorithm
    if algorithm == 'kruskal':
        maze = kruskal_maze(size)
    elif algorithm == 'prim':
        maze = prim_maze(size)
    elif algorithm == 'prim-parrallel':
        maze = parallel_prim_maze(size)
    elif algorithm == 'dfs':
        maze = dfs_maze(size)    
    elif algorithm == 'eller':
        maze = eller_maze(size)
    elif algorithm == 'torch':
        num_epochs = 100
        # Train the maze generator
        hidden_size = 100
        # Override size for Torch
        size = 20
        torch.autograd.set_detect_anomaly(True)
        model = train_maze_generator(size, hidden_size, num_epochs)
        # Generate a maze using the trained model
        maze = torch_maze(size, model)
    else:
        print("Invalid algorithm")
        return
    
    # Create a Pygame screen with appropriate dimensions
    screen = pygame.display.set_mode((maze.shape[1]*10, maze.shape[0]*10))
    
    # Set the Pygame caption
    pygame.display.set_caption("Maze")
    
    # Fill the screen with black
    screen.fill((0, 0, 0))
    
    # Loop over each cell in the maze
    for i in range(maze.shape[0]):
        for j in range(maze.shape[1]):
            
            # If the cell is a wall, draw a white rectangle
            if maze[i, j] == 1:
                rect = pygame.Rect(j*10, i*10, 10, 10)
                pygame.draw.rect(screen, (255, 255, 255), rect)
    
    # Update the Pygame display
    pygame.display.flip()
    
    # Wait for the user to close the Pygame window
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

if __name__ == '__main__':
    main()