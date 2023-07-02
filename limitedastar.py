import random
import numpy as np
import matplotlib.pyplot as plt
import time

# Define the grid size and cell size
grid_size = 7500  # Total number of cells in each dimension
cell_size = 0.001  # Size of each grid cell
obstacle_radius = 500

random.seed(15213)


def f():
    return random.randint(2, grid_size - 2)


# Define the starting point, goal point, and obstacle positions
start_point = (f(), f())  # (x, y) coordinates
goal_point = (f(), f())  # (x, y) coordinates

num_of = random.randint(0, 20)
obstacle_positions = []
for i in range(num_of):
    obstacle_positions.append((f(), f()))

start_point = (0, 0)
goal_point = (1000, 1000)
obstacle_positions = [(2000, 2000), (5000, 5000), (3000, 4000), (4000, 6000)]

start_time = time.time()

# Initialize the grid with default costs
grid = np.ones((grid_size, grid_size))

# Set the costs of cells containing obstacles to a high value (e.g., infinity)
for obstacle in obstacle_positions:
    x, y = obstacle
    grid[x, y] = float('inf')


# Implement the BFS algorithm
def bfs_algorithm():
    # Define the possible movements (up, down, left, right)
    movements = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    # Initialize the visited set and queue
    visited = set()
    queue = [(start_point, [])]

    # Run the BFS algorithm
    while queue:
        current_cell, path = queue.pop(0)
        print(current_cell)
        # Check if the goal has been reached
        if current_cell == goal_point:
            return path + [current_cell]

        # Add the current cell to the visited set
        visited.add(current_cell)

        # Explore the neighbors of the current cell
        for dx, dy in movements:
            neighbor = (current_cell[0] + dx, current_cell[1] + dy)

            # Check if the neighbor is within the grid bounds and not visited or an obstacle
            if 0 <= neighbor[0] < grid_size and 0 <= neighbor[1] < grid_size and neighbor not in visited and grid[
                neighbor] != float('inf'):
                # Add the neighbor to the queue with the updated path
                queue.append((neighbor, path + [current_cell]))

    # If no path is found, return an empty path
    return []


# Visualize the environment and path
def visualize_environment(path):
    # Create a grid of cells
    grid = np.zeros((grid_size, grid_size))

    # Plot the obstacles as circles
    for obstacle in obstacle_positions:
        x, y = obstacle
        circle = plt.Circle((x, y), radius=obstacle_radius, fc='gray', ec='black')
        plt.gca().add_patch(circle)

    # Plot the starting point, goal point, and robot position
    plt.plot(start_point[0], start_point[1], 'bo', markersize=8, label='Start')
    plt.plot(goal_point[0], goal_point[1], 'go', markersize=8, label='Goal')

    # Smooth the path
    time2 = time.time()
    smoothed_path = path
    print("2. --- %s seconds ---" % (time.time() - time2))
    plt.plot([p[0] for p in smoothed_path], [p[1] for p in smoothed_path], 'b--', linewidth=2,
             label='Smoothed Path')

    # Set the x and y axis limits
    plt.xlim(0, grid_size)
    plt.ylim(0, grid_size)

    # Add labels and legend
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()

    # Show the plot
    plt.show()


# Run the BFS algorithm
path = bfs_algorithm()
print("1. --- %s seconds ---" % (time.time() - start_time))

# Visualize the environment and path
visualize_environment(path)
print("2. --- %s seconds ---" % (time.time() - start_time))
