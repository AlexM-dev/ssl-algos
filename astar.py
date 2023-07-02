import random
import numpy as np
import heapq
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline
import time

# Define the grid size and cell size
grid_size = 7500  # Total number of cells in each dimension
cell_size = 0.001  # Size of each grid cell
obstacle_radius = 500

start_point = (0, 0)
goal_point = (2000, 2000)
obstacle_positions = [(2000, 2000), (5000, 5000), (3000, 4000), (4000, 6000)]

start_time = time.time()

# Initialize the grid with default costs
grid = np.ones((grid_size, grid_size))

# Set the costs of cells containing obstacles to a high value (e.g., infinity)
for obstacle in obstacle_positions:
    x, y = obstacle
    grid[x, y] = float('inf')


# Helper function to calculate the heuristic cost (Euclidean distance) from a cell to the goal
def calculate_heuristic(cell):
    return np.linalg.norm(np.array(cell) - np.array(goal_point))


def is_within_obstacle_radius(cell):
    for obstacle in obstacle_positions:
        obstacle_x, obstacle_y = obstacle
        distance = np.linalg.norm(np.array(cell) - np.array(obstacle))
        if distance <= obstacle_radius:
            return True
    return False


# Implement the A* algorithm
def astar_algorithm():
    # Define the possible movements (up, down, left, right, diagonal)
    movements = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]

    # Initialize the open and closed sets
    open_set = []
    closed_set = set()

    # Initialize the costs and parent pointers for each cell
    costs = np.full((grid_size, grid_size), fill_value=float('inf'))
    parents = {}

    # Set the cost of the starting point to 0
    costs[start_point] = 0

    # Push the starting point to the open set
    heapq.heappush(open_set, (0, start_point))

    # Run the A* algorithm
    while open_set:

        # Pop the cell with the lowest cost from the open set
        current_cost, current_cell = heapq.heappop(open_set)

        # Check if the goal has been reached
        if current_cell == goal_point:
            break

        # Add the current cell to the closed set
        closed_set.add(current_cell)

        # Explore the neighbors of the current cell
        for dx, dy in movements:
            neighbor = (current_cell[0] + dx, current_cell[1] + dy)

            # Check if the neighbor is within the grid bounds and not in the closed set
            if 0 <= neighbor[0] < grid_size and 0 <= neighbor[1] < grid_size and neighbor not in closed_set:
                # Check if the neighbor is an obstacle
                if grid[neighbor] != float('inf') and not is_within_obstacle_radius(neighbor):
                    # Calculate the cost to move to the neighbor
                    movement_cost = np.linalg.norm(np.array((dx, dy))) * cell_size

                    # Calculate the new cost to reach the neighbor
                    new_cost = costs[current_cell] + movement_cost

                    # Update the cost and parent pointers if the new cost is lower
                    if new_cost < costs[neighbor]:
                        costs[neighbor] = new_cost
                        parents[neighbor] = current_cell

                        # Calculate the total cost (cost + heuristic) for the neighbor
                        total_cost = new_cost + calculate_heuristic(neighbor)

                        # Push the neighbor to the open set with the total cost
                        heapq.heappush(open_set, (total_cost, neighbor))

    # Generate the path by backtracking from the goal to the start
    path = []
    current = goal_point
    while current in parents:
        path.append(current)
        current = parents[current]
    path.append(start_point)
    path.reverse()

    return path


# Smooth the path using B-spline interpolation
def smooth_path(path):
    t = np.arange(len(path))
    x, y = zip(*path)
    x = np.array(x)
    y = np.array(y)

    # Add additional knots if there are not enough for degree 3 interpolation
    if len(t) < 8:
        t_new = np.linspace(0, len(path) - 1, 8)  # Use 8 knots for degree 3
        x_new = np.interp(t_new, t, x)
        y_new = np.interp(t_new, t, y)
    else:
        t_new = np.linspace(0, len(path) - 1, 10 * len(path))
        x_new = BSpline(t, x, 3)(t_new)  # Use order=3 for cubic B-spline
        y_new = BSpline(t, y, 3)(t_new)  # Use order=3 for cubic B-spline

    # Combine the smoothed x and y coordinates into a new path
    smoothed_path = [(x, y) for x, y in zip(x_new, y_new)]

    return smoothed_path


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
    smoothed_path = smooth_path(path)
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


# Run the A* algorithm
path = astar_algorithm()
print("1. --- %s seconds ---" % (time.time() - start_time))

# Visualize the environment and path
visualize_environment(path)
