import matplotlib.pyplot as plt
import numpy as np


class Robot:
    def __init__(self, start, goal, obstacles):
        self.position = np.array(start, dtype=float)
        self.goal = np.array(goal, dtype=float)
        self.obstacles = obstacles

    def attractive_force(self, k_att):
        return k_att * (self.goal - self.position)

    def repulsive_force(self, k_rep, eta):
        repulsive_force = np.array([0.0, 0.0])
        for obstacle in self.obstacles:
            distance = np.linalg.norm(self.position - obstacle)
            if distance < eta:
                repulsive_force += k_rep * ((1.0 / distance) - (1.0 / eta)) * \
                                  ((self.position - obstacle) / np.power(distance, 3))
        return repulsive_force

    def total_force(self, k_att, k_rep, eta):
        force = self.attractive_force(k_att) + self.repulsive_force(k_rep, eta)
        return force

    def update_position(self, delta_t, k_att, k_rep, eta):
        force = self.total_force(k_att, k_rep, eta)
        self.position += force * delta_t


def plot_path(robot, obstacles):
    plt.figure()
    for obstacle in obstacles:
        circle = plt.Circle(obstacle, 0.5, color='red')
        plt.gca().add_patch(circle)
    plt.plot(robot.position[0], robot.position[1], 'bo', label='Robot')
    plt.plot(robot.goal[0], robot.goal[1], 'go', label='Goal')
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
    plt.show()


# Test the code
start = (0.0, 0.0)
goal = (5.0, 5.0)
obstacles = [(2.0, 1.0), (-1.0, -3.0), (3.0, 1.0)]

robot = Robot(start, goal, obstacles)
delta_t = 0.1  # Time step size
k_att = 1.0  # Attractive force constant
k_rep = 5.0  # Repulsive force constant
eta = 1.0  # Influence distance

# Update robot position iteratively
for _ in range(100):
    robot.update_position(delta_t, k_att, k_rep, eta)

plot_path(robot, obstacles)
