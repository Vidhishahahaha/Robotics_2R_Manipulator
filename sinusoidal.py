import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

l1, l2 = 1.0, 1.0

def forward_kinematics(theta1, theta2):
    x1 = l1 * np.cos(theta1)
    y1 = l1 * np.sin(theta1)
    x2 = x1 + l2 * np.cos(theta1 + theta2)
    y2 = y1 + l2 * np.sin(theta1 + theta2)
    return x1, y1, x2, y2

def inverse_kinematics(x, y):
    r = np.sqrt(x**2 + y**2)
    cos_theta2 = (r**2 - l1**2 - l2**2) / (2 * l1 * l2)
    sin_theta2 = np.sqrt(1 - cos_theta2**2)
    theta2 = np.arctan2(sin_theta2, cos_theta2)
    k1 = l1 + l2 * cos_theta2
    k2 = l2 * sin_theta2
    theta1 = np.arctan2(y, x) - np.arctan2(k2, k1)
    return theta1, theta2

def sinusoidal_trajectory(t, amplitude=0.5, frequency=1.0):
    x = t * 2 - 1  
    y = amplitude * np.sin(2 * np.pi * frequency * t)
    return x, y

dt = 0.01
time_steps = np.arange(0, 1, dt)

original_x, original_y = [], []
recalculated_x, recalculated_y = [], []
for t in time_steps:
    x, y = sinusoidal_trajectory(t)
    original_x.append(x)
    original_y.append(y)
    theta1, theta2 = inverse_kinematics(x, y)
    _, _, x2, y2 = forward_kinematics(theta1, theta2)
    recalculated_x.append(x2)
    recalculated_y.append(y2)

fig, ax = plt.subplots()
ax.plot(original_x, original_y, 'ro', markersize=2, label="Original Trajectory")
ax.plot(recalculated_x, recalculated_y, 'b-', label="Recalculated Trajectory")
ax.set_aspect('equal')
ax.set_xlim(-2, 2)
ax.set_ylim(-1, 1)
ax.legend()
ax.grid()


def animate(i):
    ax.clear()

    ax.plot(original_x, original_y, 'ro', markersize=2, label="Original Trajectory")
    ax.plot(recalculated_x, recalculated_y, 'b-', label="Recalculated Trajectory")
    

    x, y = sinusoidal_trajectory(time_steps[i])
    theta1, theta2 = inverse_kinematics(x, y)
    x1, y1, x2, y2 = forward_kinematics(theta1, theta2)
    

    ax.plot([0, x1], [0, y1], 'm-o', label='Link 1')

    ax.plot([x1, x2], [y1, y2], 'g-o', label='Link 2')

    ax.plot(x2, y2, 'bo', label='End-Effector')
    
    ax.set_aspect('equal')
    ax.set_xlim(-2, 2)
    ax.set_ylim(-1, 1)
    ax.legend()
    ax.grid()
    ax.set_title("Comparison of Paths by Forward and Inverse Kinematics")


ani = FuncAnimation(fig, animate, frames=len(time_steps), interval=20)

plt.show()
