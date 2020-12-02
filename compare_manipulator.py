"""
Created on Wed Aug  7 10:37:05 2019

This script generates the manipulator equations of motion for comparison with
Alex's Matlab script.

@author: Kirk
"""
from environment_manipulator import Environment
from environment_manipulator import dynamics_equations_of_motion
import numpy as np

x = 3.0
y = 1.0
theta = 0.0
theta_1 = np.pi/8
theta_2 = np.pi
theta_3 = -np.pi/3
x_dot= 2.6
y_dot = -0.875
theta_dot = 0
theta_1_dot = 0.1
theta_2_dot = -3
theta_3_dot = 0.22

env = Environment()

control_effort = np.array([0, 00, 0, 8, -8, 8])
LENGTH = env.LENGTH
PHI =env.PHI
B0 =env.B0
MASS =env.MASS
M1 =env.M1
M2 =env.M2
M3 =env.M3
A1 =env.A1
B1 =env.B1
A2 =env.A2
B2 =env.B2
A3 =env.A3
B3 =env.B3
INERTIA =env.INERTIA
INERTIA1 =env.INERTIA1
INERTIA2 =env.INERTIA2
INERTIA3 =env.INERTIA3

state = (x, y, theta, theta_1, theta_2, theta_3, x_dot, y_dot, theta_dot, theta_1_dot, theta_2_dot, theta_3_dot)
parameters = (control_effort, LENGTH, PHI, B0, MASS, M1, M2, M3, A1, B1, A2, B2, A3, B3, INERTIA, INERTIA1, INERTIA2, INERTIA3) # Unpacking parameters

derivatives = dynamics_equations_of_motion(state, 0, parameters)

print(derivatives)