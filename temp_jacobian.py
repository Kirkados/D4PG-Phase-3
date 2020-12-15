# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 15:36:59 2020

@author: Kirk
"""

import numpy as np

#q0 = np.pi/4
#q1 = 3*np.pi/8
#
#q_dot = np.array([1, 1]).reshape([2,1]) 
#Forces = np.array([0,0,0,0,0,1]).reshape([6,1])
#
#L0 = 1
#L1 = 1
#
#J= np.array([[-L0*np.sin(q0) - L1*np.sin(q0+q1), -L1*np.sin(q0+q1)],
#             [L0*np.cos(q0) + L1*np.cos(q0+q1),  L1*np.cos(q0+q1)],
#             [0,0],
#             [0,0],
#             [0,0],
#             [1,1]])
#
#print(np.matmul(J,q_dot))
#
## Transpose jacobian
#print(np.matmul(J.T,Forces))

PHI = np.pi/2
q0 = 0
q1 = 0
q2 = 0
q3 = 0


a0 = 0.1
b0 = 0.1
a1 = 0.1
b1 = 0.1
a2 = 0.1
b2 = 0.1
a3 = 0.1
b3 = 0.1

L0 = a0 + b0
L1 = a1 + b1
L2 = a2 + b2
L3 = a3 + b3

S0 = np.sin(PHI + q0)
S1 = np.sin(PHI + q0 + q1) 
S2 = np.sin(PHI + q0 + q1 + q2) 
S3 = np.sin(PHI + q0 + q1 + q2 + q3) 
C0 = np.cos(PHI + q0)
C1 = np.cos(PHI + q0 + q1) 
C2 = np.cos(PHI + q0 + q1 + q2)
C3 = np.cos(PHI + q0 + q1 + q2 + q3) 

Jc3_13 = -b0*S0 - L1*S1 - L2*S2 - a3*S3
Jc3_14 = -L1*S1 -L2*S2 - a3*S3
Jc3_15 = -L2*S2 -a3*S3
Jc3_16 = -a3*S3
Jc3_23 = b0*C0 + L1*C1 + L2*C2 +a3*C3
Jc3_24 = L1*C1 + L2*C2 + a3*C3
Jc3_25 = L2*C2 + a3*C3
Jc3_26 = a3*C3

q_dot = np.array([1,1,1,1,1,1])

Jc3 = np.array([[1,0,Jc3_13,Jc3_14,Jc3_15,Jc3_16],
                [0,1,Jc3_23,Jc3_24,Jc3_25,Jc3_26],
                [0,0,0,0,0,0],
                [0,0,0,0,0,0],
                [0,0,0,0,0,0],
                [0,0,1,1,1,1]])

# The velocity and angular rates of the end-effector given the qdot
v_w = np.matmul(Jc3,q_dot)
print(v_w)

# Transpose Jacobian time
# Calculating the forces for a desired end-effector force
EE_force = np.array([0,0,0,0,0,1]) # [EE_Fx, EE_Fy, EE_Fz, EE_tx, EE_ty, EE_tz]

joint_space_forces = np.matmul(Jc3.T,EE_force)
print(joint_space_forces)

# Seems legit!!
# Should I account for mass?