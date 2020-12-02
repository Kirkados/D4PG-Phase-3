clear
clc

% This script compares the manipulator dynamics between Python and Matlab
% to see if I've properly implemented them in Python

x = 3.0;
y = 1.0;
theta = 0.0;
theta_1 = pi/8;
theta_2 = pi;
theta_3 = -pi/3;
x_dot= 2.6;
y_dot = -0.875;
theta_dot = 0;
theta_1_dot = 0.1;
theta_2_dot = -3;
theta_3_dot = 0.22;

control_effort = [0.0, 00, -0, 8, -8, 8]';

q_dot = [x_dot, y_dot, theta_dot, theta_1_dot, theta_2_dot, theta_3_dot]';

LENGTH = 0.3;
PHI =pi/2;
B0 =0.15;
MASS =10;
M1 =1;
M2 =1;
M3 =1;
A1 =0.1;
B1 =0.1;
A2 =0.1;
B2 =0.1;
A3 =0.1;
B3 =0.1;
INERTIA  = 1/12*MASS*(LENGTH^2 + LENGTH^2);
INERTIA1 = 1/12*M1*(A1 + B1)^2;
INERTIA2 = 1/12*M2*(A2 + B2)^2;
INERTIA3 = 1/12*M3*(A3 + B3)^2;

% Getting the mass matrix
MassMatrix = InertiaFunc3LINK(INERTIA,INERTIA1,INERTIA2,INERTIA3,A1,A2,A3,B0,B1,B2,MASS,M1,M2,M3,PHI,theta,theta_1,theta_2,theta_3);
MassMatrix;

% Getting the coriolis matrix
CoriolisMatrix = CoriolisFunc3LINK(A1,A2,A3,B0,B1,B2,M1,M2,M3,PHI,theta,theta_1,theta_2,theta_3,theta_dot,theta_1_dot,theta_2_dot,theta_3_dot);
CoriolisMatrix;

derivatives = MassMatrix\(control_effort - CoriolisMatrix*q_dot)