
"""
This script provides the environment for a free flying spacecraft with a
three-link manipulator.

The spacecraft & manipulator are tasked with docking to a tumbling piece of debris
and bringing it to rest at a desired location. This will be accomplished as follows:
    1) First, learn how to become an expert at docking to a piece of debris safely. 
    2) Second, learn how to bring the chaser and target to rest at a desired location.
    Unsolved questions: 
        1) Should I train separate policies to accomplish each sub-task? Then, the moment the debris is 
           captured I could switch from policy 1 to policy 2? **Leaning towards this** 
              - Should I have to completely separate environments?? Probably!!
			  - The only problem is that there won't be continuity between the environments. Maybe I should 
			    capture with a certain speed to ease the detumbling???! I can imagine an optimal path having
				both goals in mind. Maybe a flag is what I'll need. Either way, let's just build the first 
				phase first.
        2) Should one policy be responsible for it all? With a flag saying if the target was 
           captured yet? Or maybe even without a flag?

The policy is trained in a DYNAMICS environment (in contrast to my previous work) for a number of reasons:
    1) The kinematics assumes that each state has no influence on any other state. The perfect controller
       is assumed to handle everything. This is fine for planar motion where there is no coupling between
       the states, but it does not apply to complex scenarios where there is coupling between the states.
       For example, moving the shoulder joint will affect the motion of the spacecraft--a kinematics 
       environment would not capture that.
    2) By training in a dynamics environment, the policy will overfit the simulated dynamics. For that 
       reason, I'll try to make them as accurate as possible. However, I will still be commanding 
       acceleration signals which an on-board controller is responsible for tracking. So, small
       changes should be tolerated so long as they do not spoil the learned logic.
    3) I'll also have to use a controller in simulation which will become overfit. However,
       overfitting to a real controller is probably better than overfitting to an ideal controller
       like we were doing before. Plus, we know what sort of controller we actually have in the lab.
    4) These changes are needed to solve the dynamic coupling problem present in most complex scenarios.

All dynamic environments I create will have a standardized architecture. The
reason for this is I have one learning algorithm and many environments. All
environments are responsible for:
    - dynamics propagation (via the step method)
    - initial conditions   (via the reset method)
    - reporting environment properties (defined in __init__)
    - seeding the dynamics (via the seed method)
    - animating the motion (via the render method):
        - Rendering is done all in one shot by passing the completed states
          from an episode to the render() method.

Outputs:
    Reward must be of shape ()
    State must be of shape (OBSERVATION_SIZE,)
    Done must be a bool

Inputs:
    Action input is of shape (ACTION_SIZE,)

Communication with agent:
    The agent communicates to the environment through two queues:
        agent_to_env: the agent passes actions or reset signals to the environment
        env_to_agent: the environment returns information to the agent

Reward system:
        - Zero reward at all timesteps except when docking is achieved
        - A large reward when docking occurs. The episode also terminates when docking occurs
        - A variety of penalties to help with docking, such as:
            - penalty for end-effector angle (so it goes into the docking cone properly)
            - penalty for relative velocity during the docking (so the end-effector doesn't jab the docking cone)
        - A penalty for colliding with the target

State clarity:
    - Note: TOTAL_STATE contains all relevant information describing the problem, and all the information needed to animate the motion
        = TOTAL_STATE is returned from the environment to the agent.
        = A subset of the TOTAL_STATE, called the 'observation', is passed to the policy network to calculate acitons. This takes place in the agent
        = The TOTAL_STATE is passed to the animator below to animate the motion.
        = The chaser and target state are contained in the environment. They are packaged up before being returned to the agent.
        = The total state information returned must be as commented beside self.TOTAL_STATE_SIZE.
        
        
Started December 2, 2020
@author: Kirk Hovell (khovell@gmail.com)
"""
import numpy as np
import os
import signal
import multiprocessing
import queue
from scipy.integrate import odeint # Numerical integrator

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec

from shapely.geometry import Point, Polygon # for collision detection

class Environment:

    def __init__(self):
        ##################################
        ##### Environment Properties #####
        ##################################
        START HERE
        self.TOTAL_STATE_SIZE         = 18 # [relative_x, relative_y, relative_vx, relative_vy, relative_angle, relative_angular_velocity, chaser_x, chaser_y, chaser_theta, target_x, target_y, target_theta, chaser_vx, chaser_vy, chaser_omega, target_vx, target_vy, target_omega] *# Relative pose expressed in the chaser's body frame; everythign else in Inertial frame #*
        ### Note: TOTAL_STATE contains all relevant information describing the problem, and all the information needed to animate the motion
        #         TOTAL_STATE is returned from the environment to the agent.
        #         A subset of the TOTAL_STATE, called the 'observation', is passed to the policy network to calculate acitons. This takes place in the agent
        #         The TOTAL_STATE is passed to the animator below to animate the motion.
        #         The chaser and target state are contained in the environment. They are packaged up before being returned to the agent.
        #         The total state information returned must be as commented beside self.TOTAL_STATE_SIZE.
        self.IRRELEVANT_STATES                = [6,7,8,9,10,11,12,13,14,15,16,17] # indices of states who are irrelevant to the policy network
        self.OBSERVATION_SIZE                 = self.TOTAL_STATE_SIZE - len(self.IRRELEVANT_STATES) # the size of the observation input to the policy
        self.ACTION_SIZE                      = 3 # [x_dot_dot, y_dot_dot, theta_dot_dot] in the BODY frame
        self.MAX_VELOCITY                     = 0.5 # [m/s]
        self.MAX_ANGULAR_VELOCITY             = np.pi/6 # [rad/s]
        self.LOWER_ACTION_BOUND               = np.array([-0.025, -0.025, -0.1]) # [m/s^2, m/s^2, rad/s^2]
        self.UPPER_ACTION_BOUND               = np.array([ 0.025,  0.025,  0.1]) # [m/s^2, m/s^2, rad/s^2]
        self.LOWER_STATE_BOUND                = np.array([-3., -3., -self.MAX_VELOCITY, -self.MAX_VELOCITY, -2*np.pi, -self.MAX_ANGULAR_VELOCITY, -3., -3., -2*np.pi, -3., -3., -2*np.pi, -self.MAX_VELOCITY, -self.MAX_VELOCITY, -self.MAX_ANGULAR_VELOCITY, -self.MAX_VELOCITY, -self.MAX_VELOCITY, -self.MAX_ANGULAR_VELOCITY]) # [m, m, m/s, m/s, rad, rad/s, m, m, rad, m, m, rad, m/s, m/s, rad/s, m/s, m/s, rad/s] // lower bound for each element of TOTAL_STATE
        self.UPPER_STATE_BOUND                = np.array([ 3.,  3.,  self.MAX_VELOCITY,  self.MAX_VELOCITY,  2*np.pi,  self.MAX_ANGULAR_VELOCITY,  3.,  3.,  2*np.pi,  3.,  3.,  2*np.pi,  self.MAX_VELOCITY,  self.MAX_VELOCITY,  self.MAX_ANGULAR_VELOCITY,  self.MAX_VELOCITY,  self.MAX_VELOCITY,  self.MAX_ANGULAR_VELOCITY]) # [m, m, m,s, m,s, rad, rad/s, m, m, rad, m, m, rad, m/s, m/s, rad/s, m/s, m/s, rad/s] // upper bound for each element of TOTAL_STATE
        self.INITIAL_CHASER_POSITION          = np.array([1.0, 1.2, 0.0]) # [m, m, rad]
        self.INITIAL_CHASER_VELOCITY          = np.array([0.0, 0.0, 0.0]) # [m/s, m/s, rad/s]
        self.INITIAL_TARGET_POSITION          = np.array([2.0, 1.0, 0.0]) # [m, m, rad]
        self.INITIAL_TARGET_VELOCITY          = np.array([0.0, 0.0, 0.0]) # [m/s, m/s, rad/s]
        self.NORMALIZE_STATE                  = True # Normalize state on each timestep to avoid vanishing gradients
        self.RANDOMIZE                        = False # whether or not to RANDOMIZE the state & target location
        self.RANDOMIZATION_LENGTH             = 0.5 # [m] standard deviation of position randomization
        self.RANDOMIZATION_ANGLE              = np.pi/2 # [rad] standard deviation of angular randomization
        self.RANDOMIZATION_TARGET_VELOCITY    = 0.0 # [m/s] standard deviation of the target's velocity randomization
        self.RANDOMIZATION_TARGET_OMEGA       = 0.0 # [rad/s] standard deviation of the target's angular velocity randomization
        self.MIN_V                            = -100.
        self.MAX_V                            =  100.
        self.N_STEP_RETURN                    =   5
        self.DISCOUNT_FACTOR                  =   0.95**(1/self.N_STEP_RETURN)
        self.TIMESTEP                         =   0.2 # [s]
        self.DYNAMICS_DELAY                   =   0 # [timesteps of delay] how many timesteps between when an action is commanded and when it is realized
        self.AUGMENT_STATE_WITH_ACTION_LENGTH =   0 # [timesteps] how many timesteps of previous actions should be included in the state. This helps with making good decisions among delayed dynamics.
        self.MAX_NUMBER_OF_TIMESTEPS          = 150 # per episode
        self.ADDITIONAL_VALUE_INFO            = False # whether or not to include additional reward and value distribution information on the animations
        self.SKIP_FAILED_ANIMATIONS           = True # Error the program or skip when animations fail?

        # Physical properties
        self.LENGTH                        = 0.3  # [m] side length
        self.MASS                          = 10.0   # [kg] for chaser
        self.INERTIA                       = 1/12*self.MASS*(self.LENGTH**2 + self.LENGTH**2) # 0.15 [kg m^2]
        self.DOCKING_PORT_MOUNT_POSITION   = np.array([0, self.LENGTH/2]) # position of the docking cone on the target in its body frame
        self.DOCKING_PORT_CORNER1_POSITION = self.DOCKING_PORT_MOUNT_POSITION + [ 0.05, 0.1] # position of the docking cone on the target in its body frame
        self.DOCKING_PORT_CORNER2_POSITION = self.DOCKING_PORT_MOUNT_POSITION + [-0.05, 0.1] # position of the docking cone on the target in its body frame
        self.ARM_MOUNT_POSITION            = np.array([0, self.LENGTH/2]) # [m] position of the arm mounting point on the chaser in the body frame
        self.SHOULDER_POSITION             = self.ARM_MOUNT_POSITION + [0, 0.05] # [m] position of the arm's shoulder in the chaser body frame
        self.ELBOW_POSITION                = self.SHOULDER_POSITION + [0.3*np.sin(np.pi/6), 0.3*np.cos(np.pi/6)] # [m] position of the arm's elbow in the chaser body frame
        self.WRIST_POSITION                = self.ELBOW_POSITION + [0.3*np.sin(np.pi/4),-0.3*np.cos(np.pi/4)] # [m] position of the arm's wrist in the chaser body frame
        self.END_EFFECTOR_POSITION         = self.WRIST_POSITION + [0.1, 0] # po sition of the optimally-deployed end-effector on the chaser in the body frame
        
        # Reward function properties
        self.DOCKING_REWARD                   = 100 # A lump-sum given to the chaser when it docks
        self.SUCCESSFUL_DOCKING_DISTANCE      = 0.03 # [m] distance at which the magnetic docking can occur
        self.MAX_DOCKING_ANGLE_PENALTY        = 25 # A penalty given to the chaser, upon docking, for having an angle when docking. The penalty is 0 upon perfect docking and MAX_DOCKING_ANGLE_PENALTY upon perfectly bad docking
        self.DOCKING_EE_VELOCITY_PENALTY      = 25 # A penalty given to the chaser, upon docking, for every 1 m/s end-effector collision velocity upon docking
        self.DOCKING_ANGULAR_VELOCITY_PENALTY = 25 # A penalty given to the chaser, upon docking, for every 1 rad/s angular body velocity upon docking
        self.END_ON_FALL                      = True # end episode on a fall off the table        
        self.FALL_OFF_TABLE_PENALTY           = 100.
        self.CHECK_CHASER_TARGET_COLLISION    = True
        self.TARGET_COLLISION_PENALTY         = 2 # [rewards/timestep] penalty given for colliding with target  
        self.CHECK_END_EFFECTOR_COLLISION     = True # Whether to do collision detection on the end-effector
        self.CHECK_END_EFFECTOR_FORBIDDEN     = True # Whether to expand the collision area to include the forbidden zone
        self.END_EFFECTOR_COLLISION_PENALTY   = 2 # [rewards/timestep] Penalty for end-effector collisions (with target or optionally with the forbidden zone)
        
        # Test time properties
        self.TEST_ON_DYNAMICS            = True # Whether or not to use full dynamics along with a PD controller at test time
        self.KINEMATIC_NOISE             = False # Whether or not to apply noise to the kinematics in order to simulate a poor controller
        self.KINEMATIC_POSITION_NOISE_SD = [0.2, 0.2, 0.2] # The standard deviation of the noise that is to be applied to each position element in the state
        self.KINEMATIC_VELOCITY_NOISE_SD = [0.1, 0.1, 0.1] # The standard deviation of the noise that is to be applied to each velocity element in the state
        self.FORCE_NOISE_AT_TEST_TIME    = False # [Default -> False] Whether or not to force kinematic noise to be present at test time
        self.KI                          = [10, 10, 0.05] # Integral gain for the integral-linear acceleration controller in [X, Y, and angle] (how fast does the commanded acceleration get realized)
        
        # Physical properties
        self.LENGTH   = 0.3  # [m] side length of spacecraft base
        self.PHI      = np.pi/2 # [rad] angle of anchor point of arm with respect to spacecraft body frame
        self.B0       = (self.LENGTH/2)/np.cos(np.pi/2-self.PHI) # scalar distance from centre of mass to arm attachment point
        self.MASS     = 10   # [kg]
        self.M1       = 1 # [kg] link mass
        self.M2       = 1 # [kg] link mass
        self.M3       = 1 # [kg] link mass
        self.A1       = 0.1 # [m] base of link to centre of mass
        self.B1       = 0.1 # [m] centre of mass to end of link
        self.A2       = 0.1 # [m] base of link to centre of mass
        self.B2       = 0.1 # [m] centre of mass to end of link
        self.A3       = 0.1 # [m] base of link to centre of mass
        self.B3       = 0.1 # [m] centre of mass to end of link
        self.INERTIA  = 1/12*self.MASS*(self.LENGTH**2 + self.LENGTH**2) # 0.15 [kg m^2] base inertia
        self.INERTIA1 = 1/12*self.M1*(self.A1 + self.B1)**2 # [kg m^2] link inertia
        self.INERTIA2 = 1/12*self.M2*(self.A2 + self.B2)**2 # [kg m^2] link inertia
        self.INERTIA3 = 1/12*self.M3*(self.A3 + self.B3)**2 # [kg m^2] link inertia
        
        
        
        # Some calculations that don't need to be changed
        self.VELOCITY_LIMIT           = np.array([self.MAX_VELOCITY, self.MAX_VELOCITY, self.MAX_ANGULAR_VELOCITY]) # [m/s, m/s, rad/s] maximum allowable velocity/angular velocity; a hard cap is enforced if this velocity is exceeded in kinematics & the controller enforces the limit in dynamics & experiment
        self.LOWER_STATE_BOUND        = np.concatenate([self.LOWER_STATE_BOUND, np.tile(self.LOWER_ACTION_BOUND, self.AUGMENT_STATE_WITH_ACTION_LENGTH)]) # lower bound for each element of TOTAL_STATE
        self.UPPER_STATE_BOUND        = np.concatenate([self.UPPER_STATE_BOUND, np.tile(self.UPPER_ACTION_BOUND, self.AUGMENT_STATE_WITH_ACTION_LENGTH)]) # upper bound for each element of TOTAL_STATE        
        self.OBSERVATION_SIZE         = self.TOTAL_STATE_SIZE - len(self.IRRELEVANT_STATES) # the size of the observation input to the policy


    ###################################
    ##### Seeding the environment #####
    ###################################
    def seed(self, seed):
        np.random.seed(seed)


    ######################################
    ##### Resettings the Environment #####
    ######################################
    def reset(self, use_dynamics, test_time):
        # This method resets the state and returns it
        """ NOTES:
               - if use_dynamics = True -> use dynamics
               - if test_time = True -> do not add "controller noise" to the kinematics
        """
        # Setting the default to be kinematics
        self.dynamics_flag = False

        # Logging whether it is test time for this episode
        self.test_time = test_time

        # If we are randomizing the initial conditions and state
        if self.RANDOMIZE:
            # Randomizing initial state in Inertial frame
            self.chaser_position = self.INITIAL_CHASER_POSITION + np.random.randn(3)*[self.RANDOMIZATION_LENGTH, self.RANDOMIZATION_LENGTH, self.RANDOMIZATION_ANGLE]
            # Randomizing target state in Inertial frame
            self.target_position = self.INITIAL_TARGET_POSITION + np.random.randn(3)*[self.RANDOMIZATION_LENGTH, self.RANDOMIZATION_LENGTH, self.RANDOMIZATION_ANGLE]
            # Randomizing target velocity in Inertial frame
            self.target_velocity = self.INITIAL_TARGET_VELOCITY + np.random.randn(3)*[self.RANDOMIZATION_TARGET_VELOCITY, self.RANDOMIZATION_TARGET_VELOCITY, self.RANDOMIZATION_TARGET_OMEGA]
            

        else:
            # Constant initial state in Inertial frame
            self.chaser_position = self.INITIAL_CHASER_POSITION
            # Constant target location in Inertial frame
            self.target_position = self.INITIAL_TARGET_POSITION
            # Constant target velocity in Inertial frame
            self.target_velocity = self.INITIAL_TARGET_VELOCITY
        
        # Resetting the chaser's initial velocity
        self.chaser_velocity = self.INITIAL_CHASER_VELOCITY
        
        # Update docking component locations
        self.update_docking_locations()
        
        # Check for collisions
        self.check_collisions()

        # Initializing the previous velocity and control effort for the integral-acceleration controller
        self.previous_velocity = np.zeros(len(self.INITIAL_CHASER_VELOCITY))
        self.previous_control_effort = np.zeros(self.ACTION_SIZE)
                
        if use_dynamics:            
            self.dynamics_flag = True # for this episode, dynamics will be used

        # Resetting the time
        self.time = 0.

        # Resetting the action delay queue
        if self.DYNAMICS_DELAY > 0:
            self.action_delay_queue = queue.Queue(maxsize = self.DYNAMICS_DELAY + 1)
            for i in range(self.DYNAMICS_DELAY):
                self.action_delay_queue.put(np.zeros(self.ACTION_SIZE), False)

    def end_effector_position(self):
        """
        This method returns the location of the end-effector of the manipulator
        based off the current state
        """

        # Unpacking the state
        x, y, theta, theta_1, theta_2, theta_3 = self.state[:self.POSITION_STATE_LENGTH]

        x_ee = x + self.B0*np.cos(self.PHI + theta) + (self.A1 + self.B1)*np.cos(np.pi/2 + theta + theta_1) + \
               (self.A2 + self.B2)*np.cos(np.pi/2 + theta + theta_1 + theta_2) + \
               (self.A3 + self.B3)*np.cos(np.pi/2 + theta + theta_1 + theta_2 + theta_3)

        y_ee = y + self.B0*np.sin(self.PHI + theta) + (self.A1 + self.B1)*np.sin(np.pi/2 + theta + theta_1) + \
               (self.A2 + self.B2)*np.sin(np.pi/2 + theta + theta_1 + theta_2) + \
               (self.A3 + self.B3)*np.sin(np.pi/2 + theta + theta_1 + theta_2 + theta_3)

        return np.array([x_ee, y_ee])

    def update_docking_locations(self):
        # Updates the position of the end-effector and the docking port in the Inertial frame
        
        # Make rotation matrices        
        C_Ib_chaser = self.make_C_bI(self.chaser_position[-1]).T
        C_Ib_target = self.make_C_bI(self.target_position[-1]).T
        
        # Position in Inertial = Body position (inertial) + C_Ib * EE position in body
        self.end_effector_position = self.chaser_position[:-1] + np.matmul(C_Ib_chaser, self.END_EFFECTOR_POSITION)
        self.docking_port_position = self.target_position[:-1] + np.matmul(C_Ib_target, self.DOCKING_PORT_MOUNT_POSITION)

        
    #####################################
    ##### Step the Dynamics forward #####
    #####################################
    def step(self, action):

        # Integrating forward one time step using the calculated action.
        # Oeint returns initial condition on first row then next TIMESTEP on the next row
        #########################################
        ##### PROPAGATE KINEMATICS/DYNAMICS #####
        #########################################
        if self.dynamics_flag:
            ############################
            #### PROPAGATE DYNAMICS ####
            ############################

            # First, calculate the control effort
            control_effort = self.controller(action)

            # Anything that needs to be sent to the dynamics integrator
            dynamics_parameters = [control_effort, self.MASS, self.INERTIA]

            # Propagate the dynamics forward one timestep
            next_states = odeint(dynamics_equations_of_motion, np.concatenate([self.chaser_position, self.chaser_velocity]), [self.time, self.time + self.TIMESTEP], args = (dynamics_parameters,), full_output = 0)

            # Saving the new state
            self.chaser_position = next_states[1,:len(self.INITIAL_CHASER_POSITION)] # extract position
            self.chaser_velocity = next_states[1,len(self.INITIAL_CHASER_POSITION):] # extract velocity

        else:

            # Parameters to be passed to the kinematics integrator
            kinematics_parameters = [action, len(self.INITIAL_CHASER_POSITION)]

            ###############################
            #### PROPAGATE KINEMATICS #####
            ###############################
            next_states = odeint(kinematics_equations_of_motion, np.concatenate([self.chaser_position, self.chaser_velocity]), [self.time, self.time + self.TIMESTEP], args = (kinematics_parameters,), full_output = 0)

            # Saving the new state
            self.chaser_position = next_states[1,:len(self.INITIAL_CHASER_POSITION)] # extract position
            self.chaser_velocity = next_states[1,len(self.INITIAL_CHASER_POSITION):] # extract velocity  
            
            # Optionally, add noise to the kinematics to simulate "controller noise"
            if self.KINEMATIC_NOISE and (not self.test_time or self.FORCE_NOISE_AT_TEST_TIME):
                 # Add some noise to the position part of the state
                 self.chaser_position += np.random.randn(len(self.chaser_position)) * self.KINEMATIC_POSITION_NOISE_SD
                 self.chaser_velocity += np.random.randn(len(self.chaser_velocity)) * self.KINEMATIC_VELOCITY_NOISE_SD
            
            # Ensuring the velocity is within the bounds
            self.chaser_velocity = np.clip(self.chaser_velocity, -self.VELOCITY_LIMIT, self.VELOCITY_LIMIT)


        # Step target's state ahead one timestep
        self.target_position += self.INITIAL_TARGET_VELOCITY * self.TIMESTEP

        # Update docking locations
        self.update_docking_locations()
        
        # Check for collisions
        self.check_collisions()
        
        # Increment the timestep
        self.time += self.TIMESTEP

        # Calculating the reward for this state-action pair
        reward = self.reward_function(action)

        # Check if this episode is done
        done = self.is_done()

        # Return the (reward, done)
        return reward, done


    def controller(self, action):
        # This function calculates the control effort based on the state and the
        # desired acceleration (action)
        
        ########################################
        ### Integral-acceleration controller ###
        ########################################
        desired_accelerations = action
        
        current_velocity = self.chaser_velocity # [v_x, v_y, omega]
        current_accelerations = (current_velocity - self.previous_velocity)/self.TIMESTEP # Approximating the current acceleration [a_x, a_y, alpha]
        
        # Checking whether our velocity is too large AND the acceleration is trying to increase said velocity... in which case we set the desired_linear_acceleration to zero.
        desired_accelerations[(np.abs(current_velocity) > self.VELOCITY_LIMIT) & (np.sign(desired_accelerations) == np.sign(current_velocity))] = 0        
        
        # Calculating acceleration error
        acceleration_error = desired_accelerations - current_accelerations
        
        # Integral-acceleration control
        control_effort = self.previous_control_effort + self.KI * acceleration_error

        # Saving the current velocity for the next timetsep
        self.previous_velocity = current_velocity
        
        # Saving the current control effort for the next timestep
        self.previous_control_effort = control_effort

        # [F_x, F_y, torque]
        return control_effort


    def reward_function(self, action):
        # Returns the reward for this TIMESTEP as a function of the state and action
        
        """
        Reward system:
                - Zero reward at all timesteps except when docking is achieved
                - A large reward when docking occurs. The episode also terminates when docking occurs
                - A variety of penalties to help with docking, such as:
                    - penalty for end-effector angle (so it goes into the docking cone properly)
                    - penalty for relative velocity during the docking (so the end-effector doesn't jab the docking cone)
                - A penalty for colliding with the target
         """ 
                
        # Initializing the reward
        reward = 0
        
        # Give a large reward for docking
        if np.linalg.norm(self.end_effector_position - self.docking_port_position) <= self.SUCCESSFUL_DOCKING_DISTANCE:
            
            reward += self.DOCKING_REWARD
            
            # Penalize for end-effector angle
            # end-effector angle in the chaser body frame
            end_effector_angle_body = np.arctan2(self.END_EFFECTOR_POSITION[1] - self.WRIST_POSITION[1],self.END_EFFECTOR_POSITION[0] - self.WRIST_POSITION[0])
            end_effector_angle_inertial = end_effector_angle_body + self.chaser_position[-1]
            
            # Docking cone angle in the target body frame
            docking_cone_angle_body = np.arctan2(self.DOCKING_PORT_CORNER1_POSITION[1] - self.DOCKING_PORT_CORNER2_POSITION[1], self.DOCKING_PORT_CORNER1_POSITION[0] - self.DOCKING_PORT_CORNER2_POSITION[0])
            docking_cone_angle_inertial = docking_cone_angle_body + self.target_position[-1] - np.pi/2 # additional -pi/2 since we must dock perpendicular into the cone
            
            # Calculate the docking angle error
            docking_angle_error = (docking_cone_angle_inertial - end_effector_angle_inertial + np.pi) % (2*np.pi) - np.pi # wrapping to [-pi, pi] 
            
            # Penalize for any non-zero angle
            reward -= np.abs(np.sin(docking_angle_error/2)) * self.MAX_DOCKING_ANGLE_PENALTY
                        
            # Penalize for relative velocity during docking
            # Calculating the end-effector velocity; v_e = v_0 + omega x r_e/0
            end_effector_velocity = self.chaser_velocity[:-1] + self.chaser_velocity[-1] * np.matmul(self.make_C_bI(self.chaser_position[-1]).T,[-self.END_EFFECTOR_POSITION[1], self.END_EFFECTOR_POSITION[0]])
            
            # Calculating the docking cone velocity
            docking_cone_velocity = self.target_velocity[:-1] + self.target_velocity[-1] * np.matmul(self.make_C_bI(self.target_position[-1]).T,[-self.DOCKING_PORT_MOUNT_POSITION[1], self.DOCKING_PORT_MOUNT_POSITION[0]])
            
            # Calculating the docking velocity error
            docking_relative_velocity = end_effector_velocity - docking_cone_velocity
            
            # Applying the penalty
            reward -= np.linalg.norm(docking_relative_velocity) * self.DOCKING_EE_VELOCITY_PENALTY # 
            
            # Penalize for chaser angular velocity upon docking
            reward -= np.abs(self.chaser_velocity[-1] - self.target_velocity[-1]) * self.DOCKING_ANGULAR_VELOCITY_PENALTY
            
            if self.test_time:
                print("docking successful! Reward given: %.1f distance: %.3f relative velocity: %.3f velocity penalty: %.1f docking angle: %.2f angle penalty: %.1f angular rate error: %.3f angular rate penalty %.1f" %(reward, np.linalg.norm(self.end_effector_position - self.docking_port_position), np.linalg.norm(docking_relative_velocity), np.linalg.norm(docking_relative_velocity) * self.DOCKING_EE_VELOCITY_PENALTY, docking_angle_error*180/np.pi, np.abs(np.sin(docking_angle_error/2)) * self.MAX_DOCKING_ANGLE_PENALTY,np.abs(self.chaser_velocity[-1] - self.target_velocity[-1]),np.abs(self.chaser_velocity[-1] - self.target_velocity[-1]) * self.DOCKING_ANGULAR_VELOCITY_PENALTY))
        
        
        # Giving a penalty for colliding with the target
        if self.chaser_target_collision:
            reward -= self.TARGET_COLLISION_PENALTY
        
        if self.end_effector_collision:
            reward -= self.END_EFFECTOR_COLLISION_PENALTY
        
        if self.forbidden_area_collision:
            reward -= self.END_EFFECTOR_COLLISION_PENALTY
        
        # If we've fallen off the table, penalize this behaviour
        if self.chaser_position[0] > 4 or self.chaser_position[0] < -1 or self.chaser_position[1] > 3 or self.chaser_position[1] < -1 or self.chaser_position[2] > 6*np.pi or self.chaser_position[2] < -6*np.pi:
            reward -= self.FALL_OFF_TABLE_PENALTY

        return reward # possibly add .squeeze() if the shape is not ()
    
    def check_collisions(self):
        """ Calculate whether the different objects are colliding with the target. 
        
            Returns 3 booleans: end_effector_collision, forbidden_area_collision, chaser_target_collision
        """
        
        ##################################################
        ### Calculating Polygons in the inertial frame ###
        ##################################################
        
        # Target    
        target_points_body = np.array([[ self.LENGTH/2,-self.LENGTH/2],
                                       [-self.LENGTH/2,-self.LENGTH/2],
                                       [-self.LENGTH/2, self.LENGTH/2],
                                       [ self.LENGTH/2, self.LENGTH/2]]).T    
        # Rotation matrix (body -> inertial)
        C_Ib_target = self.make_C_bI(self.target_position[-1]).T        
        # Rotating body frame coordinates to inertial frame
        target_body_inertial = np.matmul(C_Ib_target, target_points_body) + np.array([self.target_position[0], self.target_position[1]]).reshape([2,-1])
        target_polygon = Polygon(target_body_inertial.T)
        
        # Forbidden Area
        forbidden_area_body = np.array([[self.LENGTH/2, self.LENGTH/2],   
                                        [self.DOCKING_PORT_CORNER1_POSITION[0],self.DOCKING_PORT_CORNER1_POSITION[1]],
                                        [self.DOCKING_PORT_MOUNT_POSITION[0],self.DOCKING_PORT_MOUNT_POSITION[1]],
                                        [self.DOCKING_PORT_CORNER2_POSITION[0],self.DOCKING_PORT_CORNER2_POSITION[1]],
                                        [-self.LENGTH/2,self.LENGTH/2],
                                        [self.LENGTH/2, self.LENGTH/2]]).T        
        # Rotating body frame coordinates to inertial frame
        forbidden_area_inertial = np.matmul(C_Ib_target, forbidden_area_body) + np.array([self.target_position[0], self.target_position[1]]).reshape([2,-1])         
        forbidden_polygon = Polygon(forbidden_area_inertial.T)
        
        # End-effector
        end_effector_point = Point(self.end_effector_position)
        
        # Chaser
        chaser_points_body = np.array([[ self.LENGTH/2,-self.LENGTH/2],
                                       [-self.LENGTH/2,-self.LENGTH/2],
                                       [-self.LENGTH/2, self.LENGTH/2],
                                       [ self.LENGTH/2, self.LENGTH/2]]).T    
        # Rotation matrix (body -> inertial)
        C_Ib_chaser = self.make_C_bI(self.chaser_position[-1]).T        
        # Rotating body frame coordinates to inertial frame
        chaser_body_inertial = np.matmul(C_Ib_chaser, chaser_points_body) + np.array([self.chaser_position[0], self.chaser_position[1]]).reshape([2,-1])
        chaser_polygon = Polygon(chaser_body_inertial.T)
        
        
        ###########################
        ### Checking collisions ###
        ###########################
        self.end_effector_collision = False
        self.forbidden_area_collision = False
        self.chaser_target_collision = False
        
        if self.CHECK_END_EFFECTOR_COLLISION and end_effector_point.within(target_polygon):
            if self.test_time:
                print("End-effector colliding with the target!")
            self.end_effector_collision = True
        
        if self.CHECK_END_EFFECTOR_FORBIDDEN and end_effector_point.within(forbidden_polygon):
            if self.test_time:
                print("End-effector within the forbidden area!")
            self.forbidden_area_collision = True
        
        if self.CHECK_CHASER_TARGET_COLLISION and chaser_polygon.intersects(target_polygon):
            if self.test_time:
                print("Chaser/target collision")
            self.chaser_target_collision = True
            
        if not np.any([self.end_effector_collision, self.forbidden_area_collision, self.chaser_target_collision]):
            pass
            #print("No collisions")                                                            


    def is_done(self):
        # Checks if this episode is done or not
        """
            NOTE: THE ENVIRONMENT MUST RETURN done = True IF THE EPISODE HAS
                  REACHED ITS LAST TIMESTEP
        """
        done = False
        
        # If we've docked with the target
        if np.linalg.norm(self.end_effector_position - self.docking_port_position) <= self.SUCCESSFUL_DOCKING_DISTANCE:
            return True

        # If we've fallen off the table, end the episode
        if self.chaser_position[0] > 4 or self.chaser_position[0] < -1 or self.chaser_position[1] > 3 or self.chaser_position[1] < -1 or self.chaser_position[2] > 6*np.pi or self.chaser_position[2] < -6*np.pi:
            if self.test_time:
                print("Fell off table!")
            return True

        # If we've run out of timesteps
        if round(self.time/self.TIMESTEP) == self.MAX_NUMBER_OF_TIMESTEPS:
            return True

        return done


    def generate_queue(self):
        # Generate the queues responsible for communicating with the agent
        self.agent_to_env = multiprocessing.Queue(maxsize = 1)
        self.env_to_agent = multiprocessing.Queue(maxsize = 1)

        return self.agent_to_env, self.env_to_agent

    
    def make_C_bI(self, angle):
        
        C_bI = np.array([[ np.cos(angle), np.sin(angle)],
                         [-np.sin(angle), np.cos(angle)]]) # [2, 2]        
        return C_bI
    
    
    def relative_pose_body_frame(self):
        # Calculate the relative_x, relative_y, relative_vx, relative_vy, relative_angle, relative_angular_velocity
        # All in the body frame
                
        chaser_angle = self.chaser_position[-1]        
        # Rotation matrix (inertial -> body)
        C_bI = self.make_C_bI(chaser_angle)
                
        # [X,Y] relative position in inertial frame
        relative_position_inertial = self.target_position[:-1] - self.chaser_position[:-1]    
        relative_position_body = np.matmul(C_bI, relative_position_inertial)
        
        # [X, Y] Relative velocity in inertial frame
        relative_velocity_inertial = self.target_velocity[:-1] - self.chaser_velocity[:-1]
        relative_velocity_body = np.matmul(C_bI, relative_velocity_inertial)
        
        # Relative angle and wrap it to [0, 2*np.pi]
        relative_angle = np.array([(self.target_position[-1] - self.chaser_position[-1])%(2*np.pi)])
        
        # Relative angular velocity
        relative_angular_velocity = np.array([self.target_velocity[-1] - self.chaser_velocity[-1]])

        return np.concatenate([relative_position_body, relative_velocity_body, relative_angle, relative_angular_velocity])


    def run(self):
        ###################################
        ##### Running the environment #####
        ###################################
        """
        This method is called when the environment process is launched by main.py.
        It is responsible for continually listening for an input action from the
        agent through a Queue. If an action is received, it is to step the environment
        and return the results.
        
        TOTAL_STATE_SIZE = 18 [relative_x, relative_y, relative_vx, relative_vy, relative_angle, relative_angular_velocity, 
                               chaser_x, chaser_y, chaser_theta, target_x, target_y, target_theta, 
                               chaser_vx, chaser_vy, chaser_omega, target_vx, target_vy, target_omega] *# Relative pose expressed in the chaser's body frame; everythign else in Inertial frame #*        
        """
        # Instructing this process to treat Ctrl+C events (called SIGINT) by going SIG_IGN (ignore).
        # This permits the process to continue upon a Ctrl+C event to allow for graceful quitting.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        
        # Loop until the process is terminated
        while True:
            # Blocks until the agent passes us an action
            action, *test_time = self.agent_to_env.get()

            if type(action) == bool:
                # The signal to reset the environment was received
                self.reset(action, test_time[0])
                
                # Return the TOTAL_STATE
                self.env_to_agent.put(np.concatenate([self.relative_pose_body_frame(), self.chaser_position, self.target_position, self.chaser_velocity, self.target_velocity]))

            else:
                
                # Delay the action by DYNAMICS_DELAY timesteps. The environment accumulates the action delay--the agent still thinks the sent action was used.
                if self.DYNAMICS_DELAY > 0:
                    self.action_delay_queue.put(action,False) # puts the current action to the bottom of the stack
                    action = self.action_delay_queue.get(False) # grabs the delayed action and treats it as truth.   
                
                # Rotating the action from the body frame into the inertial frame
                action[:-1] = np.matmul(self.make_C_bI(self.chaser_position[-1]).T, action[:-1])
            

                ################################
                ##### Step the environment #####
                ################################                
                reward, done = self.step(action)

                # Return (TOTAL_STATE, reward, done)
                self.env_to_agent.put((np.concatenate([self.relative_pose_body_frame(), self.chaser_position, self.target_position, self.chaser_velocity, self.target_velocity]), reward, done))


#####################################################################
##### Generating the dynamics equations representing the motion #####
#####################################################################
def dynamics_equations_of_motion(state, t, parameters):
    # state = [x, y, theta, xdot, ydot, thetadot]

    # Unpacking the state
    x, y, theta, theta_1, theta_2, theta_3, x_dot, y_dot, theta_dot, theta_1_dot, theta_2_dot, theta_3_dot = state

    control_effort, LENGTH, PHI, B0, \
    MASS, M1, M2, M3, \
    A1, B1, A2, B2, A3, B3, \
    INERTIA, INERTIA1, INERTIA2, INERTIA3 = parameters # Unpacking parameters


    # Generating mass matrix using Alex's equations from InertiaFinc3LINK
    t2 = A1+B1
    t3 = A1*M1
    t4 = M2*t2
    t5 = M3*t2
    t6 = theta+theta_1
    t7 = np.cos(t6)
    t8 = t3+t4+t5
    t9 = A2*M2
    t10 = A2+B2
    t11 = M3*t10
    t12 = theta+theta_1+theta_2
    t13 = np.cos(t12)
    t14 = t9+t11
    t15 = theta+theta_1+theta_2+theta_3
    t16 = np.cos(t15)
    t17 = MASS+M1+M2+M3
    t18 = B0*M1
    t19 = B0*M2
    t20 = B0*M3
    t21 = t18+t19+t20
    t22 = PHI+theta
    t23 = np.sin(t6)
    t24 = np.sin(t12)
    t25 = np.sin(t15)
    t26 = A2*M3
    t27 = B2*M3
    t28 = t9+t26+t27
    t29 = np.sin(t22)
    t86 = t7*t8
    t87 = t13*t14
    t88 = A3*M3*t16
    t30 = -t86-t87-t88-t21*t29
    t31 = np.cos(t22)
    t32 = t21*t31
    t89 = t8*t23
    t90 = t14*t24
    t91 = A3*M3*t25
    t33 = t32-t89-t90-t91
    t34 = A1**2
    t35 = A2**2
    t36 = B0**2
    t37 = B1**2
    t38 = A1*A2*M2*2.0
    t39 = A1*A2*M3*2.0
    t40 = A2*B1*M2*2.0
    t41 = A1*B2*M3*2.0
    t42 = A2*B1*M3*2.0
    t43 = B1*B2*M3*2.0
    t44 = t38+t39+t40+t41+t42+t43
    t45 = np.cos(theta_2)
    t46 = t44*t45
    t47 = A2*A3*M3*2.0
    t48 = A3*B2*M3*2.0
    t49 = t47+t48
    t50 = np.cos(theta_3)
    t51 = t49*t50
    t52 = A1*A3*M3*2.0
    t53 = A3*B1*M3*2.0
    t54 = t52+t53
    t55 = theta_2+theta_3
    t56 = np.cos(t55)
    t57 = t54*t56
    t58 = PHI-theta_1
    t59 = np.sin(t58)
    t60 = -PHI+theta_1+theta_2
    t61 = np.sin(t60)
    t62 = -PHI+theta_1+theta_2+theta_3
    t63 = np.sin(t62)
    t64 = M1*t34
    t65 = M2*t34
    t66 = M3*t34
    t67 = M2*t35
    t68 = M3*t35
    t69 = A3**2
    t70 = M3*t69
    t71 = M2*t37
    t72 = M3*t37
    t73 = B2**2
    t74 = M3*t73
    t75 = A1*B1*M2*2.0
    t76 = A1*B1*M3*2.0
    t77 = A2*B2*M3*2.0
    t78 = A2*B0*M2
    t79 = A2*B0*M3
    t80 = B0*B2*M3
    t81 = t78+t79+t80
    t82 = A1*A3*M3
    t83 = A3*B1*M3
    t84 = t82+t83
    t85 = t56*t84
    t92 = A1*B0*M1
    t93 = A1*B0*M2
    t94 = A1*B0*M3
    t95 = B0*B1*M2
    t96 = B0*B1*M3
    t97 = t92+t93+t94+t95+t96
    t98 = t59*t97
    t112 = t61*t81
    t113 = A3*B0*M3*t63
    t99 = INERTIA1+INERTIA2+INERTIA3+t46+t51+t57+t64+t65+t66+t67+t68+t70+t71+t72+t74+t75+t76+t77+t98-t112-t113
    t100 = A1*A2*M2
    t101 = A1*A2*M3
    t102 = A2*B1*M2
    t103 = A1*B2*M3
    t104 = A2*B1*M3
    t105 = B1*B2*M3
    t106 = t100+t101+t102+t103+t104+t105
    t107 = t45*t106
    t108 = A2*A3*M3
    t109 = A3*B2*M3
    t110 = t108+t109
    t111 = t50*t110
    t114 = INERTIA2+INERTIA3+t51+t67+t68+t70+t74+t77+t85+t107
    t115 = INERTIA3+t70+t85+t111
    t116 = INERTIA3+t70+t111
    MassMatrix = np.array([t17,0.0,t30,-t86-t87-t88,-t88-t13*t28,-t88,0.0,t17,t33,-t89-t90-t91,-t91-t24*t28,-t91,t30,t33,INERTIA+INERTIA1+INERTIA2+INERTIA3+t46+t51+t57+t64+t65+t66+t67+t68+t70+t71+t72+t74+t75+t76+t77+t59*(A1*B0*M1*2.0+A1*B0*M2*2.0+A1*B0*M3*2.0+B0*B1*M2*2.0+B0*B1*M3*2.0)+M1*t36+M2*t36+M3*t36-t61*(A2*B0*M2*2.0+A2*B0*M3*2.0+B0*B2*M3*2.0)-A3*B0*M3*t63*2.0,t99,INERTIA2+INERTIA3+t51+t67+t68+t70+t74+t77+t85+t107-t112-t113,INERTIA3+t70+t85+t111-t113,-t7*t8-t13*t14-A3*M3*t16,-t8*t23-t14*t24-A3*M3*t25,t99,INERTIA1+INERTIA2+INERTIA3+t46+t51+t57+t64+t65+t66+t67+t68+t70+t71+t72+t74+t75+t76+t77,t114,t115,-t13*t28-A3*M3*t16,-t24*t28-A3*M3*t25,INERTIA2+INERTIA3+t51+t67+t68+t70+t74+t77+t85+t107-t61*t81-A3*B0*M3*t63,t114,INERTIA2+INERTIA3+t51+t67+t68+t70+t74+t77,t116,-A3*M3*t16,-A3*M3*t25,INERTIA3+t70+t85+t111-A3*B0*M3*t63,t115,t116,INERTIA3+t70]).reshape([6,6], order ='F') # default order is different from matlab

    # Generating coriolis matrix using Alex's equations from CoriolisFinc3LINK
    t2 = A1+B1
    t3 = A1*M1
    t4 = M2*t2
    t5 = M3*t2
    t6 = t3+t4+t5
    t7 = A2*M2
    t8 = A2+B2
    t9 = M3*t8
    t10 = t7+t9
    t11 = np.pi*(10/20)
    t12 = theta_dot*t6
    t13 = theta_1_dot*t6
    t14 = theta+theta_1+t11
    t15 = np.cos(t14)
    t16 = t12+t13
    t17 = theta_dot*t10
    t18 = theta_1_dot*t10
    t19 = theta_2_dot*t10
    t20 = theta+theta_1+theta_2+t11
    t21 = np.cos(t20)
    t22 = t17+t18+t19
    t23 = A3*M3*theta_dot
    t24 = A3*M3*theta_1_dot
    t25 = A3*M3*theta_2_dot
    t26 = A3*M3*theta_3_dot
    t27 = theta+theta_1+theta_2+theta_3+t11
    t28 = np.cos(t27)
    t29 = t23+t24+t25+t26
    t30 = B0*M1
    t31 = B0*M2
    t32 = B0*M3
    t33 = t30+t31+t32
    t34 = PHI+theta
    t35 = np.sin(t14)
    t36 = np.sin(t20)
    t37 = np.sin(t27)
    t38 = theta_dot+theta_1_dot+theta_2_dot+theta_3_dot
    t39 = theta+theta_1+theta_2+theta_3
    t40 = A1*B0*M1*theta_1_dot
    t41 = A1*B0*M2*theta_1_dot
    t42 = A1*B0*M3*theta_1_dot
    t43 = B0*B1*M2*theta_1_dot
    t44 = B0*B1*M3*theta_1_dot
    t45 = PHI-theta_1
    t46 = np.cos(t45)
    t47 = A2*B0*M2*theta_1_dot
    t48 = A2*B0*M2*theta_2_dot
    t49 = A2*B0*M3*theta_1_dot
    t50 = A2*B0*M3*theta_2_dot
    t51 = B0*B2*M3*theta_1_dot
    t52 = B0*B2*M3*theta_2_dot
    t53 = -PHI+theta_1+theta_2
    t54 = np.cos(t53)
    t55 = A3*B0*M3*theta_1_dot
    t56 = A3*B0*M3*theta_2_dot
    t57 = A3*B0*M3*theta_3_dot
    t58 = -PHI+theta_1+theta_2+theta_3
    t59 = np.cos(t58)
    t60 = A2*B1*M2*theta_2_dot
    t61 = A1*B2*M3*theta_2_dot
    t62 = A2*B1*M3*theta_2_dot
    t63 = B1*B2*M3*theta_2_dot
    t64 = A1*A2*M2*theta_2_dot
    t65 = A1*A2*M3*theta_2_dot
    t66 = np.sin(theta_2)
    t67 = t60+t61+t62+t63+t64+t65
    t68 = A3*B2*M3*theta_3_dot
    t69 = A2*A3*M3*theta_3_dot
    t70 = np.sin(theta_3)
    t71 = t68+t69
    t72 = A3*B1*M3*theta_2_dot
    t73 = A3*B1*M3*theta_3_dot
    t74 = A1*A3*M3*theta_2_dot
    t75 = A1*A3*M3*theta_3_dot
    t76 = theta_2+theta_3
    t77 = np.sin(t76)
    t78 = t72+t73+t74+t75
    t79 = A2*B0*M2*theta_dot
    t80 = A2*B0*M3*theta_dot
    t81 = B0*B2*M3*theta_dot
    t82 = t47+t48+t49+t50+t51+t52+t79+t80+t81
    t83 = A3*B0*M3*theta_dot
    t84 = t55+t56+t57+t83
    t85 = A1*B0*M1*theta_dot
    t86 = A1*B0*M2*theta_dot
    t87 = A1*B0*M3*theta_dot
    t88 = B0*B1*M2*theta_dot
    t89 = B0*B1*M3*theta_dot
    t90 = A2*B1*M2*theta_dot
    t91 = A1*B2*M3*theta_dot
    t92 = A2*B1*M2*theta_1_dot
    t93 = A2*B1*M3*theta_dot
    t94 = A1*B2*M3*theta_1_dot
    t95 = A2*B1*M3*theta_1_dot
    t96 = B1*B2*M3*theta_dot
    t97 = B1*B2*M3*theta_1_dot
    t98 = A1*A2*M2*theta_dot
    t99 = A1*A2*M2*theta_1_dot
    t100 = A1*A2*M3*theta_dot
    t101 = A1*A2*M3*theta_1_dot
    t102 = t60+t61+t62+t63+t64+t65+t90+t91+t92+t93+t94+t95+t96+t97+t98+t99+t100+t101
    t103 = A3*B1*M3*theta_dot
    t104 = A3*B1*M3*theta_1_dot
    t105 = A1*A3*M3*theta_dot
    t106 = A1*A3*M3*theta_1_dot
    t107 = t72+t73+t74+t75+t103+t104+t105+t106
    t108 = t79+t80+t81
    t109 = t54*t108
    t110 = t59*t83
    t111 = t90+t91+t92+t93+t94+t95+t96+t97+t98+t99+t100+t101
    t112 = t66*t111
    t113 = t103+t104+t105+t106
    t114 = t77*t113
    t115 = A2*theta_dot
    t116 = A2*theta_1_dot
    t117 = A2*theta_2_dot
    t118 = B2*theta_dot
    t119 = B2*theta_1_dot
    t120 = B2*theta_2_dot
    t121 = t115+t116+t117+t118+t119+t120
    t122 = A3*M3*t70*t121
    t123 = A1*theta_dot
    t124 = A1*theta_1_dot
    t125 = B1*theta_dot
    t126 = B1*theta_1_dot
    t127 = t123+t124+t125+t126
    t128 = A3*M3*t77*t127

    # Assembling the matrix
    CoriolisMatrix = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-t15*t16-t21*t22-t28*t29-theta_dot*t33*np.cos(t34),-t16*t35-t22*t36-t29*t37-theta_dot*t33*np.sin(t34),-t66*t67-t70*t71-t77*t78-t46*(t40+t41+t42+t43+t44)-t59*(t55+t56+t57)-t54*(t47+t48+t49+t50+t51+t52),t109+t110-t66*t67-t70*t71-t77*t78+t46*(t85+t86+t87+t88+t89),t109+t110+t112+t114-t70*t71,t110+t122+t128,-t15*t16-t21*t22-t28*t29,-t16*t35-t22*t36-t29*t37,-t66*t67-t54*t82-t70*t71-t59*t84-t77*t78-t46*(t40+t41+t42+t43+t44+t85+t86+t87+t88+t89),-t66*t67-t70*t71-t77*t78,t112+t114-t70*t71,t122+t128,-t21*t22-t28*t29,-t22*t36-t29*t37,-t54*t82-t70*t71-t59*t84-t66*t102-t77*t107,-t70*t71-t66*t102-t77*t107,-A3*M3*theta_3_dot*t8*t70,A3*M3*t8*t70*(theta_dot+theta_1_dot+theta_2_dot),A3*M3*t38*np.sin(t39),-A3*M3*t38*np.cos(t39),-A3*B0*M3*t38*t59-A3*M3*t8*t38*t70-A3*M3*t2*t38*t77,-A3*M3*t8*t38*t70-A3*M3*t2*t38*t77,-A3*M3*t8*t38*t70,00]).reshape([6,6], order='F') # default order is different than matlab

    #control_effort = np.array([0, 0, 0, 0, 0, np.pi/1200])
    second_derivatives = np.matmul(np.linalg.inv(MassMatrix),(control_effort - np.matmul(CoriolisMatrix, state[6:]))) # Should it be state[6:] instead??!? I think so
    #second_derivatives = np.matmul(np.linalg.inv(MassMatrix),(0 - np.matmul(CoriolisMatrix, state[6:]))) # Should it be state[6:] instead??!? I think so

    first_derivatives = np.array([x_dot, y_dot, theta_dot, theta_dot + theta_1_dot, theta_dot + theta_1_dot + theta_2_dot, theta_dot + theta_1_dot + theta_2_dot + theta_3_dot])

    full_derivative = np.concatenate((first_derivatives, second_derivatives))#.squeeze()

    #print(control_effort, full_derivative)

    return full_derivative


##########################################
##### Function to animate the motion #####
##########################################
def render(states, actions, desired_pose, instantaneous_reward_log, cumulative_reward_log, critic_distributions, target_critic_distributions, projected_target_distribution, bins, loss_log, guidance_position_log, episode_number, filename, save_directory):

    # Load in a temporary environment, used to grab the physical parameters
    temp_env = Environment()

    # Checking if we want the additional reward and value distribution information
    extra_information = temp_env.ADDITIONAL_VALUE_INFO

    # Unpacking state
    x, y, theta, theta_1, theta_2, theta_3 = states[:,0], states[:,1], states[:,2], states[:,3], states[:,4], states[:,5]

    # Extracting physical properties
    LENGTH = temp_env.LENGTH
    PHI    = temp_env.PHI
    B0     = temp_env.B0
    A1     = temp_env.A1
    B1     = temp_env.B1
    A2     = temp_env.A2
    B2     = temp_env.B2
    A3     = temp_env.A3
    B3     = temp_env.B3

    # Calculating manipulator joint locations through time. Coordinates are in the inertial frame!
    # Shoulder
    shoulder_x = x + B0*np.cos(theta + PHI)
    shoulder_y = y + B0*np.sin(theta + PHI)

    # Elbow
    elbow_x = shoulder_x + (A1 + B1)*np.cos(np.pi/2 + theta + theta_1)
    elbow_y = shoulder_y + (A1 + B1)*np.sin(np.pi/2 + theta + theta_1)

    # Wrist
    wrist_x = elbow_x + (A2 + B2)*np.cos(np.pi/2 + theta + theta_1 + theta_2)
    wrist_y = elbow_y + (A2 + B2)*np.sin(np.pi/2 + theta + theta_1 + theta_2)

    # End-effector
    end_effector_x = wrist_x + (A3 + B3)*np.cos(np.pi/2 + theta + theta_1 + theta_2 + theta_3)
    end_effector_y = wrist_y + (A3 + B3)*np.sin(np.pi/2 + theta + theta_1 + theta_2 + theta_3)

    # Calculating spacecraft corner locations through time #

    # Corner locations in body frame
    r1_b = LENGTH/2.*np.array([[ 1.], [ 1.]]) # [2, 1]
    r2_b = LENGTH/2.*np.array([[ 1.], [-1.]])
    r3_b = LENGTH/2.*np.array([[-1.], [-1.]])
    r4_b = LENGTH/2.*np.array([[-1.], [ 1.]])

    # Rotation matrix (body -> inertial)
    C_Ib = np.moveaxis(np.array([[np.cos(theta), -np.sin(theta)],
                                 [np.sin(theta),  np.cos(theta)]]), source = 2, destination = 0) # [NUM_TIMESTEPS, 2, 2]

    # Rotating the corner locations to the inertial frame
    r1_I = np.matmul(C_Ib, r1_b) # [NUM_TIMESTEPS, 2, 1]
    r2_I = np.matmul(C_Ib, r2_b) # [NUM_TIMESTEPS, 2, 1]
    r3_I = np.matmul(C_Ib, r3_b) # [NUM_TIMESTEPS, 2, 1]
    r4_I = np.matmul(C_Ib, r4_b) # [NUM_TIMESTEPS, 2, 1]

    # Calculating desired pose #

    # Calculating target angles
    #target_angles = desired_pose[2] + [temp_env.TARGET_ANGULAR_VELOCITY*i*temp_env.TIMESTEP for i in range(len(theta))]

    # Rotation matrix (body -> inertial)
    #C_Ib = np.moveaxis(np.array([[np.cos(target_angles), -np.sin(target_angles)],
    #                 [np.sin(target_angles),  np.cos(target_angles)]]), source = 2, destination = 0) # [NUM_TIMESTEPS, 2, 2]

    # Rotating corner locations to the inertial frame
    #r1_des = np.matmul(C_Ib, r1_b) # [2, 1]
    #r2_des = np.matmul(C_Ib, r2_b) # [2, 1]
    #r3_des = np.matmul(C_Ib, r3_b) # [2, 1]
    #r4_des = np.matmul(C_Ib, r4_b) # [2, 1]

    # Assembling desired pose into lists
    #r_des_x = [r2_des[0], r3_des[0], r4_des[0], r1_des[0]] + desired_pose[0]
    #r_des_y = [r2_des[1], r3_des[1], r4_des[1], r1_des[1]] + desired_pose[1]
    #r_des_front_x = [r1_des[0], r2_des[0]] + desired_pose[0]
    #r_des_front_y = [r1_des[1], r2_des[1]] + desired_pose[1]

    # Table edges
    #table = np.array([[0,0], [3.5, 0], [3.5, 2.41], [0, 2.41], [0, 0]])

    # Generating figure window
    figure = plt.figure(constrained_layout = True)
    figure.set_size_inches(5, 4, True)

    if extra_information:
        grid_spec = gridspec.GridSpec(nrows = 2, ncols = 3, figure = figure)
        subfig1 = figure.add_subplot(grid_spec[0,0], aspect = 'equal', autoscale_on = False, xlim = (0, 3.4), ylim = (0, 2.4))
        subfig2 = figure.add_subplot(grid_spec[0,1], xlim = (np.min([np.min(instantaneous_reward_log), 0]) - (np.max(instantaneous_reward_log) - np.min(instantaneous_reward_log))*0.02, np.max([np.max(instantaneous_reward_log), 0]) + (np.max(instantaneous_reward_log) - np.min(instantaneous_reward_log))*0.02), ylim = (-0.5, 0.5))
        subfig3 = figure.add_subplot(grid_spec[0,2], xlim = (np.min(loss_log)-0.01, np.max(loss_log)+0.01), ylim = (-0.5, 0.5))
        subfig4 = figure.add_subplot(grid_spec[1,0], ylim = (0, 1.02))
        subfig5 = figure.add_subplot(grid_spec[1,1], ylim = (0, 1.02))
        subfig6 = figure.add_subplot(grid_spec[1,2], ylim = (0, 1.02))

        # Setting titles
        subfig1.set_xlabel("X Position (m)",    fontdict = {'fontsize': 8})
        subfig1.set_ylabel("Y Position (m)",    fontdict = {'fontsize': 8})
        subfig2.set_title("Timestep Reward",    fontdict = {'fontsize': 8})
        subfig3.set_title("Current loss",       fontdict = {'fontsize': 8})
        subfig4.set_title("Q-dist",             fontdict = {'fontsize': 8})
        subfig5.set_title("Target Q-dist",      fontdict = {'fontsize': 8})
        subfig6.set_title("Bellman projection", fontdict = {'fontsize': 8})

        # Changing around the axes
        subfig1.tick_params(labelsize = 8)
        subfig2.tick_params(which = 'both', left = False, labelleft = False, labelsize = 8)
        subfig3.tick_params(which = 'both', left = False, labelleft = False, labelsize = 8)
        subfig4.tick_params(which = 'both', left = False, labelleft = False, right = True, labelright = False, labelsize = 8)
        subfig5.tick_params(which = 'both', left = False, labelleft = False, right = True, labelright = False, labelsize = 8)
        subfig6.tick_params(which = 'both', left = False, labelleft = False, right = True, labelright = True, labelsize = 8)

        # Adding the grid
        subfig4.grid(True)
        subfig5.grid(True)
        subfig6.grid(True)

        # Setting appropriate axes ticks
        subfig2.set_xticks([np.min(instantaneous_reward_log), 0, np.max(instantaneous_reward_log)] if np.sign(np.min(instantaneous_reward_log)) != np.sign(np.max(instantaneous_reward_log)) else [np.min(instantaneous_reward_log), np.max(instantaneous_reward_log)])
        subfig3.set_xticks([np.min(loss_log), np.max(loss_log)])
        subfig4.set_xticks([bins[i*5] for i in range(round(len(bins)/5) + 1)])
        subfig4.tick_params(axis = 'x', labelrotation = -90)
        subfig4.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.])
        subfig5.set_xticks([bins[i*5] for i in range(round(len(bins)/5) + 1)])
        subfig5.tick_params(axis = 'x', labelrotation = -90)
        subfig5.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.])
        subfig6.set_xticks([bins[i*5] for i in range(round(len(bins)/5) + 1)])
        subfig6.tick_params(axis = 'x', labelrotation = -90)
        subfig6.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.])

    else:
        subfig1 = figure.add_subplot(1, 1, 1, aspect = 'equal', autoscale_on = False, xlim = (0, 3.4), ylim = (0, 2.4), xlabel = 'X Position (m)', ylabel = 'Y Position (m)')

    # Defining plotting objects that change each frame
    body,        = subfig1.plot([], [], color = 'k', linestyle = '-', linewidth = 2) # Note, the comma is needed
    front_face,  = subfig1.plot([], [], color = 'g', linestyle = '-', linewidth = 2) # Note, the comma is needed
    body_dot     = subfig1.scatter([], [], s = 3, color = 'r')
    target_dot   = subfig1.scatter([], [], s = 2, color = 'b')
    manipulator, = subfig1.plot([], [], color = 'r', linestyle = '-', linewidth = 2) # Note, the comma is needed

    if extra_information:
        reward_bar           = subfig2.barh(y = 0, height = 0.2, width = 0)
        loss_bar             = subfig3.barh(y = 0, height = 0.2, width = 0)
        q_dist_bar           = subfig4.bar(x = bins, height = np.zeros(shape = len(bins)), width = bins[1]-bins[0])
        target_q_dist_bar    = subfig5.bar(x = bins, height = np.zeros(shape = len(bins)), width = bins[1]-bins[0])
        projected_q_dist_bar = subfig6.bar(x = bins, height = np.zeros(shape = len(bins)), width = bins[1]-bins[0])
        time_text            = subfig1.text(x = 0.2, y = 0.91, s = '', fontsize = 8, transform=subfig1.transAxes)
        reward_text          = subfig1.text(x = 0.0,  y = 1.02, s = '', fontsize = 8, transform=subfig1.transAxes)
    else:
        time_text    = subfig1.text(x = 0.03, y = 0.96, s = '', fontsize = 8, transform=subfig1.transAxes)
        reward_text  = subfig1.text(x = 0.62, y = 0.96, s = '', fontsize = 8, transform=subfig1.transAxes)
        episode_text = subfig1.text(x = 0.40, y = 1.02, s = '', fontsize = 8, transform=subfig1.transAxes)

    # Plotting constant items once (background, etc)
    #table,             = subfig1.plot(table[:,0], table[:,1],       color = 'k', linestyle = '-', linewidth = 3)

    # Function called once to initialize axes as empty
    def initialize_axes():
        body.set_data([], [])
        front_face.set_data([], [])
        time_text.set_text('')

        if not extra_information:
            episode_text.set_text('Episode ' + str(episode_number))

        return body, front_face, time_text

    # Function called repeatedly to draw each frame
    def render_one_frame(frame, *fargs):
        temp_env = fargs[0] # Extract environment from passed args

        # Draw the spacecraft body
        thisx = [r2_I[frame,0,0], r3_I[frame,0,0], r4_I[frame,0,0], r1_I[frame,0,0]] + x[frame]
        thisy = [r2_I[frame,1,0], r3_I[frame,1,0], r4_I[frame,1,0], r1_I[frame,1,0]] + y[frame]
        body.set_data(thisx, thisy)

        # Draw the front face of the spacecraft body in a different colour
        thisx = [r1_I[frame,0,0], r2_I[frame,0,0]] + x[frame]
        thisy = [r1_I[frame,1,0], r2_I[frame,1,0]] + y[frame]
        front_face.set_data(thisx, thisy)

        # Draw the manipulator
        thisx = [shoulder_x[frame], elbow_x[frame], wrist_x[frame], end_effector_x[frame]]
        thisy = [shoulder_y[frame], elbow_y[frame], wrist_y[frame], end_effector_y[frame]]
        manipulator.set_data(thisx, thisy)

        body_dot.set_offsets(np.hstack((x[frame], y[frame])))

        target_dot.set_offsets(np.hstack((desired_pose[0], desired_pose[1])))

        if frame != 0:
            subfig1.patches.clear() # remove the last frame's arrow

        # Update the time text
        time_text.set_text('Time = %.1f s' %(frame*temp_env.TIMESTEP))

        # Update the reward text
        reward_text.set_text('Total reward = %.1f' %cumulative_reward_log[frame])

        if extra_information:
            # Updating the instantaneous reward bar graph
            reward_bar[0].set_width(instantaneous_reward_log[frame])
            # And colouring it appropriately
            if instantaneous_reward_log[frame] < 0:
                reward_bar[0].set_color('r')
            else:
                reward_bar[0].set_color('g')

            # Updating the loss bar graph
            loss_bar[0].set_width(loss_log[frame])

            # Updating the q-distribution plot
            for this_bar, new_value in zip(q_dist_bar, critic_distributions[frame,:]):
                this_bar.set_height(new_value)

            # Updating the target q-distribution plot
            for this_bar, new_value in zip(target_q_dist_bar, target_critic_distributions[frame, :]):
                this_bar.set_height(new_value)

            # Updating the projected target q-distribution plot
            for this_bar, new_value in zip(projected_q_dist_bar, projected_target_distribution[frame, :]):
                this_bar.set_height(new_value)

        # If dynamics are present, draw an arrow showing the location of the guided position
        if temp_env.TEST_ON_DYNAMICS:

            position_arrow = plt.Arrow(x[frame], y[frame], guidance_position_log[frame,0] - x[frame], guidance_position_log[frame,1] - y[frame], width = 0.06, color = 'k')

#            # Adding the rotational arrow
#            angle_error = guidance_position_log[-1,2] - theta[frame]
#            style="Simple,tail_width=0.5,head_width=4,head_length=8"
#            kw = dict(arrowstyle=style, color="k")
#            start_point = np.array((x[frame] + temp_env.LENGTH, y[frame]))
#            end_point   = np.array((x[frame] + temp_env.LENGTH*np.cos(angle_error), y[frame] + temp_env.LENGTH*np.sin(angle_error)))
#            half_chord = np.linalg.norm(end_point - start_point)/2
#            mid_dist = temp_env.LENGTH - (temp_env.LENGTH**2 - half_chord**2)**0.5
#            ratio = mid_dist/(half_chord)
#            ratio = 1
#            rotation_arrow = patches.FancyArrowPatch((x[frame] + temp_env.LENGTH, y[frame]), (x[frame] + temp_env.LENGTH*np.cos(angle_error), y[frame] + temp_env.LENGTH*np.sin(angle_error)), connectionstyle = "arc3,rad=" + str(ratio), **kw)
            subfig1.add_patch(position_arrow)
#            subfig1.add_patch(rotation_arrow)

        # Since blit = True, must return everything that has changed at this frame
        return body, front_face, time_text, body_dot, manipulator, target_dot

    # Generate the animation!
    fargs = [temp_env] # bundling additional arguments
    animator = animation.FuncAnimation(figure, render_one_frame, frames = np.linspace(0, len(states)-1, len(states)).astype(int),
                                       blit = True, init_func = initialize_axes, fargs = fargs)
    """
    frames = the int that is passed to render_one_frame. I use it to selectively plot certain data
    fargs = additional arguments for render_one_frame
    interval = delay between frames in ms
    """

    # Save the animation!
    try:
        # Save it to the working directory [have to], then move it to the proper folder
        animator.save(filename = filename + '_episode_' + str(episode_number) + '.mp4', fps = 30, dpi = 100)
        # Make directory if it doesn't already exist
        os.makedirs(os.path.dirname(save_directory + filename + '/videos/'), exist_ok=True)
        # Move animation to the proper directory
        os.rename(filename + '_episode_' + str(episode_number) + '.mp4', save_directory + filename + '/videos/episode_' + str(episode_number) + '.mp4')
    except:
        print("Skipping animation for episode %i due to an error" %episode_number)
        # Try to delete the partially completed video file
        try:
            os.remove(filename + '_episode_' + str(episode_number) + '.mp4')
        except:
            pass

    del temp_env
    plt.close(figure)