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
            - penalty for relatlive angular velocity of the end-effector during docking
        - A penalty for colliding with the target
        Once learned, some optional rewards can be applied to see how it affects the motion:
            - In the future, a penalty for attitude disturbance on the chaser base attitude??? Or a penalty to all accelerations??
            - Extend the forbidden area into a larger cone to force the approach to be more cone-shaped

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
        """
        ==The State==
        what states do I need? Relative is good, but I think I need some absolute especially for the part where I'll bring the assembly to a desired location. 
        I'll use absolute since I'll be getting my data from PhaseSpace.
        I've got the manipulator angles along with the end-effector position. It is redundant to have the position if I've got the angles but maybe it'll help??
        
        The positions are in inertial frame but the manipulator angles are in the joint frame.
        
        """
        self.ON_COMPUTE_CANADA        = True
        self.TOTAL_STATE_SIZE         = 22 # [chaser_x, chaser_y, chaser_theta, chaser_x_dot, chaser_y_dot, chaser_theta_dot, shoulder_theta, elbow_theta, wrist_theta, shoulder_theta_dot, elbow_theta_dot, wrist_theta_dot, target_x, target_y, target_theta, target_x_dot, target_y_dot, target_theta_dot, ee_x, ee_y, ee_x_dot, ee_y_dot]
        ### Note: TOTAL_STATE contains all relevant information describing the problem, and all the information needed to animate the motion
        #         TOTAL_STATE is returned from the environment to the agent.
        #         A subset of the TOTAL_STATE, called the 'observation', is passed to the policy network to calculate acitons. This takes place in the agent
        #         The TOTAL_STATE is passed to the animator below to animate the motion.
        #         The chaser and target state are contained in the environment. They are packaged up before being returned to the agent.
        #         The total state information returned must be as commented beside self.TOTAL_STATE_SIZE.
        self.IRRELEVANT_STATES                = [18,19,20,21] # [end-effector states] indices of states who are irrelevant to the policy network
        self.OBSERVATION_SIZE                 = self.TOTAL_STATE_SIZE - len(self.IRRELEVANT_STATES) # the size of the observation input to the policy
        self.ACTION_SIZE                      = 6 # [x_dot_dot, y_dot_dot, theta_dot_dot, shoulder_theta_dot_dot, elbow_theta_dot_dot, wrist_theta_dot_dot] in the inertial frame (for x and y), in the joint frame for the others.
        self.MAX_X_POSITION                   = 3.5 # [m]
        self.MAX_Y_POSITION                   = 2.4 # [m]
        self.MAX_VELOCITY                     = 0.5 # [m/s]
        self.MAX_ANGULAR_VELOCITY             = np.pi/6 # [rad/s] for joints or body
        self.MAX_LINEAR_ACCELERATION          = 0.025 # [m/s^2]
        self.MAX_ANGULAR_ACCELERATION         = 0.1 # [rad/s^2]
        self.MAX_THRUST                       = 0.5 # [N] Experimental limitation
        self.MAX_BODY_TORQUE                  = 0.064 # [Nm] # Experimental limitation
        self.MAX_JOINT1n2_TORQUE              = 0.02 # [Nm] # Limited by the simulator NOT EXPERIMENT
        self.MAX_JOINT3_TORQUE                = 0.0002 # [Nm] Limited by the simulator NOT EXPERIMENT
        self.LOWER_ACTION_BOUND               = np.array([-self.MAX_LINEAR_ACCELERATION, -self.MAX_LINEAR_ACCELERATION, -self.MAX_ANGULAR_ACCELERATION, -self.MAX_ANGULAR_ACCELERATION, -self.MAX_ANGULAR_ACCELERATION, -self.MAX_ANGULAR_ACCELERATION]) # [m/s^2, m/s^2, rad/s^2, rad/s^2, rad/s^2, rad/s^2]
        self.UPPER_ACTION_BOUND               = np.array([ self.MAX_LINEAR_ACCELERATION,  self.MAX_LINEAR_ACCELERATION,  self.MAX_ANGULAR_ACCELERATION,  self.MAX_ANGULAR_ACCELERATION,  self.MAX_ANGULAR_ACCELERATION,  self.MAX_ANGULAR_ACCELERATION]) # [m/s^2, m/s^2, rad/s^2, rad/s^2, rad/s^2, rad/s^2]
                
        self.LOWER_STATE_BOUND                = np.array([ 0.0, 0.0, -6*np.pi, -self.MAX_VELOCITY, -self.MAX_VELOCITY, -self.MAX_ANGULAR_VELOCITY,  # Chaser 
                                                          -np.pi/2, -np.pi/2, -np.pi/2, # Shoulder_theta, Elbow_theta, Wrist_theta
                                                          -self.MAX_ANGULAR_VELOCITY, -self.MAX_ANGULAR_VELOCITY, -self.MAX_ANGULAR_VELOCITY, # Shoulder_theta_dot, Elbow_theta_dot, Wrist_theta_dot
                                                          0.0, 0.0, -6*np.pi, -self.MAX_VELOCITY, -self.MAX_VELOCITY, -self.MAX_ANGULAR_VELOCITY, # Target
                                                          0.0, 0.0, -3*self.MAX_VELOCITY, -3*self.MAX_VELOCITY]) # End-effector
                                                          # [m, m, rad, m/s, m/s, rad/s, rad, rad, rad, rad/s, rad/s, rad/s, m, m, rad, m/s, m/s, rad/s, m, m, m/s, m/s] // lower bound for each element of TOTAL_STATE
        self.UPPER_STATE_BOUND                = np.array([ self.MAX_X_POSITION, self.MAX_Y_POSITION, 6*np.pi, self.MAX_VELOCITY, self.MAX_VELOCITY, self.MAX_ANGULAR_VELOCITY,  # Chaser 
                                                          np.pi/2, np.pi/2, np.pi/2, # Shoulder_theta, Elbow_theta, Wrist_theta
                                                          self.MAX_ANGULAR_VELOCITY, self.MAX_ANGULAR_VELOCITY, self.MAX_ANGULAR_VELOCITY, # Shoulder_theta_dot, Elbow_theta_dot, Wrist_theta_dot
                                                          self.MAX_X_POSITION, self.MAX_Y_POSITION, 6*np.pi, self.MAX_VELOCITY, self.MAX_VELOCITY, self.MAX_ANGULAR_VELOCITY, # Target
                                                          self.MAX_X_POSITION, self.MAX_Y_POSITION, 3*self.MAX_VELOCITY, 3*self.MAX_VELOCITY]) # End-effector
                                                          # [m, m, rad, m/s, m/s, rad/s, rad, rad, rad, rad/s, rad/s, rad/s, m, m, rad, m/s, m/s, rad/s, m, m, m/s, m/s] // Upper bound for each element of TOTAL_STATE
        self.INITIAL_CHASER_POSITION          = np.array([self.MAX_X_POSITION/3, self.MAX_Y_POSITION/2, 0.0]) # [m, m, rad]
        self.INITIAL_CHASER_VELOCITY          = np.array([0.0,  0.0, 0.0]) # [m/s, m/s, rad/s]
        self.INITIAL_ARM_ANGLES               = np.array([0.0,  0.0, 0.0]) # [rad, rad, rad]
        self.INITIAL_ARM_RATES                = np.array([0.0,  0.0, 0.0]) # [rad/s, rad/s, rad/s]
        self.INITIAL_TARGET_POSITION          = np.array([self.MAX_X_POSITION*2/3, self.MAX_Y_POSITION/2, 0.0]) # [m, m, rad]
        self.INITIAL_TARGET_VELOCITY          = np.array([0.0,  0.0, 0.0]) # [m/s, m/s, rad/s]
        self.NORMALIZE_STATE                  = True # Normalize state on each timestep to avoid vanishing gradients
        self.RANDOMIZE_INITIAL_CONDITIONS     = True # whether or not to randomize the initial conditions
        self.RANDOMIZE_DOMAIN                 = False # whether or not to randomize the physical parameters (length, mass, size)
        self.RANDOMIZATION_POSITION           = 0.5 # [m] half-range uniform randomization position
        self.RANDOMIZATION_CHASER_VELOCITY    = 0.0 # [m/s] half-range uniform randomization chaser velocity
        self.RANDOMIZATION_CHASER_OMEGA       = 0.0 # [rad/s] half-range uniform randomization chaser omega
        self.RANDOMIZATION_ANGLE              = np.pi # [rad] half-range uniform randomization chaser and target base angle
        self.RANDOMIZATION_ARM_ANGLE          = np.pi/2 # [rad] half-range uniform randomization arm angle
        self.RANDOMIZATION_ARM_RATES          = 0.0 # [rad/s] half-range uniform randomization arm rates
        self.RANDOMIZATION_TARGET_VELOCITY    = 0.0 # [m/s] half-range uniform randomization target velocity
        self.RANDOMIZATION_TARGET_OMEGA       = 2*np.pi/30 # [rad/s] half-range uniform randomization target omega
        self.MIN_V                            = -100.
        self.MAX_V                            =  125.
        self.N_STEP_RETURN                    =   5
        self.DISCOUNT_FACTOR                  = 0.95**(1/self.N_STEP_RETURN)
        self.TIMESTEP                         = 0.2 # [s]
        self.CALIBRATE_TIMESTEP               = False # Forces a predetermined action and prints more information to the screen. Useful in calculating gains and torque limits
        self.CLIP_DURING_CALIBRATION          = True # Whether or not to clip the control forces during calibration
        self.PREDETERMINED_ACTION             = np.array([0,0,0,0,0,0])
        self.DYNAMICS_DELAY                   = 0 # [timesteps of delay] how many timesteps between when an action is commanded and when it is realized
        self.AUGMENT_STATE_WITH_ACTION_LENGTH = 0 # [timesteps] how many timesteps of previous actions should be included in the state. This helps with making good decisions among delayed dynamics.
        self.MAX_NUMBER_OF_TIMESTEPS          = 150# per episode
        self.ADDITIONAL_VALUE_INFO            = False # whether or not to include additional reward and value distribution information on the animations
        self.SKIP_FAILED_ANIMATIONS           = True # Error the program or skip when animations fail?
        self.KI                               = [10,10,0.15,0.012,0.003,0.000044] # Returned [10,10,0.15,0.012,0.003,0.000044] Dec 19 for 0.2s timestep #[10,10,0.15, 0.018,0.0075,0.000044] # [Tuned Dec 19 for 0.058s timestep] Integral gain for the integral-acceleration controller of the body and arm (x, y, theta, theta1, theta2, theta3)
                                
        # Physical properties (See Fig. 3.1 in Alex Cran's MASc Thesis for definitions)
        self.LENGTH   = 0.3 # [m] side length
        self.PHI      = 68.2840*np.pi/180#np.pi/2 # [rad] angle of anchor point of arm with respect to spacecraft body frame
        self.B0       = 0.2304#(self.LENGTH/2)/np.cos(np.pi/2-self.PHI) # scalar distance from centre of mass to arm attachment point
        self.MASS     = 16.9478#10.0  # [kg] for chaser
        self.M1       = 0.3377 # [kg] link mass
        self.M2       = 0.3281 # [kg] link mass
        self.M3       = 0.0111 # [kg] link mass
        self.A1       = 0.1933 # [m] base of link to centre of mass
        self.B1       = 0.1117 # [m] centre of mass to end of link
        self.A2       = 0.1993 # [m] base of link to centre of mass
        self.B2       = 0.1057 # [m] centre of mass to end of link
        self.A3       = 0.0621 # [m] base of link to centre of mass
        self.B3       = 0.0159 # [m] centre of mass to end of link
        self.INERTIA  = 1/12*self.MASS*(self.LENGTH**2 + self.LENGTH**2) # 0.15 [kg m^2] base inertia
        self.INERTIA1 = 1/12*self.M1*(self.A1 + self.B1)**2 # [kg m^2] link inertia
        self.INERTIA2 = 1/12*self.M2*(self.A2 + self.B2)**2 # [kg m^2] link inertia
        self.INERTIA3 = 1/12*self.M3*(self.A3 + self.B3)**2 # [kg m^2] link inertia        
        
        # Platform physical properties        
        self.LENGTH_RANDOMIZATION          = 0.1 # [m] standard deviation of the LENGTH randomization when domain randomization is performed.        
        self.MASS_RANDOMIZATION            = 1.0 # [kg] standard deviation of the MASS randomization when domain randomization is performed.
        self.DOCKING_PORT_MOUNT_POSITION   = np.array([0, self.LENGTH/2]) # position of the docking cone on the target in its body frame
        self.DOCKING_PORT_CORNER1_POSITION = self.DOCKING_PORT_MOUNT_POSITION + [ 0.05, 0.1] # position of the docking cone on the target in its body frame
        self.DOCKING_PORT_CORNER2_POSITION = self.DOCKING_PORT_MOUNT_POSITION + [-0.05, 0.1] # position of the docking cone on the target in its body frame
        
        # Reward function properties
        self.DOCKING_REWARD                   = 100 # A lump-sum given to the chaser when it docks
        self.SUCCESSFUL_DOCKING_RADIUS        = 0.04 # [m] distance at which the magnetic docking can occur
        self.MAX_DOCKING_ANGLE_PENALTY        = 50 # A penalty given to the chaser, upon docking, for having an angle when docking. The penalty is 0 upon perfect docking and MAX_DOCKING_ANGLE_PENALTY upon perfectly bad docking
        self.DOCKING_EE_VELOCITY_PENALTY      = 50 # A penalty given to the chaser, upon docking, for every 1 m/s end-effector collision velocity upon docking
        self.DOCKING_ANGULAR_VELOCITY_PENALTY = 25 # A penalty given to the chaser, upon docking, for every 1 rad/s angular body velocity upon docking
        self.END_ON_FALL                      = True # end episode on a fall off the table        
        self.FALL_OFF_TABLE_PENALTY           = 100.
        self.CHECK_CHASER_TARGET_COLLISION    = True
        self.TARGET_COLLISION_PENALTY         = 5 # [rewards/timestep] penalty given for colliding with target  
        self.CHECK_END_EFFECTOR_COLLISION     = True # Whether to do collision detection on the end-effector
        self.CHECK_END_EFFECTOR_FORBIDDEN     = True # Whether to expand the collision area to include the forbidden zone
        self.END_EFFECTOR_COLLISION_PENALTY   = 5 # [rewards/timestep] Penalty for end-effector collisions (with target or optionally with the forbidden zone)
        self.END_ON_COLLISION                 = False # Whether to end the episode upon a collision.
        self.GIVE_MID_WAY_REWARD              = True # Whether or not to give a reward mid-way towards the docking port to encourage the learning to move in the proper direction
        self.MID_WAY_REWARD_RADIUS            = 0.3 # [ms] the radius from the DOCKING_PORT_MOUNT_POSITION that the mid-way reward is given
        self.MID_WAY_REWARD                   = 25 # The value of the mid-way reward
        
        
        # Some calculations that don't need to be changed
        self.TABLE_BOUNDARY    = Polygon(np.array([[0,0], [self.MAX_X_POSITION, 0], [self.MAX_X_POSITION, self.MAX_Y_POSITION], [0, self.MAX_Y_POSITION], [0,0]]))
        self.VELOCITY_LIMIT    = np.array([self.MAX_VELOCITY, self.MAX_VELOCITY, self.MAX_ANGULAR_VELOCITY, 2*self.MAX_ANGULAR_VELOCITY, 3*self.MAX_ANGULAR_VELOCITY, 4*self.MAX_ANGULAR_VELOCITY]) # [m/s, m/s, rad/s] maximum allowable velocity/angular velocity; enforced by the controller
        self.ANGLE_LIMIT       = 1*np.pi/2 # Used as a hard limit in the dynamics in order to protect the arm from hitting the chaser
        self.LOWER_STATE_BOUND = np.concatenate([self.LOWER_STATE_BOUND, np.tile(self.LOWER_ACTION_BOUND, self.AUGMENT_STATE_WITH_ACTION_LENGTH)]) # lower bound for each element of TOTAL_STATE
        self.UPPER_STATE_BOUND = np.concatenate([self.UPPER_STATE_BOUND, np.tile(self.UPPER_ACTION_BOUND, self.AUGMENT_STATE_WITH_ACTION_LENGTH)]) # upper bound for each element of TOTAL_STATE        
        self.OBSERVATION_SIZE  = self.TOTAL_STATE_SIZE - len(self.IRRELEVANT_STATES) # the size of the observation input to the policy


    ###################################
    ##### Seeding the environment #####
    ###################################
    def seed(self, seed):
        np.random.seed(seed)


    ######################################
    ##### Resettings the Environment #####
    ######################################
    def reset(self, test_time):
        # This method resets the state
        """ NOTES:
               - if test_time = True -> do not add "controller noise" to the kinematics
        """
        
        # Resetting the time
        self.time = 0.

        # Logging whether it is test time for this episode
        self.test_time = test_time
        
        # Resetting the mid-way flag
        self.not_yet_mid_way = True

        # If we are randomizing the initial conditions and state
        if self.RANDOMIZE_INITIAL_CONDITIONS:
            # Randomizing initial state in Inertial frame
            self.chaser_position = self.INITIAL_CHASER_POSITION + np.random.uniform(low = -1, high = 1, size = 3)*[self.RANDOMIZATION_POSITION, self.RANDOMIZATION_POSITION, self.RANDOMIZATION_ANGLE]
            # Randomizing initial claser velocity in Inertial Frame
            self.chaser_velocity = self.INITIAL_CHASER_VELOCITY + np.random.uniform(low = -1, high = 1, size = 3)*[self.RANDOMIZATION_CHASER_VELOCITY, self.RANDOMIZATION_CHASER_VELOCITY, self.RANDOMIZATION_CHASER_OMEGA]
            # Randomizing target state in Inertial frame
            self.target_position = self.INITIAL_TARGET_POSITION + np.random.uniform(low = -1, high = 1, size = 3)*[self.RANDOMIZATION_POSITION, self.RANDOMIZATION_POSITION, self.RANDOMIZATION_ANGLE]
            # Randomizing target velocity in Inertial frame
            self.target_velocity = self.INITIAL_TARGET_VELOCITY + np.random.uniform(low = -1, high = 1, size = 3)*[self.RANDOMIZATION_TARGET_VELOCITY, self.RANDOMIZATION_TARGET_VELOCITY, self.RANDOMIZATION_TARGET_OMEGA]
            # Randomizing arm angles in Body frame
            self.arm_angles = self.INITIAL_ARM_ANGLES + np.random.uniform(low = -1, high = 1, size = 3)*[self.RANDOMIZATION_ARM_ANGLE, self.RANDOMIZATION_ARM_ANGLE, self.RANDOMIZATION_ARM_ANGLE]
            # Randomizing arm angular rates in body frame
            self.arm_angular_rates = self.INITIAL_ARM_RATES + np.random.uniform(low = -1, high = 1, size = 3)*[self.RANDOMIZATION_ARM_RATES, self.RANDOMIZATION_ARM_RATES, self.RANDOMIZATION_ARM_RATES]
            

        else:
            # Constant initial state in Inertial frame
            self.chaser_position = self.INITIAL_CHASER_POSITION
            # Constant chaser velocity in Inertial frame
            self.chaser_velocity = self.INITIAL_CHASER_VELOCITY
            # Constant target location in Inertial frame
            self.target_position = self.INITIAL_TARGET_POSITION
            # Constant target velocity in Inertial frame
            self.target_velocity = self.INITIAL_TARGET_VELOCITY
            # Constant initial arm position in Body frame
            self.arm_angles = self.INITIAL_ARM_ANGLES
            # Constand arm angular velocity in Body frame
            self.arm_angular_rates = self.INITIAL_ARM_RATES
        
        # TODO: Build domain randomization
        
        # Update docking component locations
        self.update_end_effector_and_docking_locations()
        
        # Check for collisions
        self.check_collisions()

        # Initializing the previous velocity and control effort for the integral-acceleration controller
        self.previous_velocity       = np.zeros(self.ACTION_SIZE)
        self.previous_control_effort = np.zeros(self.ACTION_SIZE)
        
        # Initializing integral anti-wind-up that checks if the joints angles have been reached
        self.joints_past_limits = [False, False, False]

        # Resetting the action delay queue
        if self.DYNAMICS_DELAY > 0:
            self.action_delay_queue = queue.Queue(maxsize = self.DYNAMICS_DELAY + 1)
            for i in range(self.DYNAMICS_DELAY):
                self.action_delay_queue.put(np.zeros(self.ACTION_SIZE), False)
                

    def update_end_effector_and_docking_locations(self):
        """
        This method returns the location of the end-effector of the manipulator
        based off the current state in the Inertial frame
        
        It also updates the docking port position on the target
        """
        ##########################
        ## End-effector Section ##
        ##########################
        # Unpacking the state
        x, y, theta                           = self.chaser_position
        x_dot, y_dot, theta_dot               = self.chaser_velocity
        theta_1, theta_2, theta_3             = self.arm_angles
        theta_1_dot, theta_2_dot, theta_3_dot = self.arm_angular_rates

        x_ee = x + self.B0*np.cos(self.PHI + theta) + (self.A1 + self.B1)*np.cos(np.pi/2 + theta + theta_1) + \
               (self.A2 + self.B2)*np.cos(np.pi/2 + theta + theta_1 + theta_2) + \
               (self.A3 + self.B3)*np.cos(np.pi/2 + theta + theta_1 + theta_2 + theta_3)

        x_ee_dot = x_dot - self.B0*np.sin(self.PHI + theta)*(theta_dot) - (self.A1 + self.B1)*np.sin(np.pi/2 + theta + theta_1)*(theta_dot + theta_1_dot) - \
                           (self.A2 + self.B2)*np.sin(np.pi/2 + theta + theta_1 + theta_2)*(theta_dot + theta_1_dot + theta_2_dot) - \
                           (self.A3 + self.B3)*np.sin(np.pi/2 + theta + theta_1 + theta_2 + theta_3)*(theta_dot + theta_1_dot + theta_2_dot + theta_3_dot)
                           
        y_ee = y + self.B0*np.sin(self.PHI + theta) + (self.A1 + self.B1)*np.sin(np.pi/2 + theta + theta_1) + \
               (self.A2 + self.B2)*np.sin(np.pi/2 + theta + theta_1 + theta_2) + \
               (self.A3 + self.B3)*np.sin(np.pi/2 + theta + theta_1 + theta_2 + theta_3)
        
        y_ee_dot = y_dot + self.B0*np.cos(self.PHI + theta)*(theta_dot) + (self.A1 + self.B1)*np.cos(np.pi/2 + theta + theta_1)*(theta_dot + theta_1_dot) + \
                           (self.A2 + self.B2)*np.cos(np.pi/2 + theta + theta_1 + theta_2)*(theta_dot + theta_1_dot + theta_2_dot) + \
                           (self.A3 + self.B3)*np.cos(np.pi/2 + theta + theta_1 + theta_2 + theta_3)*(theta_dot + theta_1_dot + theta_2_dot + theta_3_dot)

        # Updates the position of the end-effector in the Inertial frame
        self.end_effector_position = np.array([x_ee, y_ee])
        
        # End effector velocity
        self.end_effector_velocity = np.array([x_ee_dot, y_ee_dot]) 
        
        ###################
        ## Elbow Section ##
        ###################
        x_elbow = x + self.B0*np.cos(self.PHI + theta) + (self.A1 + self.B1)*np.cos(np.pi/2 + theta + theta_1)                  
        y_elbow = y + self.B0*np.sin(self.PHI + theta) + (self.A1 + self.B1)*np.sin(np.pi/2 + theta + theta_1)
                  
        self.elbow_position = np.array([x_elbow, y_elbow])        
        
        ##########################
        ## Docking port Section ##
        ##########################
        # Make rotation matrix
        C_Ib_target = self.make_C_bI(self.target_position[-1]).T
        
        # Position in Inertial = Body position (inertial) + C_Ib * EE position in body
        self.docking_port_position = self.target_position[:-1] + np.matmul(C_Ib_target, self.DOCKING_PORT_MOUNT_POSITION)
        
        # Velocity in Inertial = target_velocity + omega_target [cross] r_{port/G}
        self.docking_port_velocity = self.target_velocity[:-1] + self.target_velocity[-1] * np.matmul(self.make_C_bI(self.target_position[-1]).T,[-self.DOCKING_PORT_MOUNT_POSITION[1], self.DOCKING_PORT_MOUNT_POSITION[0]])

    def make_total_state(self):
        
        # Assembles all the data into the shape of TOTAL STATE so that it is consistent
        # chaser_x, chaser_y, chaser_theta, chaser_x_dot, chaser_y_dot, chaser_theta_dot, 
        # shoulder_theta, elbow_theta, wrist_theta, shoulder_theta_dot, elbow_theta_dot, wrist_theta_dot, 
        # target_x, target_y, target_theta, target_x_dot, target_y_dot, target_theta_dot, 
        # ee_x, ee_y, ee_x_dot, ee_y_dot]

        total_state = np.concatenate([self.chaser_position, self.chaser_velocity, self.arm_angles, self.arm_angular_rates, self.target_position, self.target_velocity, self.end_effector_position, self.end_effector_velocity])
        
        return total_state
    
    def make_chaser_state(self):
        
        # Assembles all chaser-relevant data into a state to be fed to the equations of motion
        
        total_chaser_state = np.concatenate([self.chaser_position, self.arm_angles, self.chaser_velocity, self.arm_angular_rates])
        
        return total_chaser_state
    #####################################
    ##### Step the Dynamics forward #####
    #####################################
    def step(self, action):

        # Integrating forward one time step using the calculated action.
        # Oeint returns initial condition on first row then next TIMESTEP on the next row

        ############################
        #### PROPAGATE DYNAMICS ####
        ############################

        # First, calculate the control effort
        control_effort = self.controller(action)

        # Anything that needs to be sent to the dynamics integrator
        dynamics_parameters = [control_effort, self.LENGTH, self.PHI, self.B0, self.MASS, self.M1, self.M2, self.M3, self.A1, self.B1, self.A2, self.B2, self.A3, self.B3, self.INERTIA, self.INERTIA1, self.INERTIA2, self.INERTIA3]
        
        # Building the state
        current_chaser_state = self.make_chaser_state()
        
        # Propagate the dynamics forward one timestep
        next_states = odeint(dynamics_equations_of_motion, current_chaser_state, [self.time, self.time + self.TIMESTEP], args = (dynamics_parameters,), full_output = 0)

        # Saving the new state
        new_chaser_state = next_states[1,:]
        
        # The inverse of make_chaser_state()
        self.chaser_position = new_chaser_state[0:3]
        self.arm_angles = new_chaser_state[3:6]
        self.chaser_velocity = new_chaser_state[6:9]
        self.arm_angular_rates = new_chaser_state[9:12]
        
        # Setting a hard limit on the manipulator angles
        #TODO: Investigate momentum transfer when limits are hit. It seems like I have to do 
        #      this either through conservation of momentum or a collision force?
        self.joints_past_limits = np.abs(self.arm_angles) > self.ANGLE_LIMIT
        if np.any(self.joints_past_limits):
            # Hold the angle at the limit
            self.arm_angles[self.joints_past_limits] = np.sign(self.arm_angles[self.joints_past_limits]) * self.ANGLE_LIMIT
            # Set the angular rate to zero
            self.arm_angular_rates[self.joints_past_limits] = 0
            # Set the past control effort to 0 to prevent further wind-up
            # Removed because wind-up was totally removed. Instead, stop it from increasing in the controller 
            #self.previous_control_effort[3:][self.joints_past_limits] = 0            

        # Step target's state ahead one timestep
        self.target_position += self.target_velocity * self.TIMESTEP
        
        # Update docking locations
        self.update_end_effector_and_docking_locations()
        
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
        if self.CALIBRATE_TIMESTEP:
            desired_accelerations = self.PREDETERMINED_ACTION
        
        # Stopping the command of additional velocity when we are already at our maximum
        current_velocity = np.concatenate([self.chaser_velocity, self.arm_angular_rates])        
        if not self.CALIBRATE_TIMESTEP:
            desired_accelerations[(np.abs(current_velocity) > self.VELOCITY_LIMIT) & (np.sign(desired_accelerations) == np.sign(current_velocity))] = 0
        
        # Approximating the current accelerations
        current_accelerations = (current_velocity - self.previous_velocity)/self.TIMESTEP
        self.previous_velocity = current_velocity
        
        """ This integral-acceleration transpose jacobian controller is implemented but not used """
#            #######################################################################
#            ### Integral-acceleration transpose Jacobian controller for the arm ###
#            #######################################################################
#            """
#            Using the jacobian, J, for joint 3 as described in Eq 3.31 of Alex Crain's MASc Thesis
#            v = velocity of end-effector in inertial frame
#            w = angular velocity of end-effector about its centre of mass
#            q = [chaser_x, chaser_y, theta, theta_1, theta_2, theta_3]
#            [v,w] = J*qdot
#            Therefore, using the transpose Jacobian trick
#            forces_torques = J.T * F_ee
#            where F_ee is [ee_f_x, ee_f_y, ee_tau_z]
#            and forces_torques is [body_Fx, body_Fy, body_tau, tau1, tau2, tau3]
#            J = self.make_jacobian()
#            """
#    
#            current_ee_velocity = np.array([self.end_effector_velocity[0], self.end_effector_velocity[1], np.sum((self.arm_angular_rates)) + self.chaser_velocity[-1]])
#            current_ee_pose_acceleration = (current_ee_velocity - self.previous_ee_pose_velocity)/self.TIMESTEP
#            
#            # Calculating the end-effector acceleration error
#            ee_acceleration_error = desired_ee_accelerations - current_ee_pose_acceleration
#            
#            # End-effector control effort
#            ee_control_effort = self.previous_ee_control_effort + self.KI[3:] * ee_acceleration_error
#            
#            # Saving the current velocity and control effort for the next timetsep
#            self.previous_ee_pose_velocity = current_ee_pose_velocity
#            self.previous_ee_control_effort = ee_control_effort
#            
#            # Using the Transpose Jacobian
#            joint_space_torque = np.matmul(self.make_jacobian().T, ee_control_effort)
#    
#            # Assuming the body control effort was calculated previously, concatenate/add it here
#            control_effort = np.concatenate([body_control_effort, joint_space_torque[3:]]).reshape([6,1])
#            
  
        ##########################################################
        ### Integral-acceleration controller on the all states ###
        ########################################################## 
        
        # Calculate the acceleration error
        acceleration_error = desired_accelerations - current_accelerations
        
        # If the joint is currently at its limit and the desired acceleration is worsening the problem, set the acceleration error to 0. This will prevent further integral wind-up but not release the current wind-up.
        acceleration_errors_to_zero = (self.joints_past_limits) & (np.sign(desired_accelerations[3:]) == np.sign(self.arm_angles))
        acceleration_error[3:][acceleration_errors_to_zero] = 0
                
        # Apply the integral controller 
        control_effort = self.previous_control_effort + self.KI * acceleration_error

        # Clip commands to ensure they respect the hardware limits
        limits = np.concatenate([np.tile(self.MAX_THRUST,2), [self.MAX_BODY_TORQUE], np.tile(self.MAX_JOINT1n2_TORQUE,2), [self.MAX_JOINT3_TORQUE]])        
        
        # If we are trying to calibrate gains and torque bounds...
        if self.CALIBRATE_TIMESTEP:
            #print("Accelerations: ", current_accelerations, " Unclipped Control Effort: ", control_effort, end = "")
            if self.CLIP_DURING_CALIBRATION:
                control_effort = np.clip(control_effort, -limits, limits)
                #print(" Clipped Control Effort: ", control_effort)
            else:
                #print(" ")
                pass
        else:
            control_effort = np.clip(control_effort, -limits, limits)
            #pass

        # Logging current control effort for next time step
        self.previous_control_effort = control_effort
        # [F_x, F_y, torque, torque1, torque2, torque3]
        return control_effort.reshape([self.ACTION_SIZE,1])
    
    def make_jacobian(self):
        # This method calculates the jacobian for the arm

        PHI = self.PHI
        q0 = self.chaser_position[-1]
        q1 = self.arm_angles[0]
        q2 = self.arm_angles[1]
        q3 = self.arm_angles[2]
        
        b0 = self.B0
        a1 = self.A1
        b1 = self.B1
        a2 = self.A2
        b2 = self.B2
        a3 = self.A3
        
        L1 = a1 + b1
        L2 = a2 + b2
        
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
        
        jacobian = np.array([[1,0,Jc3_13,Jc3_14,Jc3_15,Jc3_16],
                             [0,1,Jc3_23,Jc3_24,Jc3_25,Jc3_26],
                             [0,0,1,1,1,1]])
        
        return jacobian

    def reward_function(self, action):
        # Returns the reward for this TIMESTEP as a function of the state and action
        
        """
        Reward system:
                - Zero reward at all timesteps except when docking is achieved
                - A large reward when docking occurs. The episode also terminates when docking occurs
                - A variety of penalties to help with docking, such as:
                    - penalty for end-effector angle (so it goes into the docking cone properly)
                    - penalty for relative velocity during the docking (so the end-effector doesn't jab the docking cone)
	- penalty for angular velocity of the end-effector upon docking
                - A penalty for colliding with the target
                - 
         """ 
                
        # Initializing the reward
        reward = 0
        
        # Give a large reward for docking
        if self.docked:
            
            reward += self.DOCKING_REWARD
            
            # Penalize for end-effector angle
            end_effector_angle_inertial = self.chaser_position[-1] + np.sum(self.arm_angles) + np.pi/2
            
            # Docking cone angle in the target body frame
            docking_cone_angle_body = np.arctan2(self.DOCKING_PORT_CORNER1_POSITION[1] - self.DOCKING_PORT_CORNER2_POSITION[1], self.DOCKING_PORT_CORNER1_POSITION[0] - self.DOCKING_PORT_CORNER2_POSITION[0])
            docking_cone_angle_inertial = docking_cone_angle_body + self.target_position[-1] - np.pi/2 # additional -pi/2 since we must dock perpendicular into the cone
            
            # Calculate the docking angle error
            docking_angle_error = (docking_cone_angle_inertial - end_effector_angle_inertial + np.pi) % (2*np.pi) - np.pi # wrapping to [-pi, pi] 
            
            # Penalize for any non-zero angle
            reward -= np.abs(np.sin(docking_angle_error/2)) * self.MAX_DOCKING_ANGLE_PENALTY

            # Calculating the docking velocity error
            docking_relative_velocity = self.end_effector_velocity - self.docking_port_velocity
            
            # Applying the penalty
            reward -= np.linalg.norm(docking_relative_velocity) * self.DOCKING_EE_VELOCITY_PENALTY # 
            
            # Penalize for relative end-effector angular velocity upon docking
            end_effector_angular_velocity = self.chaser_velocity[-1] + np.sum(self.arm_angular_rates)
            reward -= np.abs(end_effector_angular_velocity - self.target_velocity[-1]) * self.DOCKING_ANGULAR_VELOCITY_PENALTY
            
            if self.test_time:
                print("docking successful! Reward given: %.1f distance: %.3f; relative velocity: %.3f velocity penalty: %.1f; docking angle: %.2f angle penalty: %.1f; angular rate error: %.3f angular rate penalty %.1f" %(reward, np.linalg.norm(self.end_effector_position - self.docking_port_position), np.linalg.norm(docking_relative_velocity), np.linalg.norm(docking_relative_velocity) * self.DOCKING_EE_VELOCITY_PENALTY, docking_angle_error*180/np.pi, np.abs(np.sin(docking_angle_error/2)) * self.MAX_DOCKING_ANGLE_PENALTY,np.abs(self.chaser_velocity[-1] - self.target_velocity[-1]),np.abs(self.chaser_velocity[-1] - self.target_velocity[-1]) * self.DOCKING_ANGULAR_VELOCITY_PENALTY))
        
        # Give a reward for passing a "mid-way" mark
        if self.GIVE_MID_WAY_REWARD and self.not_yet_mid_way and self.mid_way:
            if self.test_time:
                print("Just passed the mid-way mark. Distance: %.3f at time %.1f" %(np.linalg.norm(self.end_effector_position - self.docking_port_position), self.time))
            self.not_yet_mid_way = False
            reward += self.MID_WAY_REWARD
        
        # Giving a penalty for colliding with the target. These booleans are updated in self.check_collisions()
        if self.chaser_target_collision:
            reward -= self.TARGET_COLLISION_PENALTY
        
        if self.end_effector_collision:
            reward -= self.END_EFFECTOR_COLLISION_PENALTY
        
        if self.forbidden_area_collision:
            reward -= self.END_EFFECTOR_COLLISION_PENALTY
            
        if self.elbow_target_collision:
            reward -= self.END_EFFECTOR_COLLISION_PENALTY
        
        # If we've fallen off the table or rotated too much, penalize this behaviour
        if not(self.chaser_on_table) or np.abs(self.chaser_position[-1]) > 6*np.pi:
            reward -= self.FALL_OFF_TABLE_PENALTY
        
        return reward
    
    def check_collisions(self):
        """ Calculate whether the different objects are colliding with the target.
            It also checks if the chaser has fallen off the table, if the end-effector has docked,
            and if it has reached the mid-way mark
        
            Returns 7 booleans: end_effector_collision, forbidden_area_collision, chaser_target_collision, chaser_on_table, mid_way, docked, and elbow_target_collision
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
        
        # Elbow position in the inertial frame
        elbow_point = Point(self.elbow_position)
        
        ###########################
        ### Checking collisions ###
        ###########################
        self.end_effector_collision = False
        self.forbidden_area_collision = False
        self.chaser_target_collision = False
        self.mid_way = False
        self.docked = False
        self.elbow_target_collision = False
        
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
        
        # Elbow can be within the forbidden area
        if self.CHECK_END_EFFECTOR_COLLISION and elbow_point.within(target_polygon):
            if self.test_time:
                print("Elbow/target collision!")
            self.elbow_target_collision = True
        
        ##########################
        ### Mid-way or docked? ###
        ##########################
        # Docking Polygon (circle)
        docking_circle = Point(self.target_position[:-1] + np.matmul(C_Ib_target, self.DOCKING_PORT_MOUNT_POSITION)).buffer(self.SUCCESSFUL_DOCKING_RADIUS)
        
        # Mid-way Polygon (circle)
        mid_way_circle = Point(self.target_position[:-1] + np.matmul(C_Ib_target, self.DOCKING_PORT_MOUNT_POSITION)).buffer(self.MID_WAY_REWARD_RADIUS)
        
        if self.GIVE_MID_WAY_REWARD and self.not_yet_mid_way and end_effector_point.within(mid_way_circle):
            if self.test_time:
                print("Mid Way!")
            self.mid_way = True
        
        if end_effector_point.within(docking_circle):
            if self.test_time:
                print("Docked!")
            self.docked = True
            
        ######################################
        ### Checking if chaser in on table ###
        ######################################
        chaser_point = Point(self.chaser_position[:-1])
        self.chaser_on_table = chaser_point.within(self.TABLE_BOUNDARY)                                                            


    def is_done(self):
        # Checks if this episode is done or not
        """
            NOTE: THE ENVIRONMENT MUST RETURN done = True IF THE EPISODE HAS
                  REACHED ITS LAST TIMESTEP
        """

        # If we've docked with the target
        if self.docked:
            return True

        # If we've fallen off the table or spun too many times, end the episode
        if not(self.chaser_on_table) or np.abs(self.chaser_position[-1]) > 6*np.pi:
            if self.test_time:
                print("Fell off table!")
            return True

        # If we want to end the episode during a collision
        if self.END_ON_COLLISION and np.any([self.end_effector_collision, self.forbidden_area_collision, self.chaser_target_collision, self.elbow_target_collision]):
            if self.test_time:
                print("Ending episode due to a collision")
            return True
        
        # If we've run out of timesteps
        if round(self.time/self.TIMESTEP) == self.MAX_NUMBER_OF_TIMESTEPS:
            return True
        
        # The episode must not be done!
        return False


    def generate_queue(self):
        # Generate the queues responsible for communicating with the agent
        self.agent_to_env = multiprocessing.Queue(maxsize = 1)
        self.env_to_agent = multiprocessing.Queue(maxsize = 1)

        return self.agent_to_env, self.env_to_agent

    
    def make_C_bI(self, angle):
        
        C_bI = np.array([[ np.cos(angle), np.sin(angle)],
                         [-np.sin(angle), np.cos(angle)]]) # [2, 2]        
        return C_bI


    def run(self):
        ###################################
        ##### Running the environment #####
        ###################################
        """
        This method is called when the environment process is launched by main.py.
        It is responsible for continually listening for an input action from the
        agent through a Queue. If an action is received, it is to step the environment
        and return the results.
        
        TOTAL_STATE_SIZE = 22 # [chaser_x, chaser_y, chaser_theta, chaser_x_dot, chaser_y_dot, chaser_theta_dot, 
        shoulder_theta, elbow_theta, wrist_theta, shoulder_theta_dot, elbow_theta_dot, wrist_theta_dot, 
        target_x, target_y, target_theta, target_x_dot, target_y_dot, target_theta_dot, 
        ee_x, ee_y, ee_x_dot, ee_y_dot]
            
        The positions are in the inertial frame but the manipulator angles are in the joint frame.
            
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
                self.reset(test_time[0])
                
                # Return the TOTAL_STATE
                self.env_to_agent.put(self.make_total_state())

            else:
                
                # Delay the action by DYNAMICS_DELAY timesteps. The environment accumulates the action delay--the agent still thinks the sent action was used.
                if self.DYNAMICS_DELAY > 0:
                    self.action_delay_queue.put(action,False) # puts the current action to the bottom of the stack
                    action = self.action_delay_queue.get(False) # grabs the delayed action and treats it as truth.               

                ################################
                ##### Step the environment #####
                ################################ 
                reward, done = self.step(action)

                # Return (TOTAL_STATE, reward, done)
                self.env_to_agent.put((self.make_total_state(), reward, done))


#####################################################################
##### Generating the dynamics equations representing the motion #####
#####################################################################
def dynamics_equations_of_motion(chaser_state, t, parameters):
    # chaser_state = [self.chaser_position, self.arm_angles, self.chaser_velocity, self.arm_angular_rates]

    # Unpacking the chaser properties from the chaser_state
    x, y, theta, theta_1, theta_2, theta_3, x_dot, y_dot, theta_dot, theta_1_dot, theta_2_dot, theta_3_dot = chaser_state
    
    # state = x, y, theta, theta_1, theta_2, theta_3
    state_dot = np.array([x_dot, y_dot, theta_dot, theta_1_dot, theta_2_dot, theta_3_dot]).reshape([6,1])

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
    MassMatrix = np.array([t17,0.0,t30,-t86-t87-t88,-t88-t13*t28,-t88,
                           0.0,t17,t33,-t89-t90-t91,-t91-t24*t28,-t91,
                           t30,t33,INERTIA+INERTIA1+INERTIA2+INERTIA3+t46+t51+t57+t64+t65+t66+t67+t68+t70+t71+t72+t74+t75+t76+t77+t59*(A1*B0*M1*2.0+A1*B0*M2*2.0+A1*B0*M3*2.0+B0*B1*M2*2.0+B0*B1*M3*2.0)+M1*t36+M2*t36+M3*t36-t61*(A2*B0*M2*2.0+A2*B0*M3*2.0+B0*B2*M3*2.0)-A3*B0*M3*t63*2.0,t99,INERTIA2+INERTIA3+t51+t67+t68+t70+t74+t77+t85+t107-t112-t113,INERTIA3+t70+t85+t111-t113,
                           -t7*t8-t13*t14-A3*M3*t16,-t8*t23-t14*t24-A3*M3*t25,t99,INERTIA1+INERTIA2+INERTIA3+t46+t51+t57+t64+t65+t66+t67+t68+t70+t71+t72+t74+t75+t76+t77,t114,t115,
                           -t13*t28-A3*M3*t16,-t24*t28-A3*M3*t25,INERTIA2+INERTIA3+t51+t67+t68+t70+t74+t77+t85+t107-t61*t81-A3*B0*M3*t63,t114,INERTIA2+INERTIA3+t51+t67+t68+t70+t74+t77,t116,
                           -A3*M3*t16,-A3*M3*t25,INERTIA3+t70+t85+t111-A3*B0*M3*t63,t115,t116,INERTIA3+t70]).reshape([6,6], order ='F') # default order is different from matlab

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
    CoriolisMatrix = np.array([0.0,0.0,0.0,0.0,0.0,0.0,
                               0.0,0.0,0.0,0.0,0.0,0.0,
                               -t15*t16-t21*t22-t28*t29-theta_dot*t33*np.cos(t34),-t16*t35-t22*t36-t29*t37-theta_dot*t33*np.sin(t34),-t66*t67-t70*t71-t77*t78-t46*(t40+t41+t42+t43+t44)-t59*(t55+t56+t57)-t54*(t47+t48+t49+t50+t51+t52),t109+t110-t66*t67-t70*t71-t77*t78+t46*(t85+t86+t87+t88+t89),t109+t110+t112+t114-t70*t71,t110+t122+t128,
                               -t15*t16-t21*t22-t28*t29,-t16*t35-t22*t36-t29*t37,-t66*t67-t54*t82-t70*t71-t59*t84-t77*t78-t46*(t40+t41+t42+t43+t44+t85+t86+t87+t88+t89),-t66*t67-t70*t71-t77*t78,t112+t114-t70*t71,t122+t128,
                               -t21*t22-t28*t29,-t22*t36-t29*t37,-t54*t82-t70*t71-t59*t84-t66*t102-t77*t107,-t70*t71-t66*t102-t77*t107,-A3*M3*theta_3_dot*t8*t70,A3*M3*t8*t70*(theta_dot+theta_1_dot+theta_2_dot),
                               A3*M3*t38*np.sin(t39),-A3*M3*t38*np.cos(t39),-A3*B0*M3*t38*t59-A3*M3*t8*t38*t70-A3*M3*t2*t38*t77,-A3*M3*t8*t38*t70-A3*M3*t2*t38*t77,-A3*M3*t8*t38*t70,00]).reshape([6,6], order='F') # default order is different than matlab
            
    second_derivatives = np.matmul(np.linalg.inv(MassMatrix),(control_effort - np.matmul(CoriolisMatrix, state_dot)))

    first_derivatives = np.array([x_dot, y_dot, theta_dot, theta_1_dot, theta_2_dot, theta_3_dot]).reshape([6,1])
    
    full_derivative = np.concatenate([first_derivatives, second_derivatives]).squeeze()
    
    return full_derivative


##########################################
##### Function to animate the motion #####
##########################################
def render(states, actions, instantaneous_reward_log, cumulative_reward_log, critic_distributions, target_critic_distributions, projected_target_distribution, bins, loss_log, episode_number, filename, save_directory):

    # Load in a temporary environment, used to grab the physical parameters
    temp_env = Environment()

    # Checking if we want the additional reward and value distribution information
    extra_information = temp_env.ADDITIONAL_VALUE_INFO

    # Unpacking state from TOTAL_STATE 
    """
    [chaser_x, chaser_y, chaser_theta, chaser_x_dot, chaser_y_dot, chaser_theta_dot, 
     shoulder_theta, elbow_theta, wrist_theta, shoulder_theta_dot, elbow_theta_dot, wrist_theta_dot, 
     target_x, target_y, target_theta, target_x_dot, target_y_dot, target_theta_dot, ee_x, ee_y, ee_x_dot, ee_y_dot]
    """
    # Chaser positions
    chaser_x, chaser_y, chaser_theta, theta_1, theta_2, theta_3 = states[:,0], states[:,1], states[:,2], states[:,6], states[:,7], states[:,8]
    
    # Target positions
    target_x, target_y, target_theta = states[:,12], states[:,13], states[:,14]

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
    DOCKING_PORT_MOUNT_POSITION = temp_env.DOCKING_PORT_MOUNT_POSITION
    DOCKING_PORT_CORNER1_POSITION = temp_env.DOCKING_PORT_CORNER1_POSITION
    DOCKING_PORT_CORNER2_POSITION = temp_env.DOCKING_PORT_CORNER2_POSITION

    #################################################
    ### Calculating chaser locations through time ###
    #################################################
    
    ##############################################
    ### Manipulator Joint Locations (Inertial) ###
    ##############################################
    # Shoulder
    shoulder_x = chaser_x + B0*np.cos(chaser_theta + PHI)
    shoulder_y = chaser_y + B0*np.sin(chaser_theta + PHI)

    # Elbow
    elbow_x = shoulder_x + (A1 + B1)*np.cos(np.pi/2 + chaser_theta + theta_1)
    elbow_y = shoulder_y + (A1 + B1)*np.sin(np.pi/2 + chaser_theta + theta_1)

    # Wrist
    wrist_x = elbow_x + (A2 + B2)*np.cos(np.pi/2 + chaser_theta + theta_1 + theta_2)
    wrist_y = elbow_y + (A2 + B2)*np.sin(np.pi/2 + chaser_theta + theta_1 + theta_2)

    # End-effector
    end_effector_x = wrist_x + (A3 + B3)*np.cos(np.pi/2 + chaser_theta + theta_1 + theta_2 + theta_3)
    end_effector_y = wrist_y + (A3 + B3)*np.sin(np.pi/2 + chaser_theta + theta_1 + theta_2 + theta_3)

    ###############################
    ### Chaser corner locations ###
    ###############################

    # All the points to draw of the chaser (except the front-face)    
    chaser_points_body = np.array([[ LENGTH/2,-LENGTH/2],
                                   [-LENGTH/2,-LENGTH/2],
                                   [-LENGTH/2, LENGTH/2],
                                   [ LENGTH/2, LENGTH/2]]).T
    
    # The front-face points on the target
    chaser_front_face_body = np.array([[[ LENGTH/2],[ LENGTH/2]],
                                       [[ LENGTH/2],[-LENGTH/2]]]).squeeze().T

    # Rotation matrix (body -> inertial)
    C_Ib_chaser = np.moveaxis(np.array([[np.cos(chaser_theta), -np.sin(chaser_theta)],
                                        [np.sin(chaser_theta),  np.cos(chaser_theta)]]), source = 2, destination = 0) # [NUM_TIMESTEPS, 2, 2]
    
    # Rotating body frame coordinates to inertial frame    
    chaser_body_inertial       = np.matmul(C_Ib_chaser, chaser_points_body)     + np.array([chaser_x, chaser_y]).T.reshape([-1,2,1])
    chaser_front_face_inertial = np.matmul(C_Ib_chaser, chaser_front_face_body) + np.array([chaser_x, chaser_y]).T.reshape([-1,2,1])

    #############################
    ### Target Body Locations ###
    #############################
    # All the points to draw of the target (except the front-face)     
    target_points_body = np.array([[ LENGTH/2,-LENGTH/2],
                                   [-LENGTH/2,-LENGTH/2],
                                   [-LENGTH/2, LENGTH/2],
                                   [ LENGTH/2, LENGTH/2],
                                   [DOCKING_PORT_MOUNT_POSITION[0],DOCKING_PORT_MOUNT_POSITION[1]],
                                   [DOCKING_PORT_CORNER1_POSITION[0],DOCKING_PORT_CORNER1_POSITION[1]],
                                   [DOCKING_PORT_CORNER2_POSITION[0],DOCKING_PORT_CORNER2_POSITION[1]],
                                   [DOCKING_PORT_MOUNT_POSITION[0],DOCKING_PORT_MOUNT_POSITION[1]]]).T
    
    # The front-face points on the target
    target_front_face_body = np.array([[[ LENGTH/2],[ LENGTH/2]],
                                       [[ LENGTH/2],[-LENGTH/2]]]).squeeze().T

    # Rotation matrix (body -> inertial)
    C_Ib_target = np.moveaxis(np.array([[np.cos(target_theta), -np.sin(target_theta)],
                                        [np.sin(target_theta),  np.cos(target_theta)]]), source = 2, destination = 0) # [NUM_TIMESTEPS, 2, 2]
    
    # Rotating body frame coordinates to inertial frame
    target_body_inertial       = np.matmul(C_Ib_target, target_points_body)     + np.array([target_x, target_y]).T.reshape([-1,2,1])
    target_front_face_inertial = np.matmul(C_Ib_target, target_front_face_body) + np.array([target_x, target_y]).T.reshape([-1,2,1])

    
    # Calculating the accelerations for each state through time
    velocities = np.concatenate([states[:,3:6],states[:,9:12]], axis = 1)
    # Numerically differentiating to approximate the derivative
    accelerations = np.diff(velocities, axis = 0)/temp_env.TIMESTEP
    # Add a row of zeros initially to the current acceleartions
    accelerations = np.concatenate([np.zeros([1,temp_env.ACTION_SIZE]), accelerations])
    
    # Adding a row of zeros to the actions for the first timestep
    actions = np.concatenate([np.zeros([1,temp_env.ACTION_SIZE]), actions])

    #######################
    ### Plotting Motion ###
    #######################
    
    # Generating figure window
    figure = plt.figure(constrained_layout = True)
    figure.set_size_inches(5, 4, True)

    if extra_information:
        grid_spec = gridspec.GridSpec(nrows = 2, ncols = 3, figure = figure)
        subfig1 = figure.add_subplot(grid_spec[0,0], aspect = 'equal', autoscale_on = False, xlim = (0, 3.5), ylim = (0, 2.4))
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
        subfig1 = figure.add_subplot(1, 1, 1, aspect = 'equal', autoscale_on = False, xlim = (0, temp_env.MAX_X_POSITION), ylim = (0, temp_env.MAX_Y_POSITION), xlabel = 'X Position (m)', ylabel = 'Y Position (m)')

    # Defining plotting objects that change each frame
    chaser_body,       = subfig1.plot([], [], color = 'r', linestyle = '-', linewidth = 2) # Note, the comma is needed
    chaser_front_face, = subfig1.plot([], [], color = 'k', linestyle = '-', linewidth = 2) # Note, the comma is needed
    target_body,       = subfig1.plot([], [], color = 'g', linestyle = '-', linewidth = 2)
    target_front_face, = subfig1.plot([], [], color = 'k', linestyle = '-', linewidth = 2)
    manipulator,       = subfig1.plot([], [], color = 'r', linestyle = '-', linewidth = 2) # Note, the comma is needed

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
        episode_text.set_text('Episode ' + str(episode_number))
        control1_text = subfig1.text(x = 0.01, y = 0.90, s = '', fontsize = 6, transform=subfig1.transAxes)
        control2_text = subfig1.text(x = 0.01, y = 0.85, s = '', fontsize = 6, transform=subfig1.transAxes)
        control3_text = subfig1.text(x = 0.01, y = 0.80, s = '', fontsize = 6, transform=subfig1.transAxes)
        control4_text = subfig1.text(x = 0.01, y = 0.75, s = '', fontsize = 6, transform=subfig1.transAxes)
        control5_text = subfig1.text(x = 0.01, y = 0.70, s = '', fontsize = 6, transform=subfig1.transAxes)
        control6_text = subfig1.text(x = 0.01, y = 0.65, s = '', fontsize = 6, transform=subfig1.transAxes)

    # Function called repeatedly to draw each frame
    def render_one_frame(frame, *fargs):
        temp_env = fargs[0] # Extract environment from passed args

        # Draw the chaser body
        chaser_body.set_data(chaser_body_inertial[frame,0,:], chaser_body_inertial[frame,1,:])

        # Draw the front face of the chaser body in a different colour
        chaser_front_face.set_data(chaser_front_face_inertial[frame,0,:], chaser_front_face_inertial[frame,1,:])

        # Draw the target body
        target_body.set_data(target_body_inertial[frame,0,:], target_body_inertial[frame,1,:])

        # Draw the front face of the target body in a different colour
        target_front_face.set_data(target_front_face_inertial[frame,0,:], target_front_face_inertial[frame,1,:])

        # Draw the manipulator
        thisx = [shoulder_x[frame], elbow_x[frame], wrist_x[frame], end_effector_x[frame]]
        thisy = [shoulder_y[frame], elbow_y[frame], wrist_y[frame], end_effector_y[frame]]
        manipulator.set_data(thisx, thisy)

        # Update the time text
        time_text.set_text('Time = %.1f s' %(frame*temp_env.TIMESTEP))
        
        # Update the control text
        control1_text.set_text('$\ddot{x}$ = %6.3f; true = %6.3f' %(actions[frame,0], accelerations[frame,0]))
        control2_text.set_text('$\ddot{y}$ = %6.3f; true = %6.3f' %(actions[frame,1], accelerations[frame,1]))
        control3_text.set_text(r'$\ddot{\theta}$ = %1.3f; true = %6.3f' %(actions[frame,2], accelerations[frame,2]))
        control4_text.set_text('$\ddot{q_0}$ = %6.3f; true = %6.3f' %(actions[frame,3], accelerations[frame,3]))
        control5_text.set_text('$\ddot{q_1}$ = %6.3f; true = %6.3f' %(actions[frame,4], accelerations[frame,4]))
        control6_text.set_text('$\ddot{q_2}$ = %6.3f; true = %6.3f' %(actions[frame,5], accelerations[frame,5]))

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

        # Since blit = True, must return everything that has changed at this frame
        return time_text, reward_text, chaser_body, chaser_front_face, target_body, target_front_face, manipulator

    # Generate the animation!
    fargs = [temp_env] # bundling additional arguments
    animator = animation.FuncAnimation(figure, render_one_frame, frames = np.linspace(0, len(states)-1, len(states)).astype(int),
                                       blit = False, fargs = fargs)
    """
    frames = the int that is passed to render_one_frame. I use it to selectively plot certain data
    fargs = additional arguments for render_one_frame
    interval = delay between frames in ms
    """

    # Save the animation!
    if temp_env.SKIP_FAILED_ANIMATIONS:
        try:
            if temp_env.ON_COMPUTE_CANADA:
                # Save it to the working directory [have to], then move it to the proper folder
                animator.save(filename = os.environ['SLURM_TMPDIR'] + '/' + filename + '_episode_' + str(episode_number) + '.mp4', fps = 30, dpi = 100)
                # Make directory if it doesn't already exist
                os.makedirs(os.path.dirname(save_directory + filename + '/videos/'), exist_ok=True)
                # Move animation to the proper directory
                os.rename(os.environ['SLURM_TMPDIR'] + '/' + filename + '_episode_' + str(episode_number) + '.mp4', save_directory + filename + '/videos/episode_' + str(episode_number) + '.mp4')                                
            else:
                # Save it to the working directory [have to], then move it to the proper folder
                animator.save(filename = filename + '_episode_' + str(episode_number) + '.mp4', fps = 30, dpi = 100)
                # Make directory if it doesn't already exist
                os.makedirs(os.path.dirname(save_directory + filename + '/videos/'), exist_ok=True)
                # Move animation to the proper directory
                os.rename(filename + '_episode_' + str(episode_number) + '.mp4', save_directory + filename + '/videos/episode_' + str(episode_number) + '.mp4')
        except:
            ("Skipping animation for episode %i due to an error" %episode_number)
            # Try to delete the partially completed video file
            try:
                os.remove(filename + '_episode_' + str(episode_number) + '.mp4')
            except:
                pass
    else:
        if temp_env.ON_COMPUTE_CANADA:
            # Save it to the working directory [have to], then move it to the proper folder
            animator.save(filename = os.environ['SLURM_TMPDIR'] + '/' + filename + '_episode_' + str(episode_number) + '.mp4', fps = 30, dpi = 100)
            # Make directory if it doesn't already exist
            os.makedirs(os.path.dirname(save_directory + filename + '/videos/'), exist_ok=True)
            # Move animation to the proper directory
            os.rename(os.environ['SLURM_TMPDIR'] + '/' + filename + '_episode_' + str(episode_number) + '.mp4', save_directory + filename + '/videos/episode_' + str(episode_number) + '.mp4')

        else:
            # Save it to the working directory [have to], then move it to the proper folder
            animator.save(filename = filename + '_episode_' + str(episode_number) + '.mp4', fps = 30, dpi = 100)
            # Make directory if it doesn't already exist
            os.makedirs(os.path.dirname(save_directory + filename + '/videos/'), exist_ok=True)
            # Move animation to the proper directory
            os.rename(filename + '_episode_' + str(episode_number) + '.mp4', save_directory + filename + '/videos/episode_' + str(episode_number) + '.mp4')

    del temp_env
    plt.close(figure)