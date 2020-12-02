
"""
This script provides the environment for a free flying spacecraft with a
three-link manipulator.

A manipulator-equipped spacecraft is tasked with SOMETHING. A policy is learned
to accomplish this task.

The policy is trained in a kinematics environment, which is the result of
assuming a perfect controller is present.

The policy is then tested in a full dynamics environment where a controller is
used. The policy is used as a guidance signal along with a controller.

If it hypothesized that if Noisy Kinematics are used to train the guidance policy,
the policy should be able to be combined with any type of controller when deployed
in reality. The guidance policy should still perform well when deployed to reality
because it knows how to handle the non-ideal conditions that will be encountered
by the real controller and poorly-modeled dynamics.

My training will be performed as follows:
    1) Train a policy using noise-free kinematics:
        - Policy exploration noise used during training
        - No policy noise used during test time
            - Constant conditions
            - Randomized conditions
        This will test the implementation.

    2) Train a policy using noisy kinematics:
        - Policy exploration noise used during training
        - No poilicy noise used during test time
            - Constant conditions
            - Randomized conditions
        This will force the agent to learn how to handle undesirable states,
        which will be encountered by a real-life non-ideal controller.

    3) Evaluate the policy trained in 2) with full dynamics
        - Evaluate during the training of 2) [I have this as item 3) because I
          need 2) working on its own first]
        - Use policy with no noise (since it's test time) to get desired velocities
        - Integrate desired velocities into desired positions
        - Employ a PD controller to track those positions
        - Run the PD controller output through full dynamics
    Step 3) is what will (hopefully) eventually be the key to bridging the gap
    between simulation and reality. By training the policy on noisy kinematics,
    it should develop the ability to accomplish the goal (get the rewards) even
    when it isn't being controlled ideally (such as the case in reality).

The preceding 3 steps will be applied to the following environments of increasing
difficulty in this order:
    0) Get the essential learning and simulating functions working
    1) Manipulator moving to a given location

Experimental validation will accompany each set of results.

It is anticipated that the results of this work will lead to a JGCD Note, showing
that the deep guidance technique can be applied to a range of aerospace
applications.

### We randomize the things we want to be robust to ###

### We avoid training with a controller because we don't want to overfit to a particular controller ###

All dynamic environments I create will have a standardized architecture. The
reason for this is I have one learning algorithm and many environments. All
environments are responsible for:
    - dynamics propagation (via the step method)
    - initial conditions   (via the reset method)
    - reporting environment properties (defined in __init__)
    - seeding the dynamics (via the seed method)
    - animating the motion (via the render method):
        - Rendering is done all in one shot by passing the completed states
          from a trial to the render() method.

Outputs:
    Reward must be of shape ()
    State must be of shape (STATE_SIZE,)
    Done must be a bool

Inputs:
    Action input is of shape (ACTION_SIZE,)

Communication with agent:
    The agent communicates to the environment through two queues:
        agent_to_env: the agent passes actions or reset signals to the environment
        env_to_agent: the environment returns information to the agent

### Each environment is also responsible for performing all three Steps, listed above ###

Notes:
    - Should the inv(M)(tau - C*q) be state[6:] instead of state[:6]? Should it be the state or its derivative?
      Actually, I think it should be state[:6] since it's supposed to be q_dot


@author: Kirk Hovell (khovell@gmail.com)
"""
import numpy as np
import os
import signal
import multiprocessing
from scipy.integrate import odeint # Numerical integrator

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec

class Environment:

    def __init__(self):
        ##################################
        ##### Environment Properties #####
        ##################################
        self.TOTAL_STATE_SIZE          = 8 # [x, y, theta, theta_1, theta_2, theta_3, desired_x, desired_y]
        self.IRRELEVANT_STATES         = [] # states that are irrelevant to the policy network in its decision making
        self.STATE_SIZE                = self.TOTAL_STATE_SIZE - len(self.IRRELEVANT_STATES) # total number of relevant states
        self.ACTION_SIZE               = 6 # [x_dot, y_dot, theta_dot, theta_1_dot, theta_2_dot, theta_3_dot]
        self.LOWER_ACTION_BOUND        = np.array([-0.2, -0.2, -np.pi/6, -np.pi/12, -np.pi/12, -np.pi/12]) # [m/s, m/s, rad/s, rad/s, rad/s, rad/s]
        self.UPPER_ACTION_BOUND        = np.array([ 0.2,  0.2,  np.pi/6,  np.pi/12,  np.pi/12,  np.pi/12]) # [m/s, m/s, rad/s, rad/s, rad/s, rad/s]
        self.LOWER_STATE_BOUND         = np.array([  0.,   0., -4*2*np.pi, -np.pi/2, -np.pi/2, -np.pi/2, 0. , 0. ]) # [m, m, rad, rad, rad, rad, m, m]
        self.UPPER_STATE_BOUND         = np.array([ 3.7,  2.4,  4*2*np.pi,  np.pi/2,  np.pi/2,  np.pi/2, 3.7, 2.4]) # [m, m, rad, rad, rad, rad, m, m]
        self.NORMALIZE_STATE           = True # Normalize state on each timestep to avoid vanishing gradients
        self.RANDOMIZE                 = False # whether or not to RANDOMIZE the state & target location
        self.NOMINAL_INITIAL_POSITION  = np.array([3.0, 1.0, 0.0, np.pi/3, np.pi/3, -np.pi/3]) # [m, m, rad, rad, rad, rad]
        self.NOMINAL_TARGET_POSITION   = np.array([2.0, 1.6]) # [m, m] end-effector desired position
        self.MIN_V                     = -150.
        self.MAX_V                     =  20.
        self.N_STEP_RETURN             =   5
        self.DISCOUNT_FACTOR           =   0.95**(1/self.N_STEP_RETURN)
        self.TIMESTEP                  =   0.05 # [s] originally 0.2 seconds

        self.TARGET_REWARD             =   1. # reward per second
        self.FALL_OFF_TABLE_PENALTY    =   0.
        self.END_ON_FALL               = False # end episode on a fall off the table
        self.GOAL_REWARD               =   0. # one time reward to reaching the goal state
        self.NEGATIVE_PENALTY_FACTOR   = 1.5 # How much of a factor to additionally penalize negative rewards
        self.MAX_NUMBER_OF_TIMESTEPS   = 450 # [450] per episode
        self.ADDITIONAL_VALUE_INFO     = True # whether or not to include additional reward and value distribution information on the animations
        self.REWARD_TYPE               = True # True = Linear; False = Exponential
        self.REWARD_WEIGHTING          = [1, 1] # How much to weight the rewards in the state
        self.REWARD_MULTIPLIER         = 250 # how much to multiply the differential reward by
        self.MANIPULATOR_MAX_RATE      = 3 # [rad/s] maximum angular rate of any joint that will end the episode

        # Test time properties
        self.TEST_ON_DYNAMICS         = True # Whether or not to use full dynamics along with a PD controller at test time
        self.KINEMATIC_NOISE          = False # Whether or not to apply noise to the kinematics in order to simulate a poor controller
        self.KINEMATIC_NOISE_SD       = [0.02, 0.02, np.pi/100] # The standard deviation of the noise that is to be applied to each element in the state
        self.FORCE_NOISE_AT_TEST_TIME = False # [Default -> False] Whether or not to force kinematic noise to be present at test time

        # PD Controller Gains
        self.KP                       = 2.0 # PD controller gain
        self.KD                       = 4.0 # PD controller gain
        self.CONTROLLER_ERROR_WEIGHT  = [1.5, 1.5, 0.08, 0.02, 0.02, 0.02] # [1, 1, 0.05, 0.01, 0.01, 0.01]# How much to weight each error signal (for example, to weight the angle error less than the position error)

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

        # Resetting phase number so we complete phase 0 before moving on to phase 1
        self.phase_number = 0

        # Logging whether it is test time for this episode
        self.test_time = test_time

        # If we are randomizing the initial consitions and state
        if self.RANDOMIZE:
            # Randomizing initial state
            self.state = self.NOMINAL_INITIAL_POSITION + np.random.randn(3)*[0.3, 0.3, np.pi/2]
            # Randomizing target state
            self.target_location = self.NOMINAL_TARGET_POSITION + np.random.randn(3)*[0.3, 0.3, np.pi/2]
        else:
            # Constant initial state
            self.state = self.NOMINAL_INITIAL_POSITION
            # Constant target location
            self.target_location = self.NOMINAL_TARGET_POSITION

        # How long is the position portion of the state
        self.POSITION_STATE_LENGTH = len(self.state)

        if use_dynamics:
            # Setting the dynamics state to be equal, initially, to the kinematics state, plus the velocity initial conditions state
            velocity_initial_conditions = np.array([0., 0., 0., 0., 0., 0.])
            self.state = np.concatenate((self.state, velocity_initial_conditions))
            """ Note: dynamics_state = [x, y, theta, xdot, ydot, thetadot] """
            self.dynamics_flag = True # for this episode, dynamics will be used

        # Resetting the time
        self.time = 0.

        # Resetting the differential reward
        self.previous_position_reward = [None, None, None]


    #####################################
    ##### Step the Dynamics forward #####
    #####################################
    def step(self, action):

        # Integrating forward one time step.
        # Returns initial condition on first row then next TIMESTEP on the next row
        #########################################
        ##### PROPAGATE KINEMATICS/DYNAMICS #####
        #########################################
        if self.dynamics_flag:
            # Additional parameters to be passed to the kinematics
            kinematics_parameters = [action]

            ############################
            #### PROPAGATE DYNAMICS ####
            ############################
            # First calculate the next guidance command
            guidance_propagation = odeint(kinematics_equations_of_motion, self.state[:self.POSITION_STATE_LENGTH], [self.time, self.time + self.TIMESTEP], args = (kinematics_parameters,), full_output = 0)

            # Saving the new guidance signal
            guidance_position = guidance_propagation[1,:]

            # Next, calculate the control effort
            control_effort = self.controller(guidance_position, action) # Passing the desired position and velocity (Note: the action is the desired velocity)

            # Anything additional that needs to be sent to the dynamics integrator
            dynamics_parameters = [control_effort, self.LENGTH, self.PHI, self.B0, self.MASS, self.M1, self.M2, self.M3, self.A1, self.B1, self.A2, self.B2, self.A3, self.B3, self.INERTIA, self.INERTIA1, self.INERTIA2, self.INERTIA3]

            # Finally, propagate the dynamics forward one timestep
            next_states = odeint(dynamics_equations_of_motion, self.state, [self.time, self.time + self.TIMESTEP], args = (dynamics_parameters,), full_output = 0)

            # Saving the new state
            self.state = next_states[1,:]

        else:

            # Additional parameters to be passed to the kinematics
            kinematics_parameters = [action]

            # Dummy guidance position
            guidance_position = []

            ###############################
            #### PROPAGATE KINEMATICS #####
            ###############################
            next_states = odeint(kinematics_equations_of_motion, self.state, [self.time, self.time + self.TIMESTEP], args = (kinematics_parameters,), full_output = 0)

            # Saving the new state
            self.state = next_states[1,:]

            # Optionally, add noise to the kinematics to simulate "controller noise"
            if self.KINEMATIC_NOISE and (not self.test_time or self.FORCE_NOISE_AT_TEST_TIME):
                 # Add some noise to the position part of the state
                 self.state += np.random.randn(self.POSITION_STATE_LENGTH) * self.KINEMATIC_NOISE_SD


        # Done the differences between the kinematics and dynamics
        # Increment the timestep
        self.time += self.TIMESTEP

        # Calculating the reward for this state-action pair
        reward = self.reward_function(action)

        # Check if this episode is done
        done = self.is_done()

        # Return the (state, reward, done)
        return self.state, reward, done, guidance_position

    def controller(self, guidance_position, guidance_velocity):
        # This function calculates the control effort based on the state and the
        # desired position (guidance_command)
        position_error = guidance_position - self.state[:self.POSITION_STATE_LENGTH]
        velocity_error = guidance_velocity - self.state[self.POSITION_STATE_LENGTH:]

        # Using a PD controller on all states independently
        control_effort = self.KP * position_error*self.CONTROLLER_ERROR_WEIGHT + self.KD * velocity_error*self.CONTROLLER_ERROR_WEIGHT

        return control_effort

    def pose_error(self):
        """
        This method returns the pose error of the current state.
        Instead of returning [state, desired_state] as the state, I'll return
        [state, error]. The error will be more helpful to the policy I believe.
        """

        return self.target_location - self.end_effector_position()

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

    def reward_function(self, action):
        # Returns the reward for this TIMESTEP as a function of the state and action

        current_position_reward = np.zeros(1)

        # Calculates a reward map
        if self.REWARD_TYPE:
            # Linear reward
            #current_position_reward = -np.linalg.norm((target_location - self.state[:self.POSITION_STATE_LENGTH])*self.REWARD_WEIGHTING) * self.TARGET_REWARD
            # Negative the norm of the error vector -> The best reward possible is zero.

            #reward_old = -np.linalg.norm((target_location - self.state[:self.POSITION_STATE_LENGTH])*self.REWARD_WEIGHTING) * self.TARGET_REWARD
            current_position_reward = -np.abs((self.target_location - self.end_effector_position())*self.REWARD_WEIGHTING)* self.TARGET_REWARD

        else:
            # Exponential reward
            current_position_reward = np.exp(-np.sum(np.absolute(self.target_location - self.end_effector_position())*self.REWARD_WEIGHTING)) * self.TARGET_REWARD

        reward = np.zeros(1)

        if np.all([self.previous_position_reward[i] is not None for i in range(len(self.previous_position_reward))]):
            #print(self.previous_position_reward is not None, current_position_reward, self.previous_position_reward)
            reward = (current_position_reward - self.previous_position_reward)*self.REWARD_MULTIPLIER
            for i in range(len(reward)):
                if reward[i] < 0:
                    reward[i]*= self.NEGATIVE_PENALTY_FACTOR

        self.previous_position_reward = current_position_reward

        # Collapsing to a scalar
        reward = np.sum(reward)

        ##################################
        ### Extra Rewards or Penalties ###
        ##################################

        # Giving a massive penalty for falling off the table
        if self.state[0] > self.UPPER_STATE_BOUND[0] or self.state[0] < self.LOWER_STATE_BOUND[0] or self.state[1] > self.UPPER_STATE_BOUND[1] or self.state[1] < self.LOWER_STATE_BOUND[1]:
            reward -= self.FALL_OFF_TABLE_PENALTY/self.TIMESTEP

        # Giving a large reward for completing the task
        if np.sum(np.absolute(self.end_effector_position() - self.target_location)) < 0.01:
            reward += self.GOAL_REWARD
            
        # Penalizing base motion
        reward -= np.abs(action[0]) + np.abs(action[1])
        
        # Penalizing arm motion
        #reward -= np.abs(action[3]) + np.abs(action[4]) + np.abs(action[5])

        # Multiplying the reward by the TIMESTEP to give the rewards on a per-second basis
        return (reward*self.TIMESTEP).squeeze()

    def is_done(self):
        # Checks if this episode is done or not
        """
            NOTE: THE ENVIRONMENT MUST RETURN done = True IF THE EPISODE HAS
                  REACHED ITS LAST TIMESTEP
        """

        # If we've fallen off the table, end the episode
        if self.state[0] > self.UPPER_STATE_BOUND[0] or self.state[0] < self.LOWER_STATE_BOUND[0] or self.state[1] > self.UPPER_STATE_BOUND[1] or self.state[1] < self.LOWER_STATE_BOUND[1]:
            done = self.END_ON_FALL
        else:
            done = False

        # If we've spun too many times
        if self.state[2] > self.UPPER_STATE_BOUND[2] or self.state[2] < self.LOWER_STATE_BOUND[2]:
            pass
            #done = False

        # If the manipulator is spinning too quickly
        if np.any(np.abs(self.state[9:]) > self.MANIPULATOR_MAX_RATE):
            print("Ended from manipulator spinning too fast", self.state, self.time)
            done = True

        # If we've run out of timesteps
        if round(self.time/self.TIMESTEP) == self.MAX_NUMBER_OF_TIMESTEPS:
            done = True

        return done


    def generate_queue(self):
        # Generate the queues responsible for communicating with the agent
        self.agent_to_env = multiprocessing.Queue(maxsize = 1)
        self.env_to_agent = multiprocessing.Queue(maxsize = 1)

        return self.agent_to_env, self.env_to_agent

    def run(self):
        ###################################
        ##### Running the environment #####
        ###################################
        """
        This method is called when the environment process is launched by main.py.
        It is responsible for continually listening for an input action from the
        agent through a Queue. If an action is received, it is to step the environment
        and return the results.
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
                # Return the results
                self.env_to_agent.put((np.append(self.state[:self.POSITION_STATE_LENGTH], self.pose_error()), self.target_location))

            else:
                ################################
                ##### Step the environment #####
                ################################
                next_state, reward, done, *guidance_position = self.step(action)

                # Return the results
                self.env_to_agent.put((np.append(next_state[:self.POSITION_STATE_LENGTH], self.pose_error()), reward, done, guidance_position))

###################################################################
##### Generating kinematics equations representing the motion #####
###################################################################
def kinematics_equations_of_motion(state, t, parameters):
    # From the state, it returns the first derivative of the state

    # Unpacking the action from the parameters
    action = parameters[0]

    # Building the derivative matrix. For kinematics, d(state)/dt = action = \dot{state}
    derivatives = action

    return derivatives


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