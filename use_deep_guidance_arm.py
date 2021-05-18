"""
This script loads in a trained policy neural network and uses it for inference.

Typically this script will be executed on the Nvidia Jetson TX2 board during an
experiment in the Spacecraft Robotics and Control Laboratory at Carleton
University.

Script created: June 12, 2019
@author: Kirk (khovell@gmail.com)
"""

import tensorflow as tf
import numpy as np
import socket
import time

from settings import Settings
from build_neural_networks import BuildActorNetwork

#%%
testing = True # [boolean] Set to True for testing purposes (without using the Jetson)
counter = 1

#%%
# Clear any old graph
tf.reset_default_graph()

# Initialize Tensorflow, and load in policy
with tf.Session() as sess:
    # Building the policy network
    state_placeholder = tf.placeholder(dtype = tf.float32, shape = [None, Settings.OBSERVATION_SIZE], name = "state_placeholder")
    actor = BuildActorNetwork(state_placeholder, scope='learner_actor_main')

    # Loading in trained network weights
    print("Attempting to load in previously-trained model\n")
    saver = tf.train.Saver() # initialize the tensorflow Saver()

    # Try to load in policy network parameters
    try:
        ckpt = tf.train.get_checkpoint_state('../')
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("\nModel successfully loaded!\n")

    except (ValueError, AttributeError):
        print("No model found... quitting :(")
        raise SystemExit

    ##################################################
    #### Start communication with JetsonRepeater #####
    ##################################################
    # Initializing
    connected = False

    # Looping forever
    while True:
        if not connected and not testing: # If we aren't connected, try to connect to the JetsonRepeater program
            try: # Try to connect
                client_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                client_socket.connect("/tmp/jetsonRepeater") # Connecting...
                connected = True
                print("Connected to JetsonRepeater! Deep Guidance Started")
            except: # If connection attempt failed
                print("Connection to JetsonRepeater FAILED. Trying to re-connect in 1 second")
                time.sleep(1)
                continue # Restart from while True
        else: # We are connected!
            if not testing:
                try: # Try to receive data
                    data = client_socket.recv(4096)
                except: # If receive fails, we have lost communication with the JetsonRepeater
                    print("Lost communication with JetsonRepeater")
                    connected = False
                    continue # Restart from while True
                if data == False: # If it is time to end
                    print('Terminating Deep Guidance')
                    break
                else: # We got good data!
                    input_data = data.decode("utf-8")
                    #print('Got message: ' + str(data.decode("utf-8")))

            ############################################################
            ##### Received data! Process it and return the result! #####
            ############################################################

            # Receive position data
            if testing:
                input_data_array = np.zeros(Settings.TOTAL_STATE_SIZE)
            else:
                input_data_array = np.array(input_data.splitlines()).astype(np.float32)
                # input_data_array is: [time, red_x, red_y, red_theta, black_x, black_y, black_theta]
            
    	    # Calculating the proper policy input
            # Want the input to be: [red_x, red_y, red_theta, x_error, y_error, theta_error]
            # and we want it to be scaled properly
            
            # Determining the phase we are in
            if input_data_array[0] < 135: # 45 + phase_2_end_time
                # Hold point phase!
                desired_x = input_data_array[4] + 0.9*np.cos(input_data_array[6])
                desired_y = input_data_array[5] + 0.9*np.sin(input_data_array[6])
                desired_angle = input_data_array[6] - np.pi
            else:
                # Docking phase!
                desired_x = input_data_array[4] + 0.5*np.cos(input_data_array[6])
                desired_y = input_data_array[5] + 0.5*np.sin(input_data_array[6])
                desired_angle = input_data_array[6] - np.pi
                        
            #################################
            ### Building the Policy Input ###
            #################################
            policy_input = input_data_array

            # Normalizing
            if Settings.NORMALIZE_STATE:
                normalized_policy_input = (policy_input - Settings.STATE_MEAN)/Settings.STATE_HALF_RANGE
            else:
                normalized_policy_input = policy_input

            # Discarding irrelevant states
            normalized_policy_input = np.delete(normalized_policy_input, Settings.IRRELEVANT_STATES)

            # Reshaping the input
            normalized_policy_input = normalized_policy_input.reshape([-1, Settings.OBSERVATION_SIZE])

            # Run processed state through the policy
            deep_guidance = sess.run(actor.action_scaled, feed_dict={state_placeholder:normalized_policy_input})[0] # [accel_x, accel_y, alpha]

            # Return commanded action to the Raspberry Pi 3
            if testing:
                pass
            else:
                out_data = str(deep_guidance[0]) + "\n" + str(deep_guidance[1]) + "\n" + str(deep_guidance[2]) + "\n"
                client_socket.send(out_data.encode())
            
            if counter % 200 == 0:
                print("Input from Pi: ", input_data_array)
                print("Policy input: ", policy_input)
                print("Normalized policy input: ", normalized_policy_input)
                print("Output to Pi: ", deep_guidance)
            # Incrementing the counter
            counter = counter + 1

print('Done :)')
