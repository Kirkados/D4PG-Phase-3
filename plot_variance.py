""" This script loads the deep guidance data logged from an experiment (specifically, logged by use_deep_guidance_arm.py)
renders a few plots, and animates the motion.

It should be run from the folder where use_deep_guidance_arm.py was run from for the given experiment.
 """
import numpy as np
import csv
import matplotlib.pyplot as plt

#####################################
### Load in the experimental data ###
#####################################
data_log = csv.reader(open("biased.csv", "rb"), delimiter=",")
data = np.loadtxt(open("biased.csv", "rb"), delimiter=",", skiprows=1)
#data = np.array(list(data_log)).astype("float")

########################
### Plot some things ###
########################
episode_number = data[:,1]
combined_residual_angular_momentum = data[:,2]

# Plotting raw combined spin rate
plt.figure()
plt.plot(episode_number, combined_residual_angular_momentum)
plt.xlabel('Episode number')
plt.ylabel('Raw combined angular rate [deg/s]')



# Calculate the variance at many points throughout the data file
# Using 
window_size = 100 # how many data points to include in the variance calculation
variance = []
episode_numbers_where_variance_was_calculated = []

for i in range(len(episode_number)):
    if i < 10:
        continue
    try:
        if i - window_size < 0:
            these_data_points = combined_residual_angular_momentum[0:i]
        else:
            these_data_points = combined_residual_angular_momentum[i-window_size:i]
    except:
        print("Too few data points at i = ", str(i))
        continue
    
    # Calculate the variance
    variance.append(np.var(these_data_points))
    episode_numbers_where_variance_was_calculated.append(episode_number[i])

# Plotting the variance over time
plt.figure()
plt.plot(episode_numbers_where_variance_was_calculated, variance)
plt.xlabel('Episode number')
plt.ylabel('Variance of raw combined angular rate [deg/s]')