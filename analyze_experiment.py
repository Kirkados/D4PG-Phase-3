""" This script loads the deep guidance data logged from an experiment (specifically, logged by use_deep_guidance_arm.py)
renders a few plots, and animates the motion.

It should be run from the folder where use_deep_guidance_arm.py was run from for the given experiment.
 """
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from pyvirtualdisplay import Display # for rendering

# import code # for debugging
#code.interact(local=dict(globals(), **locals())) # Ctrl+D or Ctrl+Z to continue execution
try:    
    from settings import Settings
except:
    print("\nYou must use the manipulator environment in settings.py\n\nQuitting")
    raise SystemExit

assert Settings.ENVIRONMENT == 'manipulator'

environment_file = __import__('environment_' + Settings.ENVIRONMENT) # importing the environment

def make_C_bI(angle):        
    C_bI = np.array([[ np.cos(angle), np.sin(angle)],
                     [-np.sin(angle), np.cos(angle)]]) # [2, 2]        
    return C_bI



# Generate a virtual display for plotting
display = Display(visible = False, size = (1400,900))
display.start()

#####################################
### Load in the experimental data ###
#####################################
log_filename = glob.glob('*46-55.txt')[0]
data = np.load(log_filename)
print("Data file %s is loaded" %log_filename)
os.makedirs(log_filename.split('.')[0], exist_ok=True)

########################
### Plot some things ###
########################
time_log = data[:,0]
deep_guidances_ax = data[:,1]
deep_guidances_ay = data[:,2]
deep_guidances_alpha = data[:,3]
deep_guidances_shoulder = data[:,4]
deep_guidances_elbow = data[:,5]
deep_guidances_wrist = data[:,6]


plt.figure()
plt.plot(time_log, deep_guidances_ax)
plt.plot(time_log, deep_guidances_ay)
plt.savefig(log_filename.split('.')[0] + "/Acceleration Commands.png")
print("Saved acceleration commands figure")

plt.figure()
plt.plot(time_log, deep_guidances_alpha)
plt.savefig(log_filename.split('.')[0] + "/Angular Acceleration commands.png")
print("Saved angular acceleration commands figure")

plt.figure()
plt.plot(time_log, deep_guidances_shoulder)
plt.plot(time_log, deep_guidances_elbow)
plt.plot(time_log, deep_guidances_wrist)
plt.savefig(log_filename.split('.')[0] + "/Arm Acceleration commands.png")
print("Saved arm angular acceleration commands figure")



##########################
### Animate the motion ###
##########################
# Generate an Environment to use for reward logging
environment = environment_file.Environment()
environment.reset(False)

# Process the data. Need to make the raw total state log
# [relative_x, relative_y, relative_vx, relative_vy, 
#relative_angle, relative_angular_velocity, chaser_x, chaser_y, chaser_theta, 
#target_x, target_y, target_theta, chaser_vx, chaser_vy, chaser_omega, 
#target_vx, target_vy, target_omega] *# Relative pose expressed in the chaser's body frame; everythign else in Inertial frame #*

print("Rendering animation...", end='')
raw_total_state_log = []
cumulative_reward_log = []
action_log = []
cumulative_rewards = 0
SPOTNet_previous_relative_x = 0.0
are_we_done = False
timestep_where_docking_occurred = -1
for i in range(len(data)):
    Pi_time, deep_guidance_Ax, deep_guidance_Ay, deep_guidance_alpha_base, \
                                 deep_guidance_alpha_shoulder, deep_guidance_alpha_elbow, deep_guidance_alpha_wrist, \
                                 Pi_red_x, Pi_red_y, Pi_red_theta, \
                                 Pi_red_Vx, Pi_red_Vy, Pi_red_omega,        \
                                 Pi_black_x, Pi_black_y, Pi_black_theta,    \
                                 Pi_black_Vx, Pi_black_Vy, Pi_black_omega,  \
                                 shoulder_theta, elbow_theta, wrist_theta, \
                                 shoulder_omega, elbow_omega, wrist_omega, docked = data[i,:]

    # Raw total state log
    # [self.chaser_position, self.chaser_velocity, self.arm_angles, self.arm_angular_rates, self.target_position, self.target_velocity, self.end_effector_position, self.end_effector_velocity, self.relative_position_body, self.relative_angle, self.end_effector_position_body, self.end_effector_velocity_body]     
    raw_total_state_log.append([Pi_red_x, Pi_red_y, Pi_red_theta, Pi_red_Vx, Pi_red_Vy, Pi_red_omega, shoulder_theta, elbow_theta, wrist_theta, shoulder_omega, elbow_omega, wrist_omega, Pi_black_x, Pi_black_y, Pi_black_theta, Pi_black_Vx, Pi_black_Vy, Pi_black_omega, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
       
    # Check the reward function based off this state
    environment.chaser_position   = np.array([Pi_red_x, Pi_red_y, Pi_red_theta])
    environment.chaser_velocity   = np.array([Pi_red_Vx, Pi_red_Vy, Pi_red_omega])
    environment.target_position   = np.array([Pi_black_x, Pi_black_y, Pi_black_theta])
    environment.target_velocity   = np.array([Pi_black_Vx, Pi_black_Vy, Pi_black_omega])
    environment.arm_angles        = np.array([shoulder_theta, elbow_theta, wrist_theta])
    environment.arm_angular_rates = np.array([shoulder_omega, elbow_omega, wrist_omega])
    
    # Get environment to check for collisions
    environment.update_end_effector_and_docking_locations()
    environment.update_end_effector_location_body_frame()
    environment.update_relative_pose_body_frame()
    environment.check_collisions()
    rewards_this_timestep = environment.reward_function(0)

    # Only add rewards if we aren't done
    if not are_we_done:
        cumulative_rewards += rewards_this_timestep
        if environment.is_done(): # If we are done, this was the last reward that we added
            are_we_done = True
            timestep_where_docking_occurred = i
    cumulative_reward_log.append(cumulative_rewards)
    
    action_log.append([deep_guidance_Ax, deep_guidance_Ay, deep_guidance_alpha_base, deep_guidance_alpha_shoulder, deep_guidance_alpha_elbow, deep_guidance_alpha_wrist])

# Render the episode
environment_file.render(np.asarray(raw_total_state_log), np.asarray(action_log), 0, np.asarray(cumulative_reward_log), 0, 0, 0, 0, 0, 1, log_filename.split('.')[0], '', time_log, timestep_where_docking_occurred)
print("Done!")
# Close the display
del environment
plt.close()
display.stop()