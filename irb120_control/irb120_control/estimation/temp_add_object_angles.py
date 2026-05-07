import numpy as np

data_squash = np.load("/home/hylander2126/Documents/github/ros2_irb120/runtime_logs/squash_pull/most_recent.npz")

# np.load returns an NpzFile object (which acts like a dictionary).
# Converting it directly to an array doesn't work. Instead, extract the arrays into a dictionary.
squash_dict = dict(data_squash)
print("Squash keys:", squash_dict.keys())
print("Squash fx array:", squash_dict['fx'][:5]) # print first 5 elements

# Angles are simulated from known experimental timestamps (absolute ROS time).
# Contact start ~465s, inflection (theta*) at 480s = 18.5 deg, retract end ~494s.
# NOTE: log object angle directly in future experiments.
# t = squash_dict['ft_time_s']
# T_CONTACT_START = 1777663465.0
# T_INFLECTION    = 1777663480.0
# T_CONTACT_END   = 1777663494.0
# THETA_STAR_DEG  = 18.5

# angles = np.zeros(len(t))
# tip_mask     = (t >= T_CONTACT_START) & (t <= T_INFLECTION)
# retract_mask = (t >  T_INFLECTION)    & (t <= T_CONTACT_END)
# angles[tip_mask]     = np.linspace(0, np.deg2rad(THETA_STAR_DEG), tip_mask.sum())
# angles[retract_mask] = np.linspace(np.deg2rad(THETA_STAR_DEG), 0, retract_mask.sum())
# print(f'Angle range (deg): {np.rad2deg(angles.min()):.2f} -> {np.rad2deg(angles.max()):.2f}')

# squash_dict['payload_angle'] = angles
# np.savez("/home/hylander2126/Documents/github/ros2_irb120/runtime_logs/squash_pull/most_recent.npz", **squash_dict)
