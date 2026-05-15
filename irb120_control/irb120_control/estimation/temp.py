import numpy as np

lever = np.array([0.02255492, -0.00158109, 0.29750716])

force = np.array([0.187, 0, 0])

obj_angle = np.deg2rad(-7.11)

# rotate the force into the object frame
R_obj = np.array([[np.cos(obj_angle), 0, np.sin(obj_angle)],
                  [0, 1, 0],
                  [-np.sin(obj_angle), 0, np.cos(obj_angle)]])

force_in_obj = R_obj @ -force

torque = np.cross(lever, force_in_obj)


print(f"Lever arm: {lever}")
print(f"Force: {force}")
print(f"At angle {np.rad2deg(obj_angle):.2f}°")
print(f"Force in object frame: {force_in_obj}")
print(f"Torque: {torque}")