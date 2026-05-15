# Testing backwards wrench model to see if the forces and torques in {O} are calculated correctly
#!/usr/bin/env python3

import os
import numpy as np
from irb120_control.estimation.com_estimation import model_bkwd_wrench, compute_applied_wrench
from irb120_control.estimation.helper_fns import rotvec_to_rot

def main():
    f_sens_0 = np.array([0.0, 0.0, 10]) # at initial squash only, no pulling yet
    f_sens_1 = np.array([1.0, 0.0, 10]) # at incipient pulling
    f_sens_n = np.array([0.1, 0.0, 10]) # after some time, pulling force is reduced because gravity resists less.

    w_sens_0 = np.hstack((f_sens_0, np.zeros(3))) # assume no measured torque for simplicity
    w_sens_1 = np.hstack((f_sens_1, np.zeros(3))) # assume no measured torque for simplicity
    w_sens_n = np.hstack((f_sens_n, np.zeros(3))) # assume no measured torque for simplicity

    w_sens = np.array([w_sens_0, w_sens_1, w_sens_n]) # shape (3, 6)

    # Stationary frames
    T_B_obj_0 = np.eye(4) # assume object frame is same as base frame for simplicity
    T_B_obj_0[:3, 3] = np.array([0.2, 0.0, 0.0]) # object is at the same location as the sensor for simplicity

    T_B_sensor_0 = np.eye(4) # assume sensor frame is same as base frame for simplicity
    T_B_sensor_0[:3, 3] = np.array([0.0, 0.0, 0.3]) # sensor is roughly 20 cm 'before' the object, and at the top of the object.

    T_B_finger_0 = np.eye(4) # assume finger frame is same as base frame for simplicity
    T_B_finger_0[:3, 3] = np.array([0.21, 0.0, 0.3]) # finger is slightly 'beyond' the object (towards the end-effector) and at the same height as the sensor.

    # And for the other points, just rotate the object frame and translate the sensor and finger frames
    T_B_obj_1 = T_B_obj_0.copy() # assume object frame is same as base frame for simplicity
    T_B_obj_1[:3, :3] = rotvec_to_rot(np.array([0, np.deg2rad(-1), 0])) # As soon as it starts tipping

    T_B_sensor_1 = T_B_sensor_0.copy() # assume sensor frame is same as base frame for simplicity
    T_B_sensor_1[:3, 3] += np.array([-0.01, 0.0, -0.01]) # immediately after initial tip, sensor translates -x and slightly -z

    T_B_finger_1 = T_B_finger_0.copy() # assume finger frame is same as base frame for simplicity
    T_B_finger_1[:3, 3] += np.array([-0.01, 0.0, -0.01]) # Finger is also slightly translated


    T_B_obj_n = T_B_obj_0.copy() # assume object frame is same as base frame for simplicity
    T_B_obj_n[:3, :3] = rotvec_to_rot(np.array([0, np.deg2rad(-11), 0])) # As soon as it starts tipping

    T_B_sensor_n = T_B_sensor_0.copy() # assume sensor frame is same as base frame for simplicity
    T_B_sensor_n[:3, 3] += np.array([-0.10, 0.0, -0.02]) # After more tipping, sensor has translated a bunch in -x and slightly -z

    T_B_finger_n = T_B_finger_0.copy() # assume finger frame is same as base frame for simplicity
    T_B_finger_n[:3, 3] += np.array([-0.10, 0.0, -0.02]) # Same for finger


    T_B_obj = np.array([T_B_obj_0, T_B_obj_1, T_B_obj_n]) # shape (3, 4, 4)
    T_B_sensor = np.array([T_B_sensor_0, T_B_sensor_1, T_B_sensor_n]) # shape (3, 4, 4)
    # T_B_finger = np.array([T_B_finger_0, T_B_finger_1, T_B_finger_n]) # shape (3, 4, 4)

    # Constant throughout the test, so just use the initial ones
    contact_pt_in_O = T_B_finger_0[:3, 3] - T_B_obj_0[:3, 3] # contact in {O} is finger position in {B} minus object position in {B}
    print(f"contact_pt_in_O = {contact_pt_in_O}")


    print(f"input wrench in Sensor frame:\n {w_sens}")
    
    
    w_app_O_bkwd = model_bkwd_wrench(w_sens, T_B_sensor, T_B_obj, contact_pt_in_O)
    print(f"Old wrench model: \n{w_app_O_bkwd}")


    Q_ft_0 = np.array([0, 0, 0, 1])
    Q_ft = np.array([Q_ft_0, Q_ft_0, Q_ft_0]) # assume no rotation between sensor and object frames for simplicity
    dummy = np.zeros(3,)
    pitch_contact = np.deg2rad(np.array([0.0, -1.0, -11.0])) # just for testing
    
    w_app_O = compute_applied_wrench(w_sens[:, :3], w_sens[:, 3:], Q_ft, dummy, pitch_contact, dummy, contact_pt_in_O)
    print(f"\nNew wrench model: \n{w_app_O}\n\n\n")

    return w_app_O



if __name__ == "__main__":
    w_app_O = main()
    # print(f"Applied wrench in object frame at initial squash (f_sens_0): {w_app_O}")