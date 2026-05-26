import numpy as np

from irb120_control.estimation.com_estimation import model_bkwd_wrench
from irb120_control.estimation.helper_fns import rotvec_to_rot

def main():
    # MAJOR CHANGE: w_S is [t, f] order LIKE MODERN ROBOTICS AND ADT function!!
    w_S = np.array([[0, -10, 0, 0, 0, 10],
                    [0, -11, 0, 2, 0, 10]])  # (6,) wrench in sensor frame {S}

    # Assume 1m long finger for testing simplicity
    angle_0 = np.array([0, 0, 0]) # rad at rest
    angle_1 = np.array([0, -0.1, 0]) # rad after pulling for a short time

    T_sensor = np.array([np.eye(4), np.eye(4)])  # identity transform for testing
    T_obj = T_sensor.copy()  # identity transform for testing
    T_sensor[0, :3, 3] = np.array([-1,   0, 1]) # 1 meter above pivot edge (no torque should exist)

    T_sensor[1, :3, :3] = rotvec_to_rot(angle_1)
    T_sensor[1, :3, 3] = np.array([-1.1, 0, 0.9]) # after pulling for a short time

    T_obj[0, :3, 3] = np.array([0, 0, 0]) # at origin

    T_obj[1, :3, :3] = rotvec_to_rot(angle_1)
    T_obj[1, :3, 3] = np.array([0, 0, 0]) # still at origin, just rotated

    w_app = model_bkwd_wrench(w_S, T_sensor, T_obj)

    for i in range(2):
        print(f"w_S: {w_S[i]}")
        print("w_app:", w_app[i], "\n")

if __name__ == "__main__":
    main()