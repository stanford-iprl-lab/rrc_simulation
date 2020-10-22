import numpy as np
from sympy import *
from rrc_simulation import trifinger_platform
from rrc_simulation.control.custom_pinocchio_utils import CustomPinocchioUtils

"""
Script to compute analytical Jacobian with sympy
"""

# Fixed values from URDF
theta_base = symbols("theta_base")

# joint origin xyz values
j1_xyz = [0, 0, 0] # joint 1
j2_xyz = [0.01685, 0.0505, 0]
j3_xyz = [0.04922, 0, -0.16]
j4_xyz = [0.0185, 0, -0.1626]

q1, q2, q3 = symbols("q1 q2 q3")

H_0_wrt_base = Matrix([
                    [cos(theta_base), -sin(theta_base), 0, 0],
                    [sin(theta_base), cos(theta_base), 0, 0],
                    [0, 0, 1, 0.29],
                    [0, 0, 0, 1],
                    ])

# Frame 1 w.r.t. frame 0
# Rotation around y axis
H_1_wrt_0 = Matrix([
                    [cos(q1),       0, sin(q1), j1_xyz[0]],
                    [0,             1,       0, j1_xyz[1]],
                    [-sin(q1),      0, cos(q1), j1_xyz[2]],
                    [0, 0, 0, 1],
                    ])

# Frame 2 w.r.t. frame 1
# Rotation around x axis
H_2_wrt_1 = Matrix([
                  [1,       0,        0, j2_xyz[0]],
                  [0, cos(q2), -sin(q2), j2_xyz[1]],
                  [0, sin(q2),  cos(q2), j2_xyz[2]],
                  [0,       0,        0,         1],
                  ])

# Frame 3 w.r.t. frame 2
# Rotation around x axis
H_3_wrt_2 = Matrix([
                  [1,       0,        0, j3_xyz[0]],
                  [0, cos(q3), -sin(q3), j3_xyz[1]],
                  [0, sin(q3),  cos(q3), j3_xyz[2]],
                  [0,       0,        0,         1],
                  ])

# Transformation from frame 3 to 4
# Fixed
H_4_wrt_3 = Matrix([
                  [1, 0, 0, j4_xyz[0]],
                  [0, 1, 0, j4_xyz[1]],
                  [0, 0, 1, j4_xyz[2]],
                  [0, 0, 0,         1],
                  ])

H_4_wrt_0 = H_0_wrt_base @ H_1_wrt_0 @ H_2_wrt_1 @ H_3_wrt_2 @ H_4_wrt_3

p = np.array([[0],[0],[0],[1]])

# Compute jacobian
eef_wf = H_4_wrt_0 @ p
eef_wf = eef_wf[0:3, :]
print("sympy eef_wf")
print(eef_wf)

dq1 = eef_wf.diff(q1)
dq2 = eef_wf.diff(q2)
dq3 = eef_wf.diff(q3)

J = dq1.row_join(dq2).row_join(dq3)
#print("sympy J")
#pprint(J)

############################################
# Test forward kinematics , and Jacobian for one finger
############################################
f_id = 2
theta_base_deg = -240 # Fixed andle of finger base w.r.t. center holder {0, 120, 240} degrees
theta_base_val = theta_base_deg * (np.pi/180)

q1_val = 0.5
q2_val = 0.9
q3_val = -1.7

q_all = np.array([q1_val, q2_val, q3_val,q1_val, q2_val, q3_val,q1_val, q2_val, q3_val])

p_wf = eef_wf.subs({"q1": q1_val, "q2": q2_val, "q3": q3_val, "theta_base": theta_base_val})

print("eef_wf analytical: {}".format(p_wf))

platform = trifinger_platform.TriFingerPlatform(
    visualization=False,
    enable_cameras=False,
    initial_robot_position=np.zeros(9),
    initial_object_pose = None
)
custom_pinocchio_utils = CustomPinocchioUtils(platform.simfinger.finger_urdf_path, platform.simfinger.tip_link_names) 
ft_wf_list = custom_pinocchio_utils.forward_kinematics(q_all)

print("eef wf pinocchio: {}".format(ft_wf_list))

Ji_pin = custom_pinocchio_utils.get_tip_link_jacobian(f_id, q_all)

print("Jacobian pinocchio:")
print(Ji_pin[:3, 3*f_id:3*f_id+3])

print("Jacobian columns:")
dq1_val = dq1.subs({"q1": q1_val, "q2": q2_val, "q3": q3_val, "theta_base": theta_base_val})
print(dq1_val)
dq2_val = dq2.subs({"q1": q1_val, "q2": q2_val, "q3": q3_val, "theta_base": theta_base_val})
print(dq2_val)
dq3_val = dq3.subs({"q1": q1_val, "q2": q2_val, "q3": q3_val, "theta_base": theta_base_val})
print(dq3_val)
