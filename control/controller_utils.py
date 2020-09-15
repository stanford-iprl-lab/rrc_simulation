import numpy as np
from scipy.spatial.transform import Rotation

from contact_point import ContactPoint
from rrc_simulation.tasks import move_cube

"""
Compute joint torques to move fingertips to desired locations
Inputs:
tip_pos_desired_list: List of desired fingertip positions for each finger
q_current: Current joint angles
dq_current: Current joint velocities
"""
def impedance_controller(
                        tip_pos_desired_list,
                        q_current,  
                        dq_current,
                        custom_pinocchio_utils,
                        tip_forces_wf = None
                        ):
  torque = 0
  goal_reached = True
  for finger_id in range(3):
    # Get contact forces for single finger
    if tip_forces_wf is None:
      f_wf = None
    else:
      f_wf = np.expand_dims(np.array(tip_forces_wf[finger_id * 3:finger_id*3 + 3]),1)
    finger_torque, finger_goal_reached = impedance_controller_single_finger(
                                                finger_id,
                                                tip_pos_desired_list[finger_id],
                                                q_current,
                                                dq_current,
                                                custom_pinocchio_utils,
                                                tip_force_wf = f_wf,
                                                )
    goal_reached = goal_reached and finger_goal_reached
    torque += finger_torque
  return torque, goal_reached

"""
Compute joint torques to move fingertip to desired location
Inputs:
finger_id: Finger 0, 1, or 2
tip_desired: Desired fingertip pose **ORIENTATION??**
  for orientation: transform fingertip reference frame to world frame (take into account object orientation)
  for now, just track position
q_current: Current joint angles
dq_current: Current joint velocities
"""
def impedance_controller_single_finger(
                                      finger_id,
                                      tip_desired,
                                      q_current,
                                      dq_current,
                                      custom_pinocchio_utils,
                                      tip_force_wf = None,
                                      ):
  Kp_x = 130
  Kp_y = 130
  Kp_z = 300
  Kp = np.diag([Kp_x, Kp_y, Kp_z])
  Kv_x = Kv_y = 6
  Kv_z = 7
  Kv = np.diag([Kv_x, Kv_y, Kv_z])

  # Compute current fingertip position
  x_current = custom_pinocchio_utils.forward_kinematics(q_current)[finger_id]

  delta_x = np.expand_dims(np.array(tip_desired) - np.array(x_current), 1)
  #print("Current x: {}".format(x_current))
  #print("Desired x: {}".format(tip_desired))
  
  # Get full Jacobian for finger
  Ji = custom_pinocchio_utils.get_tip_link_jacobian(finger_id, q_current)
  # Just take first 3 rows, which correspond to linear velocities of fingertip
  Ji = Ji[:3, :]

  # Get current fingertip velocity
  dx_current = Ji @ np.expand_dims(np.array(dq_current), 1)

  if tip_force_wf is not None:
    torque = np.squeeze(Ji.T @ (Kp @ delta_x - Kv @ dx_current) + Ji.T @ tip_force_wf)
  else:
    torque = np.squeeze(Ji.T @ (Kp @ delta_x - Kv @ dx_current))
  
  tol = 0.008
  goal_reached = (np.linalg.norm(delta_x) < tol)
  #print("Finger {} delta".format(finger_id))
  #print(np.linalg.norm(delta_x))
  #print(goal_reached)
  return torque, goal_reached

"""
Compute contact point position in world frame
Inputs:
cp_param: Contact point param [px, py, pz]
cube: Block object, which contains object shape info
"""
def get_cp_wf_from_cp_param(cp_param, cube_pos_wf, cube_quat_wf, cube_half_size):
  cp = get_cp_of_from_cp_param(cp_param, cube_half_size)

  rotation = Rotation.from_quat(cube_quat_wf)
  translation = np.asarray(cube_pos_wf)

  return rotation.apply(cp.pos_of) + translation

"""
Compute contact point position in object frame
Inputs:
cp_param: Contact point param [px, py, pz]
cube: Block object, which contains object shape info
"""
def get_cp_of_from_cp_param(cp_param, cube_half_size):
  obj_shape = (cube_half_size, cube_half_size, cube_half_size)
  cp_of = []
  # Get cp position in OF
  for i in range(3):
    cp_of.append(-obj_shape[i] + (cp_param[i]+1)*obj_shape[i])

  cp_of = np.asarray(cp_of)

  x_param = cp_param[0]
  y_param = cp_param[1]
  z_param = cp_param[2]
  # For now, just hard code quat
  if y_param == -1:
    quat = (np.sqrt(2)/2, 0, 0, np.sqrt(2)/2)
  elif y_param == 1:
    quat = (np.sqrt(2)/2, 0, 0, -np.sqrt(2)/2)
  elif x_param == 1:
    quat = (0, 0, 1, 0)
  elif z_param == 1:
    quat = (np.sqrt(2)/2, 0, np.sqrt(2)/2, 0)
  elif x_param == -1:
    quat = (1, 0, 0, 0)
  elif z_param == -1:
    quat = (np.sqrt(2)/2, 0, -np.sqrt(2)/2, 0)

  cp = ContactPoint(cp_of, quat)
  return cp

def get_of_from_wf(p, obj_pose):
  cube_pos_wf = obj_pose.position
  cube_quat_wf = obj_pose.orientation

  rotation = Rotation.from_quat(cube_quat_wf)
  translation = np.asarray(cube_pos_wf)
  
  rotation_inv = rotation.inv()
  translation_inv = -rotation_inv.apply(translation)

  return rotation_inv.apply(p) + translation_inv

"""
Get initial contact points on cube
Assign closest cube face to each finger
For now, don't worry about z-axis, just care about xy plane
"""
def get_initial_cp_params(obj_pose, cube_half_size, fingertip_pos_list):
  # Transform finger tip positions to object frame

  fingertip_pos_list_wf = []
  for f_of in fingertip_pos_list:
    f_wf = get_of_from_wf(f_of, obj_pose)
    fingertip_pos_list_wf.append(f_wf)

  # Find distance from x axis and y axis, and store in xy_distances
  # Need some additional logic to prevent multiple fingers from being assigned to same face
  x_axis = np.array([1,0])
  y_axis = np.array([0,1])

  xy_distances = np.zeros((3, 2)) # Row corresponds to a finger, columns are x and y axis distances
  for f_i, f_wf in enumerate(fingertip_pos_list_wf):
    x_dist = get_distance_from_pt_2_line(x_axis, np.array([0,0]), f_wf[0,0:2])
    y_dist = get_distance_from_pt_2_line(y_axis, np.array([0,0]), f_wf[0,0:2])
    
    xy_distances[f_i, 0] = x_dist
    xy_distances[f_i, 1] = y_dist

  # Do the face assignment - greedy approach (assigned closest fingers first)
  free_faces = [1,2,3,5] # List of face ids that haven't been assigned yet
  assigned_faces = np.zeros(3) 
  for i in range(3):
    # Find indices max element in array
    max_ind = np.unravel_index(np.argmax(xy_distances), xy_distances.shape)
    curr_finger_id = max_ind[0] 
    furthest_axis = max_ind[1]

    # Do the assignment
    x_dist = xy_distances[curr_finger_id, 0]
    y_dist = xy_distances[curr_finger_id, 1]
    if furthest_axis == 0: # distance to x axis is greater than to y axis
      if fingertip_pos_list_wf[curr_finger_id][0, 1] > 0:
        face = 2
      else:
        face = 1
    else:
      if fingertip_pos_list_wf[curr_finger_id][0, 0] > 0:
        face = 3
      else:
        face = 5

    # Handle faces that may already be assigned
    if face not in free_faces:
      alternate_axis = abs(furthest_axis - 1)
      if furthest_axis == 0: # distance to x axis is greater than to y axis
        if fingertip_pos_list_wf[curr_finger_id][0, 1] > 0:
          face = 2
        else:
          face = 1
      else:
        if fingertip_pos_list_wf[curr_finger_id][0, 0] > 0:
          face = 3
        else:
          face = 5
    
    # If backup face isn't free, assign random face from free_faces
    if face not in free_faces:
      face = free_faces[0] 

    assigned_faces[curr_finger_id] = face 

    # Replace row with -np.inf so we can assign other fingers
    xy_distances[curr_finger_id, :] = -np.inf
    # Remove face from free_faces
    free_faces.remove(face)

  # Set contact point params
  cp_params = []
  for i in range(3):
    face = assigned_faces[i]
    if face == 1:
      param = [0, -1, 0]
    elif face == 2:
      param = [0, 1, 0]
    elif face == 3:
      param = [1, 0, 0]
    elif face == 5:
      param = [-1, 0, 0]
    cp_params.append(param)

  return cp_params
"""
Get distance from point to line (in 2D)
Inputs:
a, b: points on line
p: standalone point, for which we want to compute its distance to line
"""
def get_distance_from_pt_2_line(a, b, p):
  a = np.squeeze(a)
  b = np.squeeze(b)
  p = np.squeeze(p)

  ba = b - a
  ap = a - p
  c = ba * (np.dot(ap,ba) / np.dot(ba,ba))
  d = ap - c
  
  return np.sqrt(np.dot(d,d))
