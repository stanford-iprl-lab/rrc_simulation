import numpy as np
import enum
from scipy.spatial.transform import Rotation
from scipy.spatial.distance import pdist, squareform

from .contact_point import ContactPoint
from rrc_simulation.tasks import move_cube
from rrc_simulation.traj_opt.fixed_contact_point_opt import FixedContactPointOpt
from rrc_simulation.traj_opt.static_object_opt import StaticObjectOpt

# Here, hard code the base position of the fingers (as angle on the arena)
r = 0.15
theta_0 = np.pi/2 # 90 degrees
theta_1 = -np.pi/3 # 330 degrees
theta_2 = 3.66519 # 210 degrees
CUBE_HALF_SIZE = move_cube._CUBE_WIDTH/2 + 0.008

FINGER_BASE_POSITIONS = [
                        np.array([[np.cos(theta_0)*r, np.sin(theta_0)*r, 0]]),
                        np.array([[np.cos(theta_1)*r, np.sin(theta_1)*r, 0]]),
                        np.array([[np.cos(theta_2)*r, np.sin(theta_2)*r, 0]]),
                        ]


class PolicyMode(enum.Enum):
    RESET = enum.auto()
    TRAJ_OPT = enum.auto()
    IMPEDANCE = enum.auto()
    RL_PUSH = enum.auto()
    RESIDUAL = enum.auto()

# Information about object faces given face_id
OBJ_FACES_INFO = {
                  1: {"center_param": np.array([0.,-1.,0.]),
                      "face_down_default_quat": np.array([0.707,0,0,0.707]),
                      "adjacent_faces": [6,4,3,5],
                      "opposite_face": 2,
                      "up_axis": np.array([0.,1.,0.]), # UP axis when this face is ground face
                      },
                  2: {"center_param": np.array([0.,1.,0.]),
                      "face_down_default_quat": np.array([-0.707,0,0,0.707]),
                      "adjacent_faces": [6,4,3,5],
                      "opposite_face": 1,
                      "up_axis": np.array([0.,-1.,0.]),
                      },
                  3: {"center_param": np.array([1.,0.,0.]),
                      "face_down_default_quat": np.array([0,0.707,0,0.707]),
                      "adjacent_faces": [1,2,4,6],
                      "opposite_face": 5,
                      "up_axis": np.array([-1.,0.,0.]),
                      },
                  4: {"center_param": np.array([0.,0.,1.]),
                      "face_down_default_quat": np.array([0,1,0,0]),
                      "adjacent_faces": [1,2,3,5],
                      "opposite_face": 6,
                      "up_axis": np.array([0.,0.,-1.]),
                      },
                  5: {"center_param": np.array([-1.,0.,0.]),
                      "face_down_default_quat": np.array([0,-0.707,0,0.707]),
                      "adjacent_faces": [1,2,4,6],
                      "opposite_face": 3,
                      "up_axis": np.array([1.,0.,0.]),
                      },
                  6: {"center_param": np.array([0.,0.,-1.]),
                      "face_down_default_quat": np.array([0,0,0,1]),
                      "adjacent_faces": [1,2,3,5],
                      "opposite_face": 4,
                      "up_axis": np.array([0.,0.,1.]),
                      },
                 }

"""
Compute joint torques to move fingertips to desired locations
Inputs:
tip_pos_desired_list: List of desired fingertip positions for each finger
q_current: Current joint angles
dq_current: Current joint velocities
tip_forces_wf: fingertip forces in world frame
tol: tolerance for determining when fingers have reached goal
"""
def impedance_controller(
                        tip_pos_desired_list,
                        q_current,  
                        dq_current,
                        custom_pinocchio_utils,
                        tip_forces_wf = None,
                        tol           = 0.008
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
                                                tol          = tol
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
tip_forces_wf: fingertip forces in world frame
tol: tolerance for determining when fingers have reached goal
"""
def impedance_controller_single_finger(
                                      finger_id,
                                      tip_desired,
                                      q_current,
                                      dq_current,
                                      custom_pinocchio_utils,
                                      tip_force_wf = None,
                                      tol          = 0.008
                                      ):
  Kp_x = 200
  Kp_y = 200
  Kp_z = 400
  Kp = np.diag([Kp_x, Kp_y, Kp_z])
  Kv_x = 7
  Kv_y = 7
  Kv_z = 7
  Kv = np.diag([Kv_x, Kv_y, Kv_z])

  # Compute current fingertip position
  x_current = custom_pinocchio_utils.forward_kinematics(q_current)[finger_id]

  delta_x = np.expand_dims(np.array(tip_desired) - np.array(x_current), 1)
  #print("Current x: {}".format(x_current))
  #print("Desired x: {}".format(tip_desired))
  #print(delta_x)
  
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
  
  tol = tol
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
def get_cp_wf_from_cp_param(cp_param, cube_pos_wf, cube_quat_wf, cube_half_size=CUBE_HALF_SIZE):
  cp = get_cp_of_from_cp_param(cp_param, cube_half_size)

  rotation = Rotation.from_quat(cube_quat_wf)
  translation = np.asarray(cube_pos_wf)

  return rotation.apply(cp.pos_of) + translation

"""
Get contact point positions in world frame from cp_params
"""
def get_cp_wf_list_from_cp_params(cp_params, cube_pos, cube_quat, cube_half_size=CUBE_HALF_SIZE):
  # Get contact points in wf
  fingertip_goal_list = []
  for i in range(cp_params.shape[0]):
    fingertip_goal_list.append(get_cp_wf_from_cp_param(cp_params[i], cube_pos, cube_quat, cube_half_size))
  return fingertip_goal_list

"""
Compute contact point position in object frame
Inputs:
cp_param: Contact point param [px, py, pz]
"""
def get_cp_of_from_cp_param(cp_param, cube_half_size=CUBE_HALF_SIZE):
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

"""
Get face id on cube, given cp_param
cp_param: [x,y,z]
"""
def get_face_from_cp_param(cp_param):
  x_param = cp_param[0]
  y_param = cp_param[1]
  z_param = cp_param[2]
  # For now, just hard code quat
  if y_param == -1:
    face = 1
  elif y_param == 1:
    face = 2
  elif x_param == 1:
    face = 3
  elif z_param == 1:
    face = 4
  elif x_param == -1:
    face = 5
  elif z_param == -1:
    face = 6

  return face

"""
Trasform point p from world frame to object frame, given object pose
"""
def get_wf_from_of(p, obj_pose):
  cube_pos_wf = obj_pose.position
  cube_quat_wf = obj_pose.orientation

  rotation = Rotation.from_quat(cube_quat_wf)
  translation = np.asarray(cube_pos_wf)
  
  return rotation.apply(p) + translation

"""
Trasform point p from object frame to world frame, given object pose
"""
def get_of_from_wf(p, obj_pose):
  cube_pos_wf = obj_pose.position
  cube_quat_wf = obj_pose.orientation

  rotation = Rotation.from_quat(cube_quat_wf)
  translation = np.asarray(cube_pos_wf)
  
  rotation_inv = rotation.inv()
  translation_inv = -rotation_inv.apply(translation)

  return rotation_inv.apply(p) + translation_inv

##############################################################################
# Lift mode functions
##############################################################################
"""
Run trajectory optimization
obj_pose: current object pose (for getting contact points)
current_position: current joint positions of robot
x0: object initial position for traj opt
x_goal: object goal position for traj opt
nGrid: number of grid points
dt: delta t
"""
def run_traj_opt(obj_pose, current_position, custom_pinocchio_utils, x0, x_goal, nGrid, dt, save_dir=None):
  init_fingertip_pos_list = [[],[],[]] # Containts 3 lists, one for each finger
  for finger_id in range(3):
    tip_current = custom_pinocchio_utils.forward_kinematics(current_position)[finger_id]
    init_fingertip_pos_list[finger_id].append(tip_current)

  # Get initial contact points and waypoints to them
  cp_params = get_lifting_cp_params(obj_pose, init_fingertip_pos_list)

  cube_shape = (move_cube._CUBE_WIDTH, move_cube._CUBE_WIDTH, move_cube._CUBE_WIDTH)
  cube_mass = 0.02 # TODO Hardcoded

  # Formulate and solve optimization problem
  opt_problem = FixedContactPointOpt(
                                    nGrid     = nGrid, # Number of timesteps
                                    dt        = dt,   # Length of each timestep (seconds)
                                    cp_params = cp_params,
                                    x0        = x0,
                                    x_goal    = x_goal,
                                    obj_shape = cube_shape,
                                    obj_mass  = cube_mass,
                                    )
  x_soln     = np.array(opt_problem.x_soln)
  l_wf_soln  = np.array(opt_problem.l_wf_soln)
  cp_params  = np.array(cp_params)

  # Save solution in npz file
  if save_dir is not None:
    save_string = "{}/trajectory".format(save_dir)
    np.savez(save_string,
             dt         = opt_problem.dt,
             x0         = x0,
             x_goal     = x_goal,
             t          = opt_problem.t_soln,
             x          = opt_problem.x_soln,
             dx         = opt_problem.dx_soln,
             l          = opt_problem.l_soln,
             l_wf       = opt_problem.l_wf_soln,
             cp_params  = np.array(cp_params),
             obj_shape  = cube_shape,
             obj_mass   = cube_mass,
             fnum       = opt_problem.system.fnum, 
             qnum       = opt_problem.system.qnum, 
             x_dim      = opt_problem.system.x_dim, 
             dx_dim     = opt_problem.system.dx_dim, 
            )
  return x_soln, l_wf_soln, cp_params
  

"""
Get initial contact points on cube
Assign closest cube face to each finger
Since we are lifting object, don't worry about wf z-axis, just care about wf xy-plane
"""
def get_lifting_cp_params(obj_pose, fingertip_pos_list):
  # face that is touching the ground
  ground_face = get_closest_ground_face(obj_pose)

  # Transform finger base positions to object frame
  fingertip_pos_list_of = []
  for f_wf in FINGER_BASE_POSITIONS:
    f_of = get_of_from_wf(f_wf, obj_pose)
    fingertip_pos_list_of.append(f_of)

  # Find distance from x axis and y axis, and store in xy_distances
  # Need some additional logic to prevent multiple fingers from being assigned to same face
  x_axis = np.array([1,0])
  y_axis = np.array([0,1])
  
  # Object frame axis corresponding to plane parallel to ground plane
  x_ind, y_ind = __get_parallel_ground_plane_xy(ground_face)
    
  xy_distances = np.zeros((3, 2)) # Row corresponds to a finger, columns are x and y axis distances
  for f_i, f_of in enumerate(fingertip_pos_list_of):
    point_in_plane = np.array([f_of[0,x_ind], f_of[0,y_ind]]) # Ignore dimension of point that's not in the plane
    x_dist = __get_distance_from_pt_2_line(x_axis, np.array([0,0]), point_in_plane)
    y_dist = __get_distance_from_pt_2_line(y_axis, np.array([0,0]), point_in_plane)
    
    xy_distances[f_i, 0] = x_dist
    xy_distances[f_i, 1] = y_dist

  # Do the face assignment - greedy approach (assigned closest fingers first)
  free_faces = OBJ_FACES_INFO[ground_face]["adjacent_faces"].copy() # List of face ids that haven't been assigned yet
  assigned_faces = np.zeros(3) 
  for i in range(3):
    # Find indices max element in array
    max_ind = np.unravel_index(np.argmax(xy_distances), xy_distances.shape)
    curr_finger_id = max_ind[0] 
    furthest_axis = max_ind[1]

    #print("current finger {}".format(curr_finger_id))
    #print(fingertip_pos_list_of[curr_finger_id])
    # Do the assignment
    x_dist = xy_distances[curr_finger_id, 0]
    y_dist = xy_distances[curr_finger_id, 1]
    if furthest_axis == 0: # distance to x axis is greater than to y axis
      if fingertip_pos_list_of[curr_finger_id][0, y_ind] > 0:
        face = OBJ_FACES_INFO[ground_face]["adjacent_faces"][1] # 2
      else:
        face = OBJ_FACES_INFO[ground_face]["adjacent_faces"][0] # 1
    else:
      if fingertip_pos_list_of[curr_finger_id][0, x_ind] > 0:
        face = OBJ_FACES_INFO[ground_face]["adjacent_faces"][2] # 3
      else:
        face = OBJ_FACES_INFO[ground_face]["adjacent_faces"][3] # 5
    #print("first choice face: {}".format(face))

    # Handle faces that may already be assigned
    if face not in free_faces:
      alternate_axis = abs(furthest_axis - 1)
      if alternate_axis == 0:
        if fingertip_pos_list_of[curr_finger_id][0, y_ind] > 0:
          face = OBJ_FACES_INFO[ground_face]["adjacent_faces"][1] # 2
        else:
          face = OBJ_FACES_INFO[ground_face]["adjacent_faces"][0] # 1
      else:
        if fingertip_pos_list_of[curr_finger_id][0, x_ind] > 0:
          face = OBJ_FACES_INFO[ground_face]["adjacent_faces"][2] # 3
        else:
          face = OBJ_FACES_INFO[ground_face]["adjacent_faces"][3] # 5
      #print("second choice face: {}".format(face))
    
    # If backup face isn't free, assign random face from free_faces
    if face not in free_faces:
      #print("random")
      #print(xy_distances[curr_finger_id, :])
      face = free_faces[0] 
    assigned_faces[curr_finger_id] = face 

    # Replace row with -np.inf so we can assign other fingers
    xy_distances[curr_finger_id, :] = -np.inf
    # Remove face from free_faces
    free_faces.remove(face)
  #print(assigned_faces)
  # Set contact point params
  cp_params = []
  for i in range(3):
    face = assigned_faces[i]
    param = OBJ_FACES_INFO[face]["center_param"].copy()
    #print(i)
    #print(param)
    cp_params.append(param)
  #print("assigning cp params for lifting")
  #print(cp_params)
  return cp_params

"""
Set up traj opt for fingers and static object
"""
def define_static_object_opt():
  cube_shape = (move_cube._CUBE_WIDTH, move_cube._CUBE_WIDTH, move_cube._CUBE_WIDTH)
  problem = StaticObjectOpt(
               nGrid     = 20,
               dt        = 0.1,
               obj_shape = cube_shape,
               )
  return problem

"""
Solve traj opt to get finger waypoints
"""
def get_waypoints(nlp, q_cur):
  # Get list of desired fingertip positions
  ft_goal = np.array([0.08457, 0.016751647828266603, 0.1977209510032231,-0.02777764742520991, -0.08161559231227206, 0.1977209510032231,-0.0567923525742952, 0.06486394448412161, 0.1977209510032231])

  nlp.solve_nlp(ft_goal, q_cur)

  ft_pos = nlp.ft_pos_soln
  return ft_pos

"""
Get waypoints to initial contact point on object
For now, we assume that contact points are always in the center of cube face
Return waypoints in world frame
Inputs:
cp_param: target contact point param
fingertip_pos: fingertip start position in world frame
"""
def get_waypoints_to_cp_param(obj_pose, fingertip_pos, cp_param, cube_half_size=CUBE_HALF_SIZE):
  # Get ground face
  ground_face = get_closest_ground_face(obj_pose)
  # Transform finger tip positions to object frame
  fingertip_pos_of = np.squeeze(get_of_from_wf(fingertip_pos, obj_pose))

  waypoints = []
  if cp_param is not None:
    # Transform cp_param to object frame
    cp = get_cp_of_from_cp_param(cp_param, cube_half_size=CUBE_HALF_SIZE)
    cp_pos_of = cp.pos_of

    tol = 0.05

    # Get the non-zero cp_param dimension (to determine which face the contact point is on)
    # This works because we assume z is always 0, and either x or y is 0
    non_zero_dim = np.argmax(abs(cp_param))
    zero_dim = abs(1-non_zero_dim)

    # Work with absolute values, and then correct sign at the end
    w = np.expand_dims(fingertip_pos_of,0)
    w[0,:] = 0.07 * OBJ_FACES_INFO[ground_face]["up_axis"] # Bring fingers lower, to avoid links colliding with each other
    if abs(fingertip_pos_of[non_zero_dim]) < abs(cp_pos_of[non_zero_dim]) + tol:
      w[0,non_zero_dim] = cp_param[non_zero_dim] * (abs(cp_pos_of[non_zero_dim]) + tol) # fix sign
    if abs(fingertip_pos_of[zero_dim]) < abs(cp_pos_of[zero_dim]) + tol:
      w[0,zero_dim] = cp_param[zero_dim] * (abs(cp_pos_of[zero_dim]) + tol) # fix sign
    #print(w)
    waypoints.append(w.copy())

    # Align zero_dim 
    w[0,zero_dim] = 0
    waypoints.append(w.copy())

    #w[0,non_zero_dim] = cp_pos_of[non_zero_dim]
    #w[0,2] = 0
    #waypoints.append(w.copy())
    waypoints.append(cp_pos_of)
  else:
    w = np.expand_dims(fingertip_pos_of,0)
    waypoints.append(w.copy())
    waypoints.append(w.copy())
    waypoints.append(w.copy())

  # Transform waypoints from object frame to world frame
  waypoints_wf = []
  #waypoints_wf.append(fingertip_pos)
  for wp in waypoints:
    wp_wf = np.squeeze(get_wf_from_of(wp, obj_pose))
    # If world frame z coord in less than 0, clip this to 0.01
    if wp_wf[2] <= 0:
        wp_wf[2] = 0.01
    waypoints_wf.append(wp_wf)

  #return waypoints_wf
  # Add intermediate waypoints
  interp_num = 10
  waypoints_final = []
  for i in range(len(waypoints_wf) - 1):
    curr_w = waypoints_wf[i]
    next_w = waypoints_wf[i+1] 

    interp_pts = np.linspace(curr_w, next_w, interp_num)
    for r in range(interp_num):
      waypoints_final.append(interp_pts[r])

  #waypoints_final.pop(-1)

  return waypoints_final

##############################################################################
# Lift mode functions
##############################################################################

"""
Determine face that is closest to ground
"""
def get_closest_ground_face(obj_pose):
  min_z = np.inf
  min_face = None
  for i in range(1,7):
    c = OBJ_FACES_INFO[i]["center_param"].copy()
    c_wf = get_wf_from_of(c, obj_pose)
    if c_wf[2] < min_z:
      min_z = c_wf[2]
      min_face = i

  return min_face

"""
Get flipping contact points
"""
def get_flipping_cp_params(
                          init_pose,
                          goal_pose,
                          cube_half_size=CUBE_HALF_SIZE,
                          ):
  # Get goal face
  init_face = get_closest_ground_face(init_pose)
  #print("Init face: {}".format(init_face))
  # Get goal face
  goal_face = get_closest_ground_face(goal_pose)
  #print("Goal face: {}".format(goal_face))
  
  if goal_face not in OBJ_FACES_INFO[init_face]["adjacent_faces"]:
    #print("Goal face not adjacent to initial face")
    goal_face = OBJ_FACES_INFO[init_face]["adjacent_faces"][0]
    #print("Intermmediate goal face: {}".format(goal_face))

  # Common adjacent faces to init_face and goal_face
  common_adjacent_faces = list(set(OBJ_FACES_INFO[init_face]["adjacent_faces"]). intersection(OBJ_FACES_INFO[goal_face]["adjacent_faces"]))

  opposite_goal_face = OBJ_FACES_INFO[goal_face]["opposite_face"]
  
  #print("place fingers on faces {}, towards face {}".format(common_adjacent_faces, opposite_goal_face))

  # Find closest fingers to each of the common_adjacent_faces
  # Transform finger tip positions to object frame
  finger_base_of = []
  for f_wf in FINGER_BASE_POSITIONS:
    f_of = get_of_from_wf(f_wf, init_pose)
    #f_of = np.squeeze(get_of_from_wf(f_wf, init_pose))
    finger_base_of.append(f_of)

  # Object frame axis corresponding to plane parallel to ground plane
  x_ind, y_ind = __get_parallel_ground_plane_xy(init_face)
  # Find distance from x axis and y axis, and store in xy_distances
  x_axis = np.array([1,0])
  y_axis = np.array([0,1])
    
  xy_distances = np.zeros((3, 2)) # Row corresponds to a finger, columns are x and y axis distances
  for f_i, f_of in enumerate(finger_base_of):
    point_in_plane = np.array([f_of[0,x_ind], f_of[0,y_ind]]) # Ignore dimension of point that's not in the plane
    x_dist = __get_distance_from_pt_2_line(x_axis, np.array([0,0]), point_in_plane)
    y_dist = __get_distance_from_pt_2_line(y_axis, np.array([0,0]), point_in_plane)
    
    xy_distances[f_i, 0] = np.sign(f_of[0,y_ind]) * x_dist
    xy_distances[f_i, 1] = np.sign(f_of[0,x_ind]) * y_dist

  finger_assignments = {}
  for face in common_adjacent_faces:
    face_ind = OBJ_FACES_INFO[init_face]["adjacent_faces"].index(face)
    if face_ind in [2,3]:
      # Check y_ind column for finger that is furthest away
      if OBJ_FACES_INFO[face]["center_param"][x_ind] < 0: 
        # Want most negative value
        f_i = np.nanargmin(xy_distances[:,1])
      else:
        # Want most positive value
        f_i = np.nanargmax(xy_distances[:,1])
    else:
      # Check x_ind column for finger that is furthest away
      if OBJ_FACES_INFO[face]["center_param"][y_ind] < 0: 
        f_i = np.nanargmin(xy_distances[:,0])
      else:
        f_i = np.nanargmax(xy_distances[:,0])
    finger_assignments[face] = f_i
    xy_distances[f_i, :] = np.nan

  cp_params = [None, None, None]
  height_param = -0.65 # Always want cps to be at this height
  width_param = 0.65 
  for face in common_adjacent_faces:
    param = OBJ_FACES_INFO[face]["center_param"].copy()
    param += OBJ_FACES_INFO[OBJ_FACES_INFO[init_face]["opposite_face"]]["center_param"] * height_param
    param += OBJ_FACES_INFO[opposite_goal_face]["center_param"] * width_param
    cp_params[finger_assignments[face]] = param
    #cp_params.append(param)
  #print("Assignments: {}".format(finger_assignments))
  return cp_params, init_face, goal_face

"""
Get next waypoint for flipping
"""
def get_flipping_waypoint(
                          obj_pose,
                          init_face,
                          goal_face,
                          fingertips_current_wf,
                          fingertips_init_wf,
                          cp_params,
                         ):
  # Get goal face
  #goal_face = get_closest_ground_face(goal_pose)
  #print("Goal face: {}".format(goal_face))
  #print("ground face: {}".format(get_closest_ground_face(obj_pose)))

  ground_face = get_closest_ground_face(obj_pose)
  #if (get_closest_ground_face(obj_pose) == goal_face):
  #  # Move fingers away from object
  #  return fingertips_init_wf

  # Transform current fingertip positions to of
  fingertips_new_wf = []

  incr = 0.01
  for f_i in range(3):
    f_wf = fingertips_current_wf[f_i]
    if cp_params[f_i] is None:
      f_new_wf = fingertips_init_wf[f_i]
    else:
      # Get face that finger is one
      face = get_face_from_cp_param(cp_params[f_i])
      f_of = get_of_from_wf(f_wf, obj_pose)

      if ground_face == goal_face:
        # Release object
        f_new_of = f_of - 0.01 * OBJ_FACES_INFO[face]["up_axis"]
        if obj_pose.position[2] <= 0.034: # TODO: HARDCODED
          flip_done = True
          #print("FLIP SUCCESSFUL!")
        else:
          flip_done = False
      elif ground_face != init_face:
        # Ground face does not match goal force or init face, give up
        f_new_of = f_of - 0.01 * OBJ_FACES_INFO[face]["up_axis"]
        if obj_pose.position[2] <= 0.034: # TODO: HARDCODED
          flip_done = True
        else:
          flip_done = False
      else:
        # Increment up_axis of f_of
        f_new_of = f_of + incr * OBJ_FACES_INFO[ground_face]["up_axis"]
        flip_done = False

      # Convert back to wf
      f_new_wf = get_wf_from_of(f_new_of, obj_pose)

    fingertips_new_wf.append(f_new_wf)

  #print(fingertips_current_wf)
  #print(fingertips_new_wf)
  #fingertips_new_wf[2] = fingertips_init_wf[2]
   
  return fingertips_new_wf, flip_done

##############################################################################
# Private functions
##############################################################################

"""
Given a ground face id, get the axes that are parallel to the floor
"""
def __get_parallel_ground_plane_xy(ground_face):
  if ground_face in [1,2]:
    x_ind = 0
    y_ind = 2
  if ground_face in [3,5]:
    x_ind = 2
    y_ind = 1
  if ground_face in [4,6]:
    x_ind = 0
    y_ind = 1
  return x_ind, y_ind

"""
Get distance from point to line (in 2D)
Inputs:
a, b: points on line
p: standalone point, for which we want to compute its distance to line
"""
def __get_distance_from_pt_2_line(a, b, p):
  a = np.squeeze(a)
  b = np.squeeze(b)
  p = np.squeeze(p)

  ba = b - a
  ap = a - p
  c = ba * (np.dot(ap,ba) / np.dot(ba,ba))
  d = ap - c
  
  return np.sqrt(np.dot(d,d))

