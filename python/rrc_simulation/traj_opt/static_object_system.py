import numpy as np
from casadi import *
import pybullet

from rrc_simulation.tasks import move_cube
from rrc_simulation.traj_opt import utils
from rrc_simulation.control.custom_pinocchio_utils import CustomPinocchioUtils

"""
Fingers and static object
For collision avoidance
"""
class StaticObjectSystem:

  def __init__(self,
               finger_urdf_path,
               tip_link_names,
               nGrid     = 100,
               dt        = 0.1,
               cp_params = None,
               obj_shape = None,
               obj_mass  = None,
               log_file  = None,
              ):
    print("Initialize static object system")
    
    # Time parameters
    self.nGrid = nGrid
    self.dt = dt
    self.tf = dt * (nGrid-1) # Final time

    self.fnum = 3
    self.qnum = 3
    self.obj_dof = 6
    self.x_dim = 7 # Dimension of object pose
    self.dx_dim = 6 # Dimension of object twist 

    self.p = 100

    self.obj_shape = obj_shape # (width, length, height), (x, y, z)
    self.obj_mass = obj_mass
    self.obj_mu = 1

    self.gravity = -10
    
    self.log_file = log_file
    self.theta_base_deg_list = [0, -120, -240] # List of angles of finger bases around arena (degrees)

    # Instantiate custom pinocchio utils class for access to Jacobian
    self.custom_pinocchio_utils = CustomPinocchioUtils(finger_urdf_path, tip_link_names) 
################################################################################
# Decision variable management helper functions
################################################################################

  """ 
  Define decision variables
  t     : time
  s_flat: state [q, dq] (flattened vector)
  """
  def dec_vars(self): 
    qnum = self.qnum
    fnum = self.fnum
    nGrid = self.nGrid

    # time
    t  = SX.sym("t" ,nGrid)

    # joint configurations for each joint in hand
    # one row of q is [finger1_q1, finger1_q2, finger1_q3, ...., fingerN_q3] 
    q  = SX.sym("q" ,nGrid,qnum*fnum)

    # object velocity at every timestep
    dq = SX.sym("dq",nGrid,qnum*fnum)

    # Flatten vectors
    s_flat = self.s_pack(q,dq)

    return t,s_flat

  """
  Pack the decision variables into a single horizontal vector
  """
  def decvar_pack(self,t,s):
    z = vertcat(t,s)
    return z

  """
  Unpack the decision variable vector z into:
  t: times (nGrid x 1) vector
  s: packed state vector
  u: packed u vector (joint torques)
  l: packed l vector (contact forces)
  a: slack variables
  """
  def decvar_unpack(self,z):
    qnum = self.qnum
    fnum = self.fnum
    nGrid = self.nGrid
    x_dim = self.x_dim
    dx_dim = self.dx_dim
  
    t = z[:nGrid]
  
    s_start_ind = nGrid
    s_flat = z[s_start_ind]

    return t,s_flat

  """
  Unpack the state vector s into q and dq
  """
  def s_unpack(self,s):
    nGrid = self.nGrid

    # total dof in hand
    dim = self.qnum * self.fnum

    # Get object pose
    q_flat  = s[:nGrid*dim]
    q = reshape(q_flat,dim,nGrid).T

    # Get object twist
    dq_flat = s[nGrid*dim:]
    dq = reshape(dq_flat,dim,nGrid).T

    return q,dq

  """
  Pack the state vector s into a single horizontal vector
  State:
  """
  def s_pack(self,q,dq):
    nGrid = self.nGrid

    # total dof in hand
    dim = self.qnum * self.fnum

    q_flat = reshape(q.T,nGrid*dim,1)
    dq_flat = reshape(dq.T,nGrid*dim,1)

    return vertcat(q_flat,dq_flat)

################################################################################
# End of decision variable help functions
################################################################################

################################################################################
# Constraint functions
################################################################################

  """
  Compute system dynamics (ds/dt):
  s_flat: state vector
  l_flat: contact forces

  Return:
  Derivative of state, ds, as a flattened vector with same dimension as s_flat
  """
  def dynamics(self, s_flat, l_flat):
    # Unpack variables
    x, dx  = self.s_unpack(s_flat)
    l = self.l_unpack(l_flat)

    new_dx_list = []
    ddx_list = []
    for t_ind in range(self.nGrid):
      x_i = x[t_ind, :]
      dx_i = dx[t_ind, :]

      # Compute dx at each collocation point
      # dx is a (7x1) vector
      new_dx_i = SX.zeros((7,1))
      # First 3 elements are position time-derivatives
      new_dx_i[0:3, :] = dx_i[0, 0:3]
      # Last 4 elements are quaternion time-derivatives
      ## Transform angular velocities dx into quaternion time-derivatives
      quat_i = x_i[0, 3:]
      dquat_i = 0.5 * self.get_dx_to_dquat_matrix(quat_i) @ dx_i[0, 3:].T
      new_dx_i[3:, :] = dquat_i
      new_dx_list.append(new_dx_i)

      # Compute ddx at each collocation point
      Mo = self.get_M_obj()
      G = self.get_grasp_matrix(x_i)
      gapp = self.get_gapp()
      l_i = l[t_ind, :].T
      #print("t_ind: {}".format(t_ind))
      #print(x_i)
      #print(l_i)
      #print(inv(Mo))
      #print("gapp: {}".format(gapp))
      #print(G.shape)
      #print((gapp + G@l_i).shape)
      ddx_i = inv(Mo) @ (gapp + G @ l_i)
      ddx_list.append(ddx_i)

    new_dx = horzcat(*new_dx_list).T
    ddx = horzcat(*ddx_list).T

    ds = self.s_pack(new_dx, ddx)
    return ds

  """
  Get matrix to transform angular velocity to quaternion time derivative
  Input:
    quat: [qx, qy, qz, qw]
  """
  def get_dx_to_dquat_matrix(self, quat):
    qx = quat[0]
    qy = quat[1]
    qz = quat[2]
    qw = quat[3]

    M = np.array([
                  [-qx, -qy, -qz],
                  [qw, qz, -qy],
                  [-qz, qw, qx],
                  [qy, -qx, qw],
                ])
    return SX(M)

  """
  Constrain state at end of trajectory to be at x_goal
  With a slack variable
  First, just add tolerance
  """
  def x_goal_constraint(self, s_flat,a, x_goal):
    x, dx  = self.s_unpack(s_flat)
    x_end = x[-1, :]

    con_list = []
    for i in range(self.x_dim):
      f = a[i] - (x_goal[0, i] - x_end[0, i]) ** 2
      con_list.append(f)

    return horzcat(*con_list)

################################################################################
# End of constraint functions
################################################################################
  """
  Get position of fingertips in world frame
  q_i: joint configuration at one timestep
  """
  # TODO
  def FK(self, q_cur):
    # list of fingertip positions in world frame
    ft_wf_list = []

    print(q_cur.shape)
    for i, theta_base_deg in enumerate(self.theta_base_deg_list):
      q1 = q_cur[0, self.qnum * i]
      q2 = q_cur[0, self.qnum * i + 1]
      q3 = q_cur[0, self.qnum * i + 2]
      theta_base = theta_base_deg * (np.pi/180)
      one_finger_ft_wf = SX([
         [0.1626*(sin(q1)*sin(q2)*cos(theta_base) - sin(theta_base)*cos(q2))*sin(q3) - 0.1626*(sin(q1)*cos(q2)*cos(theta_base) + sin(q2)*sin(theta_base))*cos(q3) - 0.16*sin(q1)*cos(q2)*cos(theta_base) - 0.16*sin(q2)*sin(theta_base) - 0.0505*sin(theta_base) + 0.08457*cos(q1)*cos(theta_base)],
         [0.1626*(sin(q1)*sin(q2)*sin(theta_base) + cos(q2)*cos(theta_base))*sin(q3) - 0.1626*(sin(q1)*sin(theta_base)*cos(q2) - sin(q2)*cos(theta_base))*cos(q3) - 0.16*sin(q1)*sin(theta_base)*cos(q2) + 0.16*sin(q2)*cos(theta_base) + 0.08457*sin(theta_base)*cos(q1) + 0.0505*cos(theta_base)],
         [-0.08457*sin(q1) + 0.1626*sin(q2)*sin(q3)*cos(q1) - 0.1626*cos(q1)*cos(q2)*cos(q3) - 0.16*cos(q1)*cos(q2) + 0.29]])

      ft_wf_list.append(one_finger_ft_wf)

    return ft_wf_list

  """
  Get Jacobian of fingers 
  """
  # TODO
  def get_jacobian(self, q_cur): 
    J = SX.zeros(3, self.fnum*self.qnum)

    for finger_id in range(self.fnum):
      # Get full Jacobian for finger i
      Ji = self.custom_pinocchio_utils.get_tip_link_jacobian(finger_id, q_cur)
      # Just take first 3 rows, which correspond to linear velocities of fingertip
      Ji = Ji[:3, :]
      # Add to full hand jacobian
      J += Ji
    return J

  """
  Get pnorm of cp_param tuple
  """
  def get_pnorm(self, cp_param):
    # Compute pnorm of cp
    pnorm = 0
    for param in cp_param:
      pnorm += fabs(param) ** self.p
    pnorm = pnorm ** (1/self.p)
    
    return pnorm
    
  """
  Calculate mass matrix of hand given joint positions q
  """
  # TODO
  #def get_M_hand(self, q):

  """
  Get grasp matrix
  Input:
  x: object pose [px, py, pz, qw, qx, qy, qz]
  """
  def get_grasp_matrix(self, x):
    # Transformation matrix from object frame to world frame
    quat_o_2_w = [x[0,3], x[0,4], x[0,5], x[0,6]]

    G_list = []

    # Calculate G_i (grasp matrix for each finger)
    for c in self.cp_list:
      cp_pos_of = c["position"] # Position of contact point in object frame
      quat_cp_2_o = c["orientation"] # Orientation of contact point frame w.r.t. object frame

      S = np.array([
                   [0, -cp_pos_of[2], cp_pos_of[1]],
                   [cp_pos_of[2], 0, -cp_pos_of[0]],
                   [-cp_pos_of[1], cp_pos_of[0], 0]
                   ])

      P_i = np.eye(6)
      P_i[3:6,0:3] = S

      # Orientation of cp frame w.r.t. world frame
      # quat_cp_2_w = quat_o_2_w * quat_cp_2_o
      quat_cp_2_w = utils.multiply_quaternions(quat_o_2_w, quat_cp_2_o)
      # R_i is rotation matrix from contact frame i to world frame
      R_i = utils.get_matrix_from_quaternion(quat_cp_2_w)
      R_i_bar = SX.zeros((6,6))
      R_i_bar[0:3,0:3] = R_i
      R_i_bar[3:6,3:6] = R_i

      G_iT = R_i_bar.T @ P_i.T
      G_list.append(G_iT)
    
    #GT_full = np.concatenate(G_list)
    GT_full = vertcat(*G_list)
    GT = self.H @ GT_full
    #print(GT.T)
    return GT.T

  """
  Get 6x6 object inertia matrix
  """
  def get_M_obj(self):
    M = np.zeros((6, 6))
    M[0,0] = M[1,1] = M[2,2] = self.obj_mass
    M[3,3] = self.obj_mass * (self.obj_shape[0]**2 + self.obj_shape[2]**2) / 12
    M[4,4] = self.obj_mass * (self.obj_shape[1]**2 + self.obj_shape[2]**2) / 12
    M[5,5] = self.obj_mass * (self.obj_shape[0]**2 + self.obj_shape[1]**2) / 12
    return M

  """
  Compute external gravity force on object, in -z direction
  """
  def get_gapp(self):
    gapp = np.array([[0],[0], [self.gravity * self.obj_mass], [0], [0], [0]])
    return gapp

  """
  Get 4x4 tranformation matrix from contact point frame to object frame
  Input:
  cp: dict with "position" and "orientation" fields in object frame
  """
  def get_R_cp_2_o(self, cp):
    #H = SX.zeros((4,4))
    
    quat = cp["orientation"]
    p = cp["position"]
    R = utils.get_matrix_from_quaternion(quat)

    return R
    #H[3,3] = 1
    #H[0:3,0:3] = R
    ##H[0:3,3] = p[:]
    ## Test transformation
    ##print("calculated: {}".format(H @ np.array([0,0,0,1])))
    ##print("actual: {}".format(p))
    #return H

  def get_R_o_2_w(self, x):
    quat = [x[0,3], x[0,4], x[0,5], x[0,6]]
    R = utils.get_matrix_from_quaternion(quat)
    return R

  """
  Get 4x4 tranformation matrix from object frame to world frame
  Input:
  x: object pose [px, py, pz, qw, qx, qy, qz]
  """
  def get_H_o_2_w(self, x):
    H = SX.zeros((4,4))
    
    quat = [x[0,3], x[0,4], x[0,5], x[0,6]]
    R = utils.get_matrix_from_quaternion(quat)
    p = np.array([x[0,0], x[0,1], x[0,2]])

    H[3,3] = 1
    H[0:3,0:3] = R
    H[0:3,3] = p[:]
    # Test transformation
    #print("calculated: {}".format(H @ np.array([0,0,0,1])))
    #print("actual: {}".format(p))
    return H

  """
  Get 4x4 transformation matrix from world to object frame
  """
  def get_H_w_2_o(self, x):
    H = np.zeros((4,4))
    quat = [x[0,3], x[0,4], x[0,5], x[0,6]]
    p = np.array([x[0,0], x[0,1], x[0,2]])
    p_inv, quat_inv = utils.invert_transform(p, quat)
    R = utils.get_matrix_from_quaternion(quat_inv)
    H[3,3] = 1
    H[0:3,0:3] = R
    H[0:3,3] = p_inv[:]
    # Test transformation
    #print("calculated: {}".format(H @ np.array([0,0,1,1])))
    return H

  """
  Get list of contact point dicts given cp_params list
  Each contact point is: {"position_of", "orientation_of"}
  """
  def get_contact_points_from_cp_params(self, cp_params):
    cp_list = []
    for param in cp_params:
      pos_of, quat_of = self.cp_param_to_cp_of(param)
      cp = {"position": pos_of, "orientation": quat_of}
      cp_list.append(cp)
    return cp_list
      
  """
  Get contact point position and orientation in object frame (OF)
  Input:
  (x_param, y_param, z_param) tuple
  """
  def cp_param_to_cp_of(self, cp_param):
    pnorm = self.get_pnorm(cp_param)

    #print("cp_param: {}".format(cp_param))
    #print("pnorm: {}".format(pnorm))

    cp_of = []
    # Get cp position in OF
    for i in range(3):
      cp_of.append(-self.obj_shape[i]/2 + (cp_param[i]+1)*self.obj_shape[i]/2)
    cp_of = np.asarray(cp_of)

    # TODO: Find analytical way of computing theta
    # Compute derivatives dx, dy, dz of pnorm
    #d_pnorm_list = []
    #for param in cp_param:
    #  d = (param * (fabs(param) ** (self.p - 2))) / (pnorm**(self.p-1))
    #  d_pnorm_list.append(d)

    #print("d_pnorm: {}".format(d_pnorm_list))

    #dx = d_pnorm_list[0]
    #dy = d_pnorm_list[1]
    #dz = d_pnorm_list[2]

    #w = np.sin(np.arctan2(dz*dz+dy*dy, dx)/2)
    #x = 0
    ## This if case is to deal with the -0.0 behavior in arctan2
    #if dx == 0: # TODO: this is going to be an error for through contact opt, when dx is an SX var
    #  y = np.sin(np.arctan2(dz, dx)/2)
    #else:
    #  y = np.sin(np.arctan2(dz, -dx)/2)

    #if dx == 0: # TODO: this is going to be an error for through contact opt, when dx is an SX var
    #  z = np.sin(np.arctan2(-dy, dx)/2)
    #else:
    #  z = np.sin(np.arctan2(-dy, dx)/2)
    #quat = (w,x,y,z)

    x_param = cp_param[0]
    y_param = cp_param[1]
    z_param = cp_param[2]
    # For now, just hard code quat
    if y_param == -1:
      quat = (0, 0, np.sqrt(2)/2, np.sqrt(2)/2)
    elif y_param == 1:
      quat = (0, 0, -np.sqrt(2)/2, np.sqrt(2)/2)
    elif x_param == 1:
      quat = (0, 1, 0, 0)
    elif z_param == 1:
      quat = (0, np.sqrt(2)/2, 0, np.sqrt(2)/2)
    elif x_param == -1:
      quat = (0, 0, 0, 1)
    elif z_param == -1:
      quat = (0, -np.sqrt(2)/2, 0, np.sqrt(2)/2)

    return cp_of, quat

  def test_cp_param_to_cp_of(self):
    print("\nP1")
    p1 = (0, -1, 0)
    q = self.cp_param_to_cp_of(p1)
    print("quat: {}".format(q))

    print("\nP2")
    p2 = (0, 1, 0)
    q = self.cp_param_to_cp_of(p2)
    print("quat: {}".format(q))

    print("\nP3")
    p3 = (1, 0, 0)
    q = self.cp_param_to_cp_of(p3)
    print("quat: {}".format(q))

    print("\nP4")
    p4 = (0, 0, 1)
    q = self.cp_param_to_cp_of(p4)
    print("quat: {}".format(q))

    print("\nP5")
    p5 = (-1, 0, 0)
    q = self.cp_param_to_cp_of(p5)
    print("quat: {}".format(q))

    print("\nP6")
    p6 = (0, 0, -1)
    q = self.cp_param_to_cp_of(p6)
    print("quat: {}".format(q))

################################################################################
# Path constraints
################################################################################

  """
  Define upper and lower bounds for decision variables
  Constrain initial x, q
  Constrain l if specified
  Constrain initial and final object velocity, if specified
  """
  def path_constraints(self,
                       z,
                       x0,
                       x_goal = None,
                       l0     = None,
                       dx0    = None,
                       dx_end = None,
                      ):

    if self.log_file is not None:
      with open(self.log_file, "a+") as f:
        f.write("\nPath constraints: {}\n")

    t,s_flat,l_flat,a = self.decvar_unpack(z)
  
    nGrid = self.nGrid

    # Time bounds
    t_range = [0,self.tf] # initial and final time
    t_lb = np.linspace(t_range[0],t_range[1],nGrid) # lower bound
    t_ub = t_lb # upper bound
    #print("Timestamps: {}".format(t_lb))

    # State path constraints
    # Unpack state vector
    x,dx = self.s_unpack(s_flat) # Object pose constraints
    x_range = np.array([
                       [-0.15, 0.15], # x coord range
                       [-0.15,0.15], # y coord range
                       [0.0325,0.15], # z coord range
                       [-np.inf, np.inf], # qx range
                       [-np.inf, np.inf], # qy range
                       [-np.inf, np.inf], # qz range
                       [-np.inf, np.inf], # qw range
                       ])
    x_lb = np.ones(x.shape) * x_range[:,0]
    x_ub = np.ones(x.shape) * x_range[:,1]

    # Object pose boundary contraint (starting position of object)
    if self.log_file is not None:
      with open(self.log_file, "a+") as f:
        f.write("Constrain x0 to {}\n".format(x0))
    x_lb[0] = x0 
    x_ub[0] = x0 
    #if x_goal is not None:
    #  x_lb[-1] = x_goal
    #  x_ub[-1] = x_goal
    #  # Just z goal
    #  #x_lb[-1,1] = x_goal[0,1]
    #  #x_ub[-1,1] = x_goal[0,1]
    
    # Object velocity constraints
    dx_range = np.array([
                       [-np.inf, np.inf], # qw range
                       [-np.inf, np.inf], # qw range
                       [-np.inf, np.inf], # qw range
                       [-np.inf, np.inf], # qx range
                       [-np.inf, np.inf], # qx range
                       [-np.inf, np.inf], # qx range
                       #[-10,10], # x vel range
                       #[-10,10], # y vel range
                       #[-10,10], # z vel range
                       #[-2*np.pi, 2*np.pi], # angular velocity range
                       #[-2*np.pi, 2*np.pi], # angular velocity range
                       #[-2*np.pi, 2*np.pi], # angular velocity range
                       ])
    dx_lb = np.ones(dx.shape) * dx_range[:,0]
    dx_ub = np.ones(dx.shape) * dx_range[:,1]
    if dx0 is not None:
      if self.log_file is not None:
        with open(self.log_file, "a+") as f:
          f.write("Constrain dx0 to {}\n".format(dx0))
      dx_lb[0] = dx0
      dx_ub[0] = dx0
    if dx_end is not None:
      if self.log_file is not None:
        with open(self.log_file, "a+") as f:
          f.write("Constrain dx_end to {}\n".format(dx_end))
      dx_lb[-1] = dx_end
      dx_ub[-1] = dx_end

    # Contact force contraints
    # For now, just define min and max forces
    l = self.l_unpack(l_flat)
    l_epsilon = 0
    # Limits for one finger
    f1_l_range = np.array([
                         [0, np.inf], # c1 fn force range
                         [-np.inf, np.inf], # c1 ft force range
                         [-np.inf, np.inf], # c1 ft force range
                         #[-np.inf, np.inf], # c1 ft force range
                         ])
    l_range = np.tile(f1_l_range, (self.fnum, 1))
    l_lb = np.ones(l.shape) * l_range[:,0]
    l_ub = np.ones(l.shape) * l_range[:,1]
    # Initial contact force constraints
    if l0 is not None:
      if self.log_file is not None:
        with open(self.log_file, "a+") as f:
          f.write("Constrain l0 to {}\n".format(l0))
      l_lb[0] =  l0
      l_ub[0] =  l0

    # Pack state contraints
    s_lb = self.s_pack(x_lb,dx_lb)
    s_ub = self.s_pack(x_ub,dx_ub)

    a_lb = np.zeros(a.shape)
    a_ub = np.ones(a.shape) * np.inf

    # Pack the constraints for all dec vars
    z_lb = self.decvar_pack(t_lb,s_lb,self.l_pack(l_lb),a_lb)
    z_ub = self.decvar_pack(t_ub,s_ub,self.l_pack(l_ub),a_ub)

    return z_lb, z_ub

  """
  Set initial trajectory guess
  For now, just define everything to be 0
  """
  def get_initial_guess(self, z_var, x0, x_goal):
    t_var, s_var, l_var, a_var = self.decvar_unpack(z_var)

    # Define time points to be equally spaced
    t_traj = np.linspace(0,self.tf,self.nGrid) 

    x_var, dx_var = self.s_unpack(s_var)
    dx_traj = np.zeros(dx_var.shape)
    x_traj = np.squeeze(np.linspace(x0, x_goal, self.nGrid))
    s_traj = self.s_pack(x_traj, dx_traj)
    
   
    l_traj = np.ones(self.l_unpack(l_var).shape)
    #l_traj[:,2] = 0.1
    #l_traj[:,8] = 0.1
    #l_traj[:,0] = 1
    #l_traj[:,6] = 1

    a_traj = np.zeros(a_var.shape)

    z_traj = self.decvar_pack(t_traj, s_traj, self.l_pack(l_traj),a_traj)
    
    return z_traj

def main():
  system = FixedContactPointSystem()
  system.test_cp_param_to_cp_of()

if __name__ == "__main__":
  main()
