import numpy as np
from casadi import *
import pybullet

from rrc_simulation.tasks import move_cube
from rrc_simulation.traj_opt import utils
from rrc_simulation.traj_opt.finger_model import FingerModel

"""
Fingers and static object
For collision avoidance
"""
class StaticObjectSystem:

  def __init__(self,
               nGrid     = 100,
               dt        = 0.1,
               q0        = np.array([[0,0.9,-1.7,0,0.9,-1.7,0,0.9,-1.7]]),
               obj_shape = None,
               obj_pose  = move_cube.Pose(),
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
    self.obj_mu = 1

    self.gravity = -10
    
    self.log_file = log_file
    self.theta_base_deg_list = [0, -120, -240] # List of angles of finger bases around arena (degrees)

    # Object pose and shape
    self.obj_shape = obj_shape
    self.obj_pose  = obj_pose


    print(self.FK(q0))
    # Define fingers
    self.fingers = []
    for i in range(self.fnum):
      theta_base = self.theta_base_deg_list[i] * (np.pi/180)
      self.fingers.append(FingerModel(theta_base, q0[0, i*self.qnum:i*self.qnum+self.qnum]))
    quit()

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

    # Slack variables for ft_goal
    a  = SX.sym("a", qnum * fnum)

    return t,s_flat,a

  """
  Pack the decision variables into a single horizontal vector
  """
  def decvar_pack(self,t,s,a):
    z = vertcat(t,s,a)
    return z

  """
  Unpack the decision variable vector z into:
  t: times (nGrid x 1) vector
  s: packed state vector
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
    s_end_ind = s_start_ind + 2*nGrid*fnum*qnum
    s_flat = z[s_start_ind:s_end_ind]

    a_start_ind = s_end_ind
    a = z[a_start_ind:]
    
    return t,s_flat,a

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
  Constrain fingertip positions at end of trajectory to be at ft_goal
  With a slack variable
  ft_goal_list: list of finger tip goal positions
  """
  def ft_goal_constraint(self, s_flat,a, ft_goal_list):
    q, dq  = self.s_unpack(s_flat)
    q_end = q[-1, :]

    # Get list of ft positions at q_end
    ft_end = self.FK(q_end)
    con_list = []
    for f_i in range(self.fnum):
      for q_i in range(self.qnum):
        f = a[f_i*self.qnum + q_i] - (ft_goal_list[f_i][q_i,0] - ft_end[f_i][q_i,0]) ** 2
        con_list.append(f)
    return horzcat(*con_list)

  """
  Collision constraint
  """
  def collision_constraint(self, s_flat):
    con_list = []
    q, dq  = self.s_unpack(s_flat)
    
    ft_pos_list = self.FK(q[0,:])
    ft_pos_wf = ft_pos_list[0]
    self.get_pnorm_of_pos_wf(ft_pos_wf)
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

    for i, theta_base_deg in enumerate(self.theta_base_deg_list):
      q1 = q_cur[0, self.qnum * i]
      q2 = q_cur[0, self.qnum * i + 1]
      q3 = q_cur[0, self.qnum * i + 2]
      theta_base = theta_base_deg * (np.pi/180)
    
      # FK computed in jacobian_utils.py with sympy
      one_finger_ft_wf = np.array([
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

    for i, theta_base_deg in enumerate(self.theta_base_deg_list):
      q1 = q_cur[0, self.qnum * i]
      q2 = q_cur[0, self.qnum * i + 1]
      q3 = q_cur[0, self.qnum * i + 2]
      theta_base = theta_base_deg * (np.pi/180)

      # 3x3 jacobian for single finger, computed in jacobian_utils.py with sympy
      one_finger_J = np.array([[-0.08457*sin(q1)*cos(theta_base) + 0.1626*sin(q2)*sin(q3)*cos(q1)*cos(theta_base) - 0.1626*cos(q1)*cos(q2)*cos(q3)*cos(theta_base) - 0.16*cos(q1)*cos(q2)*cos(theta_base), (0.1626*sin(q1)*sin(q2)*cos(theta_base) - 0.1626*sin(theta_base)*cos(q2))*cos(q3) + (0.1626*sin(q1)*cos(q2)*cos(theta_base) + 0.1626*sin(q2)*sin(theta_base))*sin(q3) + 0.16*sin(q1)*sin(q2)*cos(theta_base) - 0.16*sin(theta_base)*cos(q2), (0.1626*sin(q1)*sin(q2)*cos(theta_base) - 0.1626*sin(theta_base)*cos(q2))*cos(q3) - (-0.1626*sin(q1)*cos(q2)*cos(theta_base) - 0.1626*sin(q2)*sin(theta_base))*sin(q3)], [-0.08457*sin(q1)*sin(theta_base) + 0.1626*sin(q2)*sin(q3)*sin(theta_base)*cos(q1) - 0.1626*sin(theta_base)*cos(q1)*cos(q2)*cos(q3) - 0.16*sin(theta_base)*cos(q1)*cos(q2), (0.1626*sin(q1)*sin(q2)*sin(theta_base) + 0.1626*cos(q2)*cos(theta_base))*cos(q3) + (0.1626*sin(q1)*sin(theta_base)*cos(q2) - 0.1626*sin(q2)*cos(theta_base))*sin(q3) + 0.16*sin(q1)*sin(q2)*sin(theta_base) + 0.16*cos(q2)*cos(theta_base), (0.1626*sin(q1)*sin(q2)*sin(theta_base) + 0.1626*cos(q2)*cos(theta_base))*cos(q3) - (-0.1626*sin(q1)*sin(theta_base)*cos(q2) + 0.1626*sin(q2)*cos(theta_base))*sin(q3)], [-0.1626*sin(q1)*sin(q2)*sin(q3) + 0.1626*sin(q1)*cos(q2)*cos(q3) + 0.16*sin(q1)*cos(q2) - 0.08457*cos(q1), 0.1626*sin(q2)*cos(q1)*cos(q3) + 0.16*sin(q2)*cos(q1) + 0.1626*sin(q3)*cos(q1)*cos(q2), 0.1626*sin(q2)*cos(q1)*cos(q3) + 0.1626*sin(q3)*cos(q1)*cos(q2)]])
      J[:,self.qnum*i:self.qnum*i + self.qnum] = one_finger_J

    return J

  """
  Get cp_param from fingertip position in world frame
  """ 
  def get_pnorm_of_pos_wf(self, p_wf):
    H_w_2_o = self.get_H_w_2_o()
    ft_pos_of = H_w_2_o @ np.append(p_wf, 1)
    cp_param = self.get_cp_param_from_pos_of(ft_pos_of[0:3])
    pnorm = self.get_pnorm(cp_param)
    return pnorm

  """
  Get cp_param from pos_of
  Normalize pos_of to a unit cube
  """
  def get_cp_param_from_pos_of(self, p_of):
    cp_param = []  
    for i in range(3):
      param = (2 * p_of[i] + self.obj_shape[i])/self.obj_shape[i] - 1
      cp_param.append(param)
    return cp_param
  
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
  Get 4x4 transformation matrix from world to object frame
  """
  def get_H_w_2_o(self):
    H = SX.zeros((4,4))
    quat = self.obj_pose.orientation
    p = self.obj_pose.position
    p_inv, quat_inv = utils.invert_transform(p, quat)
    R = utils.get_matrix_from_quaternion(quat_inv)
    H[3,3] = 1
    H[0:3,0:3] = R
    H[0:3,3] = p_inv[:]
    # Test transformation
    #print("calculated: {}".format(H @ np.array([0,0,1,1])))
    return H

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
                       q0,
                       dq0    = None,
                       dq_end = None,
                      ):

    if self.log_file is not None:
      with open(self.log_file, "a+") as f:
        f.write("\nPath constraints: {}\n")

    t,s_flat,a = self.decvar_unpack(z)
  
    nGrid = self.nGrid

    # Time bounds
    t_range = [0,self.tf] # initial and final time
    t_lb = np.linspace(t_range[0],t_range[1],nGrid) # lower bound
    t_ub = t_lb # upper bound
    #print("Timestamps: {}".format(t_lb))

    # State path constraints
    # Unpack state vector
    q,dq = self.s_unpack(s_flat) # Object pose constraints
    one_finger_q_range = np.array([
                                  [-0.33, 1.0], # joint 1 range
                                  [0.0, 1.57],  # joint 2 range
                                  [-2.7, 0.0],  # joint 3 range
                                  ])
    q_range = np.tile(one_finger_q_range, (self.fnum, 1))
    q_lb = np.ones(q.shape) * q_range[:,0]
    q_ub = np.ones(q.shape) * q_range[:,1]

    # Object pose boundary contraint (starting position of object)
    q_lb[0] = q0 
    q_ub[0] = q0 
    
    # Object velocity constraints
    one_finger_dq_range = np.array([
                       [-2, 2], # joint 1 range 
                       [-2, 2], # joint 2 range
                       [-2, 2], # joint 3 range
                       ])

    dq_range = np.tile(one_finger_dq_range, (self.fnum, 1))

    dq_lb = np.ones(dq.shape) * dq_range[:,0]
    dq_ub = np.ones(dq.shape) * dq_range[:,1]

    if dq0 is not None:
      dq_lb[0] = dq0
      dq_ub[0] = dq0
    if dq_end is not None:
      dq_lb[-1] = dq_end
      dq_ub[-1] = dq_end

    # Pack state contraints
    s_lb = self.s_pack(q_lb,dq_lb)
    s_ub = self.s_pack(q_ub,dq_ub)

    a_lb = np.zeros(a.shape)
    a_ub = np.ones(a.shape) * np.inf

    # Pack the constraints for all dec vars
    z_lb = self.decvar_pack(t_lb,s_lb,a_lb)
    z_ub = self.decvar_pack(t_ub,s_ub,a_ub)

    return z_lb, z_ub

  """
  Set initial trajectory guess
  For now, just define everything to be 0
  """
  def get_initial_guess(self, z_var, q0):
    t_var, s_var, a_var = self.decvar_unpack(z_var)

    # Define time points to be equally spaced
    t_traj = np.linspace(0,self.tf,self.nGrid) 

    q_var, dq_var = self.s_unpack(s_var)

    # Set q0 to entire q_traj guess
    q_traj = np.tile(q0, (self.nGrid, 1))

    # Joint velocities are zero
    dq_traj = np.zeros(dq_var.shape)

    s_traj = self.s_pack(q_traj, dq_traj)
    
    a_traj = np.zeros(a_var.shape)

    z_traj = self.decvar_pack(t_traj, s_traj, a_traj)
    
    return z_traj

def main():
  system = FixedContactPointSystem()
  system.test_cp_param_to_cp_of()

if __name__ == "__main__":
  main()
