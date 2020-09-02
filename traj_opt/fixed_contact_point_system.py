import numpy as np
from casadi import *
import pybullet

from rrc_simulation.tasks import move_cube
import utils

class FixedContactPointSystem:

  def __init__(self,
               platform = None,
               obj_pose = None,
               obj_shape = None,
               obj_mass = None,
              ):
    print("Initialize fixed contact point system")

    self.p = 100

    self.obj_size = (1, 2, 3)

    self.obj_pose = obj_pose
    self.obj_shape = obj_shape # (width, length, height), (x, y, z)
    self.obj_mass = obj_mass
    self.obj_mu = 1

    self.platform = platform 

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
  #def get_grasp_matrix(self, x):

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
  Get 4x4 tranformation matrix from object frame to world frame
  Input:
  x: object pose [px, py, pz, qw, qx, qy, qz]
  """
  def get_H_o_2_w(self, x):
    H = np.zeros((4,4))
    
    quat = [x[3], x[4], x[5], x[6]]
    R = utils.get_matrix_from_quaternion(quat)
    p = np.array([x[0], x[1], x[2]])

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
    quat = [x[3], x[4], x[5], x[6]]
    p = np.array([x[0], x[1], x[2]])
    p_inv, quat_inv = utils.invert_transform(p, quat)
    R = utils.get_matrix_from_quaternion(quat_inv)
    H[3,3] = 1
    H[0:3,0:3] = R
    H[0:3,3] = p_inv[:]
    # Test transformation
    #print("calculated: {}".format(H @ np.array([0,0,1,1])))
    return H

  """
  Get contact point position and orientation in object frame (OF)
  Input:
  (x_param, y_param, z_param) tuple
  """
  def cp_params_to_cp_of(self, cp_param):
    pnorm = self.get_pnorm(cp_param)

    print("cp_param: {}".format(cp_param))
    print("pnorm: {}".format(pnorm))

    cp_wf = []
    # Get cp position in OF
    for i in range(3):
      cp_wf.append(-self.obj_size[i]/2 + (cp_param[i]+1)*self.obj_size[i]/2)
    cp_wf = np.asarray([cp_wf])

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

    return cp_wf, quat

  def test_cp_params_to_cp_of(self):
    print("\nP1")
    p1 = (0, -1, 0)
    q = self.cp_params_to_cp_of(p1)
    print("quat: {}".format(q))

    print("\nP2")
    p2 = (0, 1, 0)
    q = self.cp_params_to_cp_of(p2)
    print("quat: {}".format(q))

    print("\nP3")
    p3 = (1, 0, 0)
    q = self.cp_params_to_cp_of(p3)
    print("quat: {}".format(q))

    print("\nP4")
    p4 = (0, 0, 1)
    q = self.cp_params_to_cp_of(p4)
    print("quat: {}".format(q))

    print("\nP5")
    p5 = (-1, 0, 0)
    q = self.cp_params_to_cp_of(p5)
    print("quat: {}".format(q))

    print("\nP6")
    p6 = (0, 0, -1)
    q = self.cp_params_to_cp_of(p6)
    print("quat: {}".format(q))

def main():
  system = FixedContactPointSystem()
  system.test_cp_params_to_cp_of()

if __name__ == "__main__":
  main()
