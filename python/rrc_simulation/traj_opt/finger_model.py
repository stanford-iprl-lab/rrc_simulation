import numpy as np
from casadi import *
from rrc_simulation import trifinger_platform
from rrc_simulation.control.custom_pinocchio_utils import CustomPinocchioUtils

"""
Finger model class
For now, defines kinematic quantities
"""
class FingerModel:
  """
  theta_base: fixed z angle of finger base w.r.t base joint, in radians
  """
  def __init__(self, theta_base):

    # joint origin xyz values w.r.t previous joint
    self.j1_xyz = [0, 0, 0] # joint 1 w.r.t upper holder joint (joint 0)
    self.j2_xyz = [0.01685, 0.0505, 0] # joint 2
    self.j3_xyz = [0.04922, 0, -0.16] # joint 3
    self.j4_xyz = [0.0185, 0, -0.1626] # joint 4

    # Link axis directions
    l1_dir = np.array([0, 1, 0])
    l2_dir = np.array([0, 0, -1])
    l2_dir = np.array([0, 0, -1])
    l3_dir = np.array([0, 0, -1])
    l4_dir = np.array([0, 0, -1])
    self.l_dir = [l1_dir, l2_dir, l3_dir, l4_dir]

    # Link offsets
    l1_off = np.array([0,0,0])
    l2_off = np.array([0.04,0,0])
    l3_off = np.array([0.02,0,0])
    l4_off = np.array([0,0,0])
    self.l_off = [l1_off, l2_off, l3_off, l4_off]

    # Link lenghts, from TriFinger paper schematic
    l1_len = 0.179 + 0.0195
    l2_len = 0.16
    l3_len = 0.16
    l4_len = 0.0
    self.l_len = [l1_len, l2_len, l3_len, l4_len]

    self.l_axis = []
    for i in range(4):
      self.l_axis.append(self.l_len[i] * self.l_dir[i])

    self.theta_base = theta_base

    # bounding sphere radii for each link
    self.r_list = [0.02, 0.02, 0.015, 0.4]
    # number of bounding sphere per link, for each link
    self.snum_list = [6, 5, 5, 1]

    # define bounding sphere centers for each link, in link frames (lf)
    self.sphere_centers_lf = []
    for i in range(len(self.l_len)):
      link_centers = np.linspace(self.l_off[i] + self.r_list[i]*self.l_dir[i],
                                 self.l_off[i] + self.l_axis[i]-self.l_dir[i]*self.r_list[i],
                                 self.snum_list[i])
      if i == 3:
        link_centers = np.zeros((1,3))
      self.sphere_centers_lf.append(link_centers)

    self.q = None

  """
  Joint 0 frame wrt base joint
  theta_base is fixed z angle of finger base: 0, -120deg, -240deg
  """
  def H_0_wrt_base(self):
    H_0_wrt_base = np.array([
                        [np.cos(self.theta_base), -np.sin(self.theta_base), 0, 0],
                        [np.sin(self.theta_base), np.cos(self.theta_base), 0, 0],
                        [0, 0, 1, 0.29],
                        [0, 0, 0, 1],
                        ])
    return H_0_wrt_base

  """
  Frame 1 w.r.t. frame 0
  Rotation around y axis
  """
  def H_1_wrt_0(self):
    q1 = self.q[0]
    H_1_wrt_0 = np.array([
                        [np.cos(q1),    0, np.sin(q1), self.j1_xyz[0]],
                        [0,             1,          0, self.j1_xyz[1]],
                        [-np.sin(q1),   0, np.cos(q1), self.j1_xyz[2]],
                        [0, 0, 0, 1],
                        ])
    return H_1_wrt_0
  
  """
  Frame 2 w.r.t. frame 1
  Rotation around x axis
  """
  def H_2_wrt_1(self):
    q2 = self.q[1]
    H_2_wrt_1 = np.array([
                      [1,          0,           0, self.j2_xyz[0]],
                      [0, np.cos(q2), -np.sin(q2), self.j2_xyz[1]],
                      [0, np.sin(q2),  np.cos(q2), self.j2_xyz[2]],
                      [0,       0,        0,         1],
                      ])
    return H_2_wrt_1
  
  """
  # Frame 3 w.r.t. frame 2
  # Rotation around x axis
  """
  def H_3_wrt_2(self):
    q3 = self.q[2]
    H_3_wrt_2 = np.array([
                      [1,          0,           0, self.j3_xyz[0]],
                      [0, np.cos(q3), -np.sin(q3), self.j3_xyz[1]],
                      [0, np.sin(q3),  np.cos(q3), self.j3_xyz[2]],
                      [0,       0,        0,         1],
                      ])
    return H_3_wrt_2
  
  """
  Transformation from frame 3 to 4 (fingertip joint)
  Fixed
  """
  def H_4_wrt_3(self):
    H_4_wrt_3 = np.array([
                      [1, 0, 0, self.j4_xyz[0]],
                      [0, 1, 0, self.j4_xyz[1]],
                      [0, 0, 1, self.j4_xyz[2]],
                      [0, 0, 0,         1],
                      ])
    return H_4_wrt_3

  """
  Transform sphere centers for each link to world frame, given current q
  """
  def get_sphere_centers_wf(self, q):
    self.q = q 
    sphere_centers_wf = []
    for i in range(len(self.l_axis)):
      centers_wf = SX.zeros(self.sphere_centers_lf[i].shape) 
      if i == 0:
        H = self.H_0_wrt_base()@self.H_1_wrt_0()
      elif i == 1:
        H = self.H_0_wrt_base()@self.H_1_wrt_0()@self.H_2_wrt_1()
      elif i == 2:
        H = self.H_0_wrt_base()@self.H_1_wrt_0()@self.H_2_wrt_1()@self.H_3_wrt_2()
      else: # i == 3
        H = self.H_0_wrt_base()@self.H_1_wrt_0()@self.H_2_wrt_1()@self.H_3_wrt_2()@self.H_4_wrt_3()

      # Transform each sphere center on the link by H
      cur_link_centers = self.sphere_centers_lf[i]
      for ci in range(cur_link_centers.shape[0]):
        c_wf = H @ np.append(cur_link_centers[ci,:],1)
        centers_wf[ci, :] = c_wf[0:3]
      sphere_centers_wf.append(centers_wf)
    return sphere_centers_wf

  """
  Transform sphere centers for each link to world frame, given current q
  """
  def get_sphere_centers_wf_to_plot(self, q):
    self.q = q 
    sphere_centers_wf = []
    for i in range(len(self.l_axis)):
      centers_wf = np.zeros(self.sphere_centers_lf[i].shape) 
      if i == 0:
        H = self.H_0_wrt_base()@self.H_1_wrt_0()
      elif i == 1:
        H = self.H_0_wrt_base()@self.H_1_wrt_0()@self.H_2_wrt_1()
      elif i == 2:
        H = self.H_0_wrt_base()@self.H_1_wrt_0()@self.H_2_wrt_1()@self.H_3_wrt_2()
      else: # i == 3
        H = self.H_0_wrt_base()@self.H_1_wrt_0()@self.H_2_wrt_1()@self.H_3_wrt_2()@self.H_4_wrt_3()

      # Transform each sphere center on the link by H
      cur_link_centers = self.sphere_centers_lf[i]
      for ci in range(cur_link_centers.shape[0]):
        c_wf = H @ np.append(cur_link_centers[ci,:],1)
        centers_wf[ci, :] = c_wf[0:3]
      sphere_centers_wf.append(centers_wf)
    return sphere_centers_wf


