import numpy as np
from casadi import *
"""
Multiple 2 quaternions
return q * n
"""
def multiply_quaternions(q, n):
  qx = q[0]
  qy = q[1]
  qz = q[2]
  qw = q[3]

  nx = n[0]
  ny = n[1]
  nz = n[2]
  nw = n[3]
  
  w = qw*nw - qx*nx - qy*ny - qz*nz
  x = qw*nx + qx*nw + qy*nz - qz*ny
  y = qw*ny + qy*nw + qz*nx - qx*nz
  z = qw*nz + qz*nw + qx*ny - qy*nx

  product = [x,y,z,w]
  return product

"""
Get 3x3 rotation matrix from quaternion
Inputs:
quat: quaternion [x, y, z, w]
"""
def get_matrix_from_quaternion(quat):
  R = SX.zeros((3,3))
  
  x = quat[0]
  y = quat[1]
  z = quat[2]
  w = quat[3]
  
  R[0,0] = 2 * (w**2 + x**2) - 1
  R[0,1] = 2 * (x*y - w*z)
  R[0,2] = 2 * (x*z + w*y)
    
  R[1,0] = 2 * (x*y + w*z)
  R[1,1] = 2 * (w**2 + y**2) - 1
  R[1,2] = 2 * (y*z - w*x)

  R[2,0] = 2 * (x*z - w*y)
  R[2,1] = 2 * (y*z + w*x)
  R[2,2] = 2 * (w**2 + z**2) - 1

  return R

"""
Get inverse transform
Inputs:
p: position [x,y,z]
quat: quaternion [x, y, z, w]
"""
def invert_transform(p, quat):
  # Invert rotation
  x = quat[0]
  y = quat[1]
  z = quat[2]
  w = quat[3]

  quat_inv = [-x, -y, -z, w]
  R_inv = get_matrix_from_quaternion(quat_inv)
  
  # Invert translation
  p_inv = -R_inv @ np.array(p)

  return p_inv, quat_inv
  
