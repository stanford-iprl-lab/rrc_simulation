import numpy as np

"""
Get 3x3 rotation matrix from quaternion
Inputs:
quat: quaternion [w, x, y, z]
"""
def get_matrix_from_quaternion(quat):
  R = np.zeros((3,3))
  
  w = quat[0]
  x = quat[1]
  y = quat[2]
  z = quat[3]
  
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
quat: quaternion [w, x, y, z]
"""
def invert_transform(p, quat):
  # Invert rotation
  w = quat[0]
  x = quat[1]
  y = quat[2]
  z = quat[3]
  quat_inv = [w, -x, -y, -z]
  R_inv = get_matrix_from_quaternion(quat_inv)
  
  # Invert translation
  p_inv = -R_inv @ np.array(p)

  return p_inv, quat_inv
  
