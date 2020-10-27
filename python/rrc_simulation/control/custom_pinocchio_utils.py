import numpy as np

import pinocchio

from rrc_simulation.pinocchio_utils import PinocchioUtils

class CustomPinocchioUtils(PinocchioUtils):
  """
  Consists of kinematic methods for the finger platform.
  """

  def __init__(self, finger_urdf_path, tip_link_names):
    """
    Initializes the finger model on which control's to be performed.

    Args:
        finger (SimFinger): An instance of the SimFinger class
    """
    super().__init__(finger_urdf_path, tip_link_names)
 
  def get_tip_link_jacobian(self, finger_id, q):
    """
    Get Jacobian for tip link of specified finger
    All other columns are 0
    """
    pinocchio.computeJointJacobians(
        self.robot_model, self.data, q,
    )
    #pinocchio.framesKinematics(
    #    self.robot_model, self.data, q,
    #)
    pinocchio.framesForwardKinematics(
        self.robot_model, self.data, q,
    )
    frame_id = self.tip_link_ids[finger_id]
    Ji = pinocchio.getFrameJacobian(
      self.robot_model,
      self.data,
      frame_id,
      pinocchio.ReferenceFrame.LOCAL_WORLD_ALIGNED,
    )

    #print(self.robot_model.frames[frame_id].placement)
    #print(self.data.oMf[frame_id].rotation)
    return Ji

  def inverse_kinematics(self, fid, xdes, q0):
    """
    Method not in use right now, but is here with the intention
    of using pinocchio for inverse kinematics instead of using
    the in-house IK solver of pybullet.
    """
    raise NotImplementedError()
    dt = 1.0e-3
    pinocchio.computeJointJacobians(
        self.robot_model, self.data, q0,
    )
    pinocchio.framesKinematics(
        self.robot_model, self.data, q0,
    )
    pinocchio.framesForwardKinematics(
        self.robot_model, self.data, q0,
    )
    Ji = pinocchio.getFrameJacobian(
      self.robot_model,
      self.data,
      fid,
      pinocchio.ReferenceFrame.LOCAL_WORLD_ALIGNED,
    )[:3, :]
    xcurrent = self.data.oMf[fid].translation
    try:
      Jinv = np.linalg.inv(Ji)
    except Exception:
      Jinv = np.linalg.pinv(Ji)
    dq = Jinv.dot(xdes - xcurrent)
    qnext = pinocchio.integrate(self.robot_model, q0, dt * dq)
    return qnext

  def link_specific_forward_kinematics(self, joint_positions):
    """
    Compute end effector positions for the given joint configuration.

    Args:
        finger (SimFinger): a SimFinger object
        joint_positions (list): Flat list of angular joint positions.

    Returns:
        List of end-effector positions. Each position is given as an
        np.array with x,y,z positions.
    """

    # Get link ids for specified links
    link_names = [
        #"finger_upper_link_0",
        "finger_middle_link_0",
        #"finger_lower_link_0",
        #"finger_upper_link_120",
        "finger_middle_link_120",
        #"finger_lower_link_120",
        #"finger_upper_link_240",
        "finger_middle_link_240",
        #"finger_lower_link_240",
    ]
    link_ids = [
      self.robot_model.getFrameId(link_name)
      for link_name in link_names
    ]
    pinocchio.framesForwardKinematics(
      self.robot_model, self.data, joint_positions,
    )

    return [
      np.asarray(self.data.oMf[link_id].translation).reshape(-1).tolist()
      for link_id in link_ids
    ]
