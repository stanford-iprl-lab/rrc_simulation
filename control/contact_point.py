import numpy as np

"""
Class ContactPoint: 
--------------
A class used to represent a contact point on planar object, in WORLD frame (wf)
"""
class ContactPoint: 
    def __init__(self, pos_of, quat_of): 
        # In Object frame
        self.pos_of = pos_of
        self.quat_of = quat_of

        self.pos_wf = None
        self.quat_wf = None
      
    def print_pt_wf(self):
        print("Contact point position WF: ", self.pos_wf, ", Y: ", self.quat_wf)

    def print_pt_of(self):
        print("Contact point position OF: ", self.pos_of, ", Y: ", self.quat_of)

    #"""
    #Returns rotation matrix that transforms vector in contact point frame to vector in world frame
    #"""
    #def get_R_matrix(self): 
    #    R  = np.array([[np.cos(self.theta), -np.sin(self.theta), 0],
    #                   [np.sin(self.theta),  np.cos(self.theta), 0],
    #                   [0, 0, 1]]) 
    #    return R

    #"""
    #Returns rotation matrix that transforms vector in contact point frame to vector in object frame
    #"""
    #def get_R_obj_matrix(self, obj_theta): 
    #    theta_of = self.theta - obj_theta # Contact point theta in object frame
    #    R_obj  = np.array([[np.cos(theta_of), -np.sin(theta_of), 0],
    #                   [np.sin(theta_of),  np.cos(theta_of), 0],
    #                   [0, 0, 1]]) 
    #    return R_obj
