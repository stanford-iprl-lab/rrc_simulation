import numpy as np
from casadi import *

from rrc_simulation.traj_opt.static_object_system import StaticObjectSystem
from rrc_simulation import trifinger_platform

class StaticObjectOpt:
  def __init__(self,
               finger_urdf_path,
               tip_link_names,
               nGrid     = 100,
               dt        = 0.1,
               cp_params = None,
               x0        = np.array([[0,0,0.0325,0,0,0,1]]),
               x_goal    = None,
               obj_shape = None,
               obj_mass  = None,
               ):

    self.nGrid = nGrid
    self.dt = dt
    
    # Define system
    self.system = StaticObjectSystem(
                                     finger_urdf_path,
                                     tip_link_names,
                                     nGrid     = nGrid,
                                     dt        = dt,
                                     cp_params = cp_params,
                                     obj_shape = obj_shape,
                                     obj_mass  = obj_mass,
                                    )
    
    # Test various functions
    #x = np.zeros((1,7))
    #x[0,0:3] = obj_pose.position
    #x[0,3] = obj_pose.orientation[3]
    #x[0,4:7] = obj_pose.orientation[0:3]
    ##print("x: {}".format(x))
    #self.system.get_grasp_matrix(x)

    # Get decision variables
    self.t,self.s_flat = self.system.dec_vars()
    # Pack t,x,u,l into a vector of decision variables
    self.z = self.system.decvar_pack(self.t,self.s_flat)

    #print(self.z)
    q, dq = self.system.s_unpack(self.s_flat)
    #print(self.system.s_unpack(self.s_flat))

    test_q = np.array([[0.5, 0.9, -1.7,0.5, 0.9, -1.7,0.5, 0.9, -1.7,]])
    # Test pinocchio functions with casadi variables
    print(self.system.FK(test_q))
    #print(self.system.FK(q[0,:]))

    quit()
    # Formulate constraints
    #self.g, self.lbg, self.ubg = self.get_constraints(self.system, self.t, self.s_flat, self.l_flat,self.a,x_goal)

    ## Get cost function
    #self.cost = self.cost_func(self.t,self.s_flat,self.l_flat,self.a,x_goal)

    ## Formulate nlp
    #problem = {"x":self.z, "f":self.cost, "g":self.g}
    #options = {"ipopt.print_level":0,
    #           "ipopt.max_iter":10000,
    #            "ipopt.tol": 1e-4,
    #            "ipopt.print_level":0,
    #            "print_time": 0
    #          }
    ##options["print_time"] = 0;
    ##options = {"iteration_callback": MyCallback('callback',self.z.shape[0],self.g.shape[0],self.system)}
    ##options["monitor"] = ["nlp_g"]
    ##options = {"monitor":["nlp_f","nlp_g"]}
    #self.solver = nlpsol("S", "ipopt", problem, options)

    ## TODO: intial guess
    #self.z0 = self.system.get_initial_guess(self.z, x0, x_goal)
    ##t0, s0, l0 = self.system.decvar_unpack(self.z0)
    ##x0, dx0 = self.system.s_unpack(s0)
    ##self.get_constraints(self.system,t0,s0,l0)

    ##print("\nINITIAL TRAJECTORY")
    ##print("time: {}".format(t0))
    ##print("x: {}".format(x0))
    ##print("dx: {}".format(dx0))
    ##print("contact forces: {}".format(l0))
  
    ## TODO: path constraints
    ##self.system.get_grasp_matrix(x0)

    #self.z_lb, self.z_ub = self.system.path_constraints(self.z, x0, x_goal=x_goal, dx0=np.zeros((1,6)), dx_end=np.zeros((1,6)))

    ## Set upper and lower bounds for decision variables
    #r = self.solver(x0=self.z0,lbg=self.lbg,ubg=self.ubg,lbx=self.z_lb,ubx=self.z_ub)
    #z_soln = r["x"]

    ## Final solution and cost
    #self.cost = r["f"]
    #self.t_soln,self.s_soln,l_soln_flat,a_soln = self.system.decvar_unpack(z_soln)
    #self.x_soln, self.dx_soln = self.system.s_unpack(self.s_soln)
    #self.l_soln = self.system.l_unpack(l_soln_flat)

    # Check that all quaternions are unit quaternions
    #print("Check quaternion magnitudes")
    #for i in range(self.nGrid):
    #  quat = self.x_soln[i, 3:]
    #  print(np.linalg.norm(quat))
  
    # Save solver time
    #statistics = self.solver.stats()
    #self.total_time_sec = statistics["t_wall_total"]

  """
  Computes cost
  """
  def cost_func(self,t,s_flat,l_flat,a,x_goal):
    cost = 0
    R = np.eye(self.system.fnum * self.system.l_i) * 1
    Q = np.eye(self.system.x_dim) * 1

    l = self.system.l_unpack(l_flat) 
    x,dx = self.system.s_unpack(s_flat)

    n = 0.2
    target_normal_forces = np.zeros(l[0,:].shape)
    target_normal_forces[0,0] = n
    target_normal_forces[0,3] = n
    target_normal_forces[0,6] = n

    # Slack variable penalties
    for i in range(a.shape[0]):
      if i < 3:
        cost += a[i] * 1 # Position of object
      else:
        cost += a[i] * 1 # Orientation of object

    # Contact forces
    for i in range(t.shape[0]):
      cost += 0.5 * (l[i,:] - target_normal_forces) @ R @ (l[i,:] - target_normal_forces).T
  
    for i in range(t.shape[0]):
      # Add the current distance to goal
      x_curr = x[i, :]
      delta = x_goal - x_curr
      cost += 0.5 * delta @ Q @ delta.T

    return cost

  """
  Formulates collocation constraints
  """
  def get_constraints(self,system,t,s,l,a,x_goal):
    ds = system.dynamics(s,l)
    
    x,dx = system.s_unpack(s)
    new_dx,ddx = system.s_unpack(ds)

    # Separate x and new_dx into position and orientation (quaternion) so we can normalize quaterions
    pos = x[:, 0:3]
    new_dpos = new_dx[:, 0:3]
    quat = x[:, 3:]
    new_dquat = new_dx[:, 3:]

    g = [] # Dynamics constraints
    lbg = [] # Dynamics constraints lower bound
    ubg = [] # Dynamics constraints upper bound
  
    # Loop over entire trajectory
    for i in range(t.shape[0] - 1):
      dt = t[i+1] - t[i]
      
      # pose - velocity
      # Handle position and linear velocity constraints first, since they don't need to be normalized
      for j in range(3):
        # dx
        f = 0.5 * dt * (new_dpos[i+1,j] + new_dpos[i,j]) + pos[i,j] - pos[i+1,j] 
        #print("new_dx, x, t{}, dim{}: {}".format(i,j,f))
        g.append(f)
        lbg.append(0)
        ubg.append(0)

      # Handle orientation and angular velocity - normalize first
      quat_i_plus_one = 0.5 * dt * (new_dquat[i+1,:] + new_dquat[i,:]) + quat[i,:]
      quat_i_plus_one_unit = quat_i_plus_one / norm_2(quat_i_plus_one)
      for j in range(4):
        # dx
        f = quat_i_plus_one_unit[0,j] - quat[i+1, j]
        #print("new_dquat, quat, t{}, dim{}: {}".format(i,j,f))
        g.append(f)
        lbg.append(0)
        ubg.append(0)

      # velocity - acceleration
      # iterate over all dofs
      for j in range(system.dx_dim):
        f = 0.5 * dt * (ddx[i+1,j] + ddx[i,j]) + dx[i,j] - dx[i+1,j]
        #print("dx, ddx, t{}, dim{}: {}".format(i,j,f))
        g.append(f)
        lbg.append(0)
        ubg.append(0)

    # Linearized friction cone constraints
    f_constraints = system.friction_cone_constraints(l)
    for r in range(f_constraints.shape[0]):
      for c in range(f_constraints.shape[1]):
        g.append(f_constraints[r,c])
        lbg.append(0)
        ubg.append(np.inf)

    #tol = 1e-16
    x_goal_constraints = system.x_goal_constraint(s, a, x_goal)
    for r in range(x_goal_constraints.shape[0]):
      for c in range(x_goal_constraints.shape[1]):
        g.append(x_goal_constraints[r,c])
        lbg.append(0)
        ubg.append(np.inf)
    
    return vertcat(*g), vertcat(*lbg), vertcat(*ubg)
    
def main():
  platform = trifinger_platform.TriFingerPlatform(
      visualization=False,
      enable_cameras=False,
      initial_robot_position=np.zeros(9),
      initial_object_pose = None
  )

  StaticObjectOpt(
               platform.simfinger.finger_urdf_path,
               platform.simfinger.tip_link_names,
               nGrid     = 5,
               dt        = 0.1,
               cp_params = None,
               x0        = np.array([[0,0,0.0325,0,0,0,1]]),
               x_goal    = None,
               obj_shape = None,
               obj_mass  = None,
               )

if __name__ == "__main__":
  main()

