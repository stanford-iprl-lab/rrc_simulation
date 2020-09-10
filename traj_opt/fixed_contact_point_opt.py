import numpy as np
from casadi import *

from fixed_contact_point_system import FixedContactPointSystem

class FixedContactPointOpt:
  
  def __init__(self,
               nGrid     = 100,
               dt        = 0.1,
               x_goal    = None,
               platform  = None,
               obj_pose  = None,
               obj_shape = None,
               obj_mass  = None,
               ):

    self.nGrid = nGrid
    self.dt = dt
    
    # Define system
    self.system = FixedContactPointSystem(
                                     nGrid = nGrid,
                                     dt = dt,
                                     platform = platform,
                                     obj_pose = obj_pose,
                                     obj_shape = obj_shape,
                                     obj_mass = obj_mass,
                                    )
    
    # Test various functions
    #x = np.zeros((1,7))
    #x[0,0:3] = obj_pose.position
    #x[0,3] = obj_pose.orientation[3]
    #x[0,4:7] = obj_pose.orientation[0:3]
    ##print("x: {}".format(x))
    #self.system.get_grasp_matrix(x)

    # Get decision variables
    self.t,self.s_flat,self.l_flat = self.system.dec_vars()
    # Pack t,x,u,l into a vector of decision variables
    self.z = self.system.decvar_pack(self.t,self.s_flat,self.l_flat)

    #print(self.z)
    #print(self.system.s_unpack(self.s_flat))

    # Formulate constraints
    self.g, self.lbg, self.ubg = self.get_constraints(self.system, self.t, self.s_flat, self.l_flat)

    # Get cost function
    self.cost = self.cost_func(self.t,self.s_flat,self.l_flat,x_goal)

    # Formulate nlp
    problem = {"x":self.z, "f":self.cost, "g":self.g}
    #options = {"ipopt.print_level":5}
    options = {"ipopt.max_iter":10000,
                "ipopt.tol": 1e-4,
                "ipopt.mu_init": 0.5
              }
    #options = {"iteration_callback": MyCallback('callback',self.z.shape[0],self.g.shape[0],self.system)}
    #options["monitor"] = ["nlp_g"]
    #options = {"monitor":["nlp_f","nlp_g"]}
    self.solver = nlpsol("S", "ipopt", problem, options)

    # TODO: intial guess
    self.z0 = self.system.get_initial_guess(self.z)
    t0, s0, l0 = self.system.decvar_unpack(self.z0)
    x0, dx0 = self.system.s_unpack(s0)
    self.get_constraints(self.system,t0,s0,l0)

    print("\nINITIAL TRAJECTORY")
    print("time: {}".format(t0))
    print("x: {}".format(x0))
    print("dx: {}".format(dx0))
    print("contact forces: {}".format(l0))
  
    # TODO: path constraints
    x0 = np.array([[0,0,0.0325,1,0,0,0]])
    #self.system.get_grasp_matrix(x0)

    self.z_lb, self.z_ub = self.system.path_constraints(self.z, x0, dx0=np.zeros((1,6)), dx_end=np.zeros((1,6)))

    # Set upper and lower bounds for decision variables
    r = self.solver(x0=self.z0,lbg=self.lbg,ubg=self.ubg,lbx=self.z_lb,ubx=self.z_ub)
    z_soln = r["x"]

    # Final solution and cost
    self.cost = r["f"]
    self.t_soln,self.s_soln,l_soln_flat = self.system.decvar_unpack(z_soln)
    self.x_soln, self.dx_soln = self.system.s_unpack(self.s_soln)
    self.l_soln = self.system.l_unpack(l_soln_flat)

    # Check that all quaternions are unit quaternions
    print("Check quaternion magnitudes")
    for i in range(self.nGrid):
      quat = self.x_soln[i, 3:]
      print(np.linalg.norm(quat))
  
    # Transform contact forces from contact point frame to world frame
    self.l_wf_soln = np.zeros(self.l_soln.shape)
    for t_i in range(self.l_soln.shape[0]):
      for f_i in range(self.system.fnum):
        l_of = self.system.get_R_cp_2_o(self.system.cp_list[f_i]) @ (self.l_soln[t_i, f_i*self.system.l_i:f_i*self.system.l_i + self.system.l_i]).T
        l_wf = self.system.get_R_o_2_w(self.x_soln[t_i, :]) @ l_of

        for d in range(self.system.l_i):
          self.l_wf_soln[t_i, f_i*self.system.l_i + d] = l_wf[:, 0].elements()[d].__float__()

    # Final distance to goal
    #eef_final = self.system.get_eef_pos_world(self.q_soln)[-1, 0:2]
    #self.final_dist = norm_2(eef_final - eef_goal)

    # Save solver time
    statistics = self.solver.stats()
    self.total_time_sec = statistics["t_wall_total"]

  """
  Computes cost
  """
  def cost_func(self,t,s_flat,l_flat,x_goal):
    cost = 0
    R = np.eye(self.system.fnum * self.system.l_i)
    Q = np.eye(self.system.x_dim) * 10

    l = self.system.l_unpack(l_flat) 
    x,dx = self.system.s_unpack(s_flat)

    # Contact forces
    for i in range(t.shape[0]):
      cost += 0.5 * l[i,:] @ R @ l[i,:].T
  
    for i in range(t.shape[0]):
      # Add the current distance to goal
      x_curr = x[i, :]
      delta = x_goal - x_curr
      cost += 0.5 * delta @ Q @ delta.T

    return cost

  """
  Formulates collocation constraints
  """
  def get_constraints(self,system,t,s,l):
    ds = system.dynamics(s,l)
    
    x,dx = system.s_unpack(s)
    new_dx,ddx = system.s_unpack(ds)

    # Separate x and new_dx into position and orientation (quaternion) so we can normalize quaterions
    pos = x[:, 0:3]
    new_dpos = new_dx[:, 0:3]
    quat = x[:, 3:]
    new_dquat = new_dx[:, 3:]

    print(ddx)

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
        print("new_dx, x, t{}, dim{}: {}".format(i,j,f))
        g.append(f)
        lbg.append(0)
        ubg.append(0)

      # Handle orientation and angular velocity - normalize first
      quat_i_plus_one = 0.5 * dt * (new_dquat[i+1,:] + new_dquat[i,:]) + quat[i,:]
      quat_i_plus_one_unit = quat_i_plus_one / norm_2(quat_i_plus_one)
      for j in range(4):
        # dx
        f = quat_i_plus_one_unit[0,j] - quat[i+1, j]
        print("new_dquat, quat, t{}, dim{}: {}".format(i,j,f))
        g.append(f)
        lbg.append(0)
        ubg.append(0)

      # velocity - acceleration
      # iterate over all dofs
      for j in range(system.dx_dim):
        f = 0.5 * dt * (ddx[i+1,j] + ddx[i,j]) + dx[i,j] - dx[i+1,j]
        print("dx, ddx, t{}, dim{}: {}".format(i,j,f))
        g.append(f)
        lbg.append(0)
        ubg.append(0)

    # Unit quaternion constraints
    #unit_quat_constraints = system.unit_quat_constraint(s)
    #for r in range(unit_quat_constraints.shape[0]):
    #  for c in range(unit_quat_constraints.shape[1]):
    #    print(unit_quat_constraints[r,c])
    #    g.append(unit_quat_constraints[r,c])
    #    lbg.append(-1e-3)
    #    ubg.append(1e-3)

    # Linearized friction cone constraints
    f_constraints = system.friction_cone_constraints(l)
    for r in range(f_constraints.shape[0]):
      for c in range(f_constraints.shape[1]):
        g.append(f_constraints[r,c])
        lbg.append(0)
        ubg.append(np.inf)

    return vertcat(*g), vertcat(*lbg), vertcat(*ubg)
    
