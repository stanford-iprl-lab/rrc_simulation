import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

import argparse
import os

from rrc_simulation.tasks import move_cube

"""
Plot trajectory from npz file
"""

# Handle arguments
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def plot_traj(npz_file_string, save_string, save_plot=True):
  # Open .npz file
  npzfile = np.load(npz_file_string)

  fnum    = npzfile["fnum"]
  qnum    = npzfile["qnum"]
  t_soln  = npzfile["t"]
  q_soln  = npzfile["q"]
  dq_soln = npzfile["dq"]
  dt      = npzfile["dt"]
  ft_pos  = npzfile["ft_pos"]
  #ft_goal_list = npzfile["ft_goal_list"]
  
  print("*****************************************************************")
  print("PLOTTING")
  print("*****************************************************************")
  # Plots
  mpl.rcParams["figure.figsize"] = [10.0, 10.0] 
  mpl.rcParams["legend.fontsize"] = "small"
  mpl.rcParams["xtick.labelsize"] = "small"
  mpl.rcParams["ytick.labelsize"] = "small"
  mpl.rcParams["axes.labelsize"] = 10
  mpl.rcParams["axes.titlepad"] = 14

  plt.figure()
  ax = plt.axes(projection="3d")
  #ax.set_aspect('equal')

  color_list = ['r','g','b']
  plt.xlabel("x")
  plt.ylabel("y")

  # Plot trajectories and goal points
  for f_i in range(fnum):
    x = ft_pos[:, 3 * f_i]
    y = ft_pos[:, 3 * f_i + 1]
    z = ft_pos[:, 3 * f_i + 2]
    ax.scatter3D(x,y,z,c=color_list[f_i])

    #xg = ft_goal_list[f_i][0,0]
    #yg = ft_goal_list[f_i][1,0]
    #zg = ft_goal_list[f_i][2,0]
    #ax.scatter3D(xg, yg, zg,marker='*',c=color_list[f_i])

  # Plot cube corners
  corners = move_cube.get_cube_corner_positions(move_cube.Pose())
  ax.scatter3D(corners[:,0], corners[:,1], corners[:,2])

  plt.show()

  if save_plot:
    plt.savefig("{}.png".format(save_string))
  
def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--npz_file")
  parser.add_argument("--save", help="Boolean for whether or not to save plots", type=str2bool, default=True)
  args = parser.parse_args()

  npz_file = args.npz_file
  save_plot = args.save

  save_string = os.path.splitext(npz_file)[0]
  print(save_string)

  plot_traj(npz_file, save_string, save_plot)
  
if __name__ == "__main__":
  main()
