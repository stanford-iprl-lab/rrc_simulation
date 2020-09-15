import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import argparse
import os

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

def plot_traj(npz_file_string, save_string, save_plot=True, save_anim=True):
  # Open .npz file
  npzfile = np.load(npz_file_string)

  fnum    = npzfile["fnum"]
  qnum    = npzfile["qnum"]
  x_dim   = npzfile["x_dim"]
  dx_dim  = npzfile["dx_dim"]
  t_soln  = npzfile["t"]
  x_goal  = npzfile["x_goal"]
  x_soln  = npzfile["x"]
  dx_soln = npzfile["dx"]
  l_soln  = npzfile["l"]
  l_wf_soln  = npzfile["l_wf"]
  dt      = npzfile["dt"]

  l_i = 3

  print("*****************************************************************")
  print("PLOTTING")
  print("*****************************************************************")
  # Plots
  mpl.rcParams["figure.figsize"] = [20.0, 16.0] 
  mpl.rcParams["legend.fontsize"] = "small"
  mpl.rcParams["xtick.labelsize"] = "small"
  mpl.rcParams["ytick.labelsize"] = "small"
  mpl.rcParams["axes.labelsize"] = 10
  mpl.rcParams["axes.titlepad"] = 14
  #mpl.rcParams["figure.dpi"] = 400
  nr = 3
  nc = 4
  plt.figure()
  plt.suptitle("Goal pose [x,y,theta]: {}".format(x_goal))
  #plt.suptitle("Goal pose [x,y,theta]: {}\nExpected final joint angle: {} rad".format(x_goal, theta_goal))
  
  plt.subplots_adjust(hspace=0.3)
  
  # Object trajectory - position
  plt.subplot(nr, nc, 1)
  plt_dim = 0
  for i in range(0,3):
    plt.plot(t_soln,x_soln[:,i],'.-',label="Dimension {}".format(i))
  plt.title("Object trajectory")
  plt.xlabel("time (s)")
  plt.ylabel("y")
  plt.ticklabel_format(useOffset=False)
  plt.legend()

  # Object trajectory - quaternion
  plt.subplot(nr, nc, 2)
  plt_dim = 0
  for i in range(3,x_dim):
    plt.plot(t_soln,x_soln[:,i],'.-',label="Dimension {}".format(i))
  plt.title("Object trajectory - orientation")
  plt.xlabel("time (s)")
  plt.ylabel("y")
  plt.ticklabel_format(useOffset=False)
  plt.legend()
  
  # Object linear velocity
  plt.subplot(nr, nc, 3)
  for i in range(0,3):
    plt.plot(t_soln,dx_soln[:,i],'.-',label="Dimension {}".format(i))
  plt.title("Object velocity")
  plt.xlabel("time (s)")
  plt.ylabel("vel (unit/s)")
  plt.ticklabel_format(useOffset=False)
  plt.legend()

  # Object angular velocity
  plt.subplot(nr, nc, 4)
  for i in range(3,dx_dim):
    plt.plot(t_soln,dx_soln[:,i],'.-',label="Dimension {}".format(i))
  plt.title("Object angular velocity")
  plt.xlabel("time (s)")
  plt.ylabel("vel (unit/s)")
  plt.ticklabel_format(useOffset=False)
  plt.legend()
  
  # Norml Contact forces
  plt.subplot(nr, nc, 5)
  for i in range(fnum):
    plt.plot(t_soln,l_soln[:,i*l_i],'.-',label="Contact {}".format(i))
  plt.legend()
  plt.xlabel("time (s)")
  plt.title("Normal Contact forces")
  plt.ticklabel_format(useOffset=False)
  
  # Contact forces - y
  plt.subplot(nr, nc, 6)
  for i in range(fnum):
    plt.plot(t_soln,l_soln[:,i*l_i+1],'.-',label="Contact {}".format(i))
  plt.legend()
  plt.xlabel("time (s)")
  plt.title("Y direction Contact forces")
  plt.ticklabel_format(useOffset=False)
  
  # Contact forces - z
  plt.subplot(nr, nc, 7)
  for i in range(fnum):
    plt.plot(t_soln,l_soln[:,i*l_i+2],'.-',label="Contact {}".format(i))
  plt.legend()
  plt.xlabel("time (s)")
  plt.title("Z direction Contact forces")
  plt.ticklabel_format(useOffset=False)

  # Contact forces in world frame
  # Normal Contact forces
  plt.subplot(nr, nc, 9)
  for i in range(fnum):
    plt.plot(t_soln,l_wf_soln[:,i*l_i],'.-',label="Contact {}".format(i))
  plt.legend()
  plt.xlabel("time (s)")
  plt.title("WORLD FRAME X ontact forces")
  plt.ticklabel_format(useOffset=False)
  
  # Contact forces - y
  plt.subplot(nr, nc, 10)
  for i in range(fnum):
    plt.plot(t_soln,l_wf_soln[:,i*l_i+1],'.-',label="Contact {}".format(i))
  plt.legend()
  plt.xlabel("time (s)")
  plt.title("WORLD FRAME Y direction contact forces")
  plt.ticklabel_format(useOffset=False)
  
  # Contact forces - z
  plt.subplot(nr, nc, 11)
  for i in range(fnum):
    plt.plot(t_soln,l_wf_soln[:,i*l_i+2],'.-',label="Contact {}".format(i))
  plt.legend()
  plt.xlabel("time (s)")
  plt.title("WORLD FRAME Z direction ontact forces")
  plt.ticklabel_format(useOffset=False)

  # Contact forces - moment around x
  #plt.subplot(nr, nc, 8)
  #for i in range(fnum):
  #  plt.plot(t_soln,l_soln[:,i*l_i+3],'.-',label="Contact {}".format(i))
  #plt.legend()
  #plt.xlabel("time (s)")
  #plt.title("Moment around x contact forces")
  #plt.ticklabel_format(useOffset=False)

  if save_plot:
    plt.savefig("{}.png".format(save_string))
  
def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--npz_file")
  parser.add_argument("--save", help="Boolean for whether or not to save plots", type=str2bool, default=True)
  args = parser.parse_args()

  npz_file = args.npz_file
  save_plot = args.save
  save_anim = args.save

  save_string = os.path.splitext(npz_file)[0]
  print(save_string)

  plot_traj(npz_file, save_string, save_plot, save_anim)
  
if __name__ == "__main__":
  main()
