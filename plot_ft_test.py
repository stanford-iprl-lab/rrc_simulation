import pandas
import matplotlib.pyplot as plt
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("filename", type=str)
args = parser.parse_args()

fig_name = "{}".format(os.path.splitext(args.filename)[0])

data = pandas.read_csv(args.filename, delim_whitespace=False, header=0)

# Plot joint torques
plt.figure(figsize=(20,10))
plt.suptitle("Finger joint torques")
plt.subplots_adjust(hspace=0.3)
for f_i in range(3):
  for d_i in range(3):
    plt.subplot(3,3,f_i*3+d_i+1)
    plt.scatter(data[["step_count"]], data[["desired_torque_{}".format(3*f_i+d_i)]])
plt.savefig("{}_torques.png".format(fig_name))

# Plot fingertip positions
plt.figure(figsize=(20,10))
plt.suptitle("Fingertip positions")
plt.subplots_adjust(hspace=0.3)
for f_i in range(3):
  for d_i, dim in enumerate(["x","y","z"]):
    plt.subplot(3,3,f_i*3+d_i+1)
    plt.title("Finger {} dimension {}".format(f_i, dim))
    plt.scatter(data[["step_count"]], data[["ft_goal_{}".format(3*f_i+d_i)]], label="desired")
    plt.scatter(data[["step_count"]], data[["ft_current_{}".format(3*f_i+d_i)]], label="actual")
    plt.legend()
plt.savefig("{}_ft_pos.png".format(fig_name))

