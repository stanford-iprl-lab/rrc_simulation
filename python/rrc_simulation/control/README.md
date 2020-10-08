### control_policy.py
Hierarchical controller class and impedance controller class

### controller_utils.py
Controller functions for impedance control, getting trajectories/waypoints, and assigning contact points

### control_trifinger_platform.py
[Probably outdated] Use run_eval.py script for unit testing control_policy

Test script for unit testing a user-specified init/goal pose. Does not use control_policy.py
Instantiates TriFinger platform, loads .npz file as trajectory, and runs controller to track trajectory (done in a super bad way atm).

To run, just load a .npz file with the arg --npz_file /path/to/file
