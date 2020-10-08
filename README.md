Team Yoda Phase 1 Submission
===============================
## Setup:
1. Install the rrc_simulation package by following the steps on [this page](https://people.tuebingen.mpg.de/felixwidmaier/realrobotchallenge/simulation_phase/installation.html). Note that in step 1, instead of cloning their repo, you'll want to clone the IPRL fork, and git checkout phase_1_submission. This will create a conda env with all the necessary pacakages (except for spinning up, see step 2)
2. Install Krishnan's [fork of spinning up baselines](https://github.com/krishpop/spinningup/tree/rrc):
    - First, clone the repo (I'm not sure where the best place is to put it, but I've cloned it into rrc_simulation/scripts/, though if you put it into your rrc_simulation directory, please don't commit it!).
    - Do `pip install -e .` in the spinningup/ directory.
    - Make sure to `git checkout rrc` so that you're on the right branch.

## Running the code:
### Run full evaluation
Our final policy is in `scripts/evaluate_policy.py`.
To run their full evaluation procedure on this policy, run the `rrc_evaluate` script from the `scripts/` directory.

`./rrc_evaluate </path/to/output/dir>`

### Test with a single sample (best way to visualize a single run)
Use the script `run_eval.py` in the root directory to run the policy on one sample (from an output file)
To run:

python run_eval.py --dir /path/to/output/directory/ --l difficulty_level --i sample_number
    
flags:
-v: visualize

Real Robot Challenge Simulation
===============================

This repository is exclusively intended for participating
in the [simulation phase of the Real Robot Challenge](https://real-robot-challenge.com/simulation_phase).

For simulating the open-source version of the TriFinger robot, please use the 
repository linked on the official open-source [project website](https://sites.google.com/view/trifinger).


This repository contains a simulation, based on PyBullet, of the TriFinger robot
used in the Real Robot Challenge. In addition, it contains OpenAI gym
environments and evaluation scripts for the tasks of the simulation phase.

For more information on the challenge see:

- the [challenge website](https://real-robot-challenge.com)
- the [software
  documentation](https://people.tuebingen.mpg.de/felixwidmaier/realrobotchallenge/index.html)
  
  
## Authors
- Felix Widmaier
- Shruti Joshi
- Vaibhav Agrawal
- Manuel Wuethrich
