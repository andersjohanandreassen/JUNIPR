"""Configuration file for constants used by JUNIPR."""

__all__ = ['E_JET', 'E_SUB', 'R_JET', 'R_SUB', 
           'PHI_SHIFT', 'PHI_SCALE',
           'E_SHIFT', 'E_SCALE', 'MASS_SHIFT', 'MASS_SCALE', 'MOM_THETA_SHIFT', 'MOM_THETA_SCALE',
           'Z_SHIFT', 'Z_SCALE', 'BRANCH_THETA_SHIFT', 'BRANCH_THETA_SCALE', 'DELTA_SHIFT', 'DELTA_SCALE',
           'GRANULARITY',
           'B_PAD', 'SB_PAD', 'D_PAD', 'M_PAD', 'E_PAD', 
           'DIM_M']

import numpy as np

########################
### Variables to set ###
########################

# Granularity
GRANULARITY = 10 # number of bins in discretized branchings

# Feature scaling
E_JET = 500.0
E_SUB = 1
R_JET = np.pi / 2
R_SUB = 0.1

# Padding
B_PAD  = -99 # branching
SB_PAD = GRANULARITY # sparse branching 
D_PAD  = -1  # daughter momenta
M_PAD  = -1  # mother momenta
E_PAD  = 0 # endings and ending_weights

# Dimension of mother output layer
DIM_M = 100


#########################
### Derived variables ###
#########################

### Momentum

E_SHIFT = np.log(E_SUB)
E_SCALE = np.log(E_JET) - E_SHIFT

MOM_THETA_SHIFT = np.log(E_SUB * R_SUB / E_JET)
MOM_THETA_SCALE = np.log(R_JET) - MOM_THETA_SHIFT

PHI_SHIFT = 0
PHI_SCALE = 2 * np.pi - PHI_SHIFT

MASS_SHIFT = np.log(E_SUB)
MASS_SCALE = np.log(E_JET) - E_SHIFT


### Branching

Z_SHIFT = np.log(E_SUB / E_JET)
Z_SCALE = np.log(0.5) - Z_SHIFT 

BRANCH_THETA_SHIFT = np.log(R_SUB / 2.0)
BRANCH_THETA_SCALE = np.log(R_JET) - BRANCH_THETA_SHIFT 

# PHI same as for momentum

DELTA_SHIFT = np.log(E_SUB * R_SUB / E_JET)
DELTA_SCALE = np.log(R_JET / 2.0) - DELTA_SHIFT