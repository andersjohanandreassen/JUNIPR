"""Feature scaling functions for JUNIPR."""
from __future__ import absolute_import

from junipr.config import *

import numpy as np

# Cutoff constants
EPSILON = 1e-8
INF     = 1e8  

__all__ = ['shift_mom', 'unshift_mom', 'shift_branch', 'unshift_branch']

def shift_mom(mom):
    """Feature scale momentum."""
    
    if len(mom)==3:
        """ Feature scaling momentum = (E, theta, phi) """
        e, theta, phi = mom
        e2     = (np.log(np.clip(e, EPSILON, INF)) - E_SHIFT) / E_SCALE
        theta2 = (np.log(np.clip(theta, EPSILON, INF)) - MOM_THETA_SHIFT) / MOM_THETA_SCALE
        phi2   = (phi - PHI_SHIFT) / PHI_SCALE
        return np.asarray([e2, theta2, phi2])
    
    elif len(mom)==4:
        """ Feature scaling momentum = (E, theta, phi, mass) """
        e, theta, phi, mass = mom
        e2     = (np.log(np.clip(e, EPSILON, INF)) - E_SHIFT) / E_SCALE
        theta2 = (np.log(np.clip(theta, EPSILON, INF)) - MOM_THETA_SHIFT) / MOM_THETA_SCALE
        phi2   = (phi - PHI_SHIFT) / PHI_SCALE
        mass2  = (np.log(np.clip(mass, EPSILON, INF)) - MASS_SHIFT) / MASS_SCALE
        return np.asarray([e2, theta2, phi2, mass2])

def unshift_mom(mom2):
    if len(mom2)==3:
        """ Undo feature scaling momentum = (E, theta, phi) """
        e2, theta2, phi2, mass2 = mom2
        e     = np.exp(e2 * E_SCALE + E_SHIFT)
        theta = np.exp(theta2 * MOM_THETA_SCALE + MOM_THETA_SHIFT)
        phi   = phi2 * PHI_SCALE + PHI_SHIFT
        return np.asarray([e, theta, phi])
    if len(mom2)==4:
        """ Undo feature scaling momentum = (E, theta, phi) """
        e2, theta2, phi2, mass2 = mom2
        e     = np.exp(e2 * E_SCALE + E_SHIFT)
        theta = np.exp(theta2 * MOM_THETA_SCALE + MOM_THETA_SHIFT)
        phi   = phi2 * PHI_SCALE + PHI_SHIFT
        mass  = np.exp(mass2 * MASS_SCALE + MASS_SHIFT)
        return np.asarray([e, theta, phi, mass])
    
def shift_branch(branch):
    """ Feature scaling branching = (z, theta, phi, delta) """
    z, theta, phi, delta = branch
    z2     = (np.log(np.clip(z, EPSILON, INF)) - Z_SHIFT) / Z_SCALE
    theta2 = (np.log(np.clip(theta, EPSILON, INF)) - BRANCH_THETA_SHIFT) / BRANCH_THETA_SCALE
    phi2   = (phi - PHI_SHIFT) / PHI_SCALE
    delta2 = (np.log(np.clip(delta, EPSILON, INF)) - DELTA_SHIFT) / DELTA_SCALE
    return [z2, theta2, phi2, delta2]

def unshift_branch(branch2):
    """ Undo feature scaling branching = (z, theta, phi, delta) """
    z2, theta2, phi2, delta2 = branch2
    z     = np.exp(z2 * Z_SCALE + Z_SHIFT)
    theta = np.exp(theta2 * BRANCH_THETA_SCALE + BRANCH_THETA_SHIFT)
    phi   = phi2 * PHI_SCALE + PHI_SHIFT
    delta = np.exp(delta2 * DELTA_SCALE + DELTA_SHIFT)
    return [z, theta, phi, delta]
    
### Discretization

## branching -> i
#def shifted_branching_to_ituple(shifted_branching, granularity):
#    """ Convert branching to coordinates in (granularity, granularity, granularity, granularity) grid """
#    z2, theta2, phi2, delta2 = shifted_branching # Each value is in range [0,1]
#    width   = 1 / granularity
#    i_z     = int(np.clip(z2     / width, 0, granularity-1))
#    i_theta = int(np.clip(theta2 / width, 0, granularity-1))
#    i_phi   = int(np.clip(phi2   / width, 0, granularity-1))
#    i_delta = int(np.clip(delta2 / width, 0, granularity-1))
#    ituple  = [i_z, i_theta, i_phi, i_delta]
#    return ituple

#def ituple_to_i(ituple, granularity):
#    """ Convert from branching coordinate from (granularity, granularity, granularity, granularity) to (granularity**4)  """
#    n = granularity
#    i_z, i_theta, i_phi, i_delta = ituple
#    i = i_z * n**3 + i_theta * n**2 + i_phi * n + i_delta
#    return i
    
#def branching_to_i(branching, granularity):
#    """ Convert unshifted branching to index in (granularity**4) """
#    branching2 = shift_branch(branching)
#    ituple     = shifted_branching_to_ituple(branching2, granularity)
#    return ituple_to_i(ituple, granularity)
    
#def shifted_branching_to_i(branching2, granularity):
#    """ Convert shifted branching to index in (granularity**4) """
#    ituple     = shifted_branching_to_ituple(branching2, granularity)
#    return ituple_to_i(ituple, granularity)    
    
## i -> branching
#def i_to_ituple(i, granularity):
#    """ Convert from index in (granularity**4) to index in (granularity, granularity, granularity, granularity) """
#    n = granularity
#    i_z, r     = np.divmod(i, n**3)
#    i_theta, r = np.divmod(r, n**2)
#    i_phi, r   = np.divmod(r, n)
#    i_delta    = r
#    ituple     = [i_z, i_theta, i_phi, i_delta]
#    return ituple
    
#def ituple_to_shifted_branching(ituple, granularity):
#    """ Convert from  ituple in (granularity, granularity, granularity, granularity) to shifted branching """
#    i_z, i_theta, i_phi, i_delta = ituple
#    width   = 1 / granularity
#    z2      = width * (i_z + 0.5)
#    theta2  = width * (i_theta + 0.5)
#    phi2    = width * (i_phi + 0.5)
#    delta2  = width * (i_delta + 0.5)
#    return z2, theta2, phi2, delta2

#def i_to_shifted_branching(i, granularity):
#    """ Convert index in (granularity**4) to shifted branching """
#    ituple     = i_to_ituple(i, granularity)
#    branching2 = ituple_to_shifted_branching(ituple, granularity)
#    return branching2

#def i_to_branching(i, granularity):
#    """ Convert index in (granularity**4) to unshifted branching """
#    ituple     = i_to_ituple(i, granularity)
#    branching2 = ituple_to_shifted_branching(ituple, granularity)
#    return unshift_branch(branching2)