import numpy as np

# Constants
epsilon = 1e-8
inf     = 1e8  

#####################
## Feature scaling ##
#####################

e_jet = 500.0
e_sub = 1
r_jet = np.pi / 2
r_sub = 0.1

e_shift = np.log(e_sub)
e_scale = np.log(e_jet) - e_shift

mass_shift = np.log(e_sub)
mass_scale = np.log(e_jet) - e_shift

mom_theta_shift = np.log(e_sub * r_sub / e_jet)
mom_theta_scale = np.log(r_jet) - mom_theta_shift

phi_shift = 0
phi_scale = 2 * np.pi - phi_shift

z_shift = np.log(e_sub / e_jet)
z_scale = np.log(0.5) - z_shift 

branch_theta_shift = np.log(r_sub / 2.0)
branch_theta_scale = np.log(r_jet) - branch_theta_shift 

delta_shift = np.log(e_sub * r_sub / e_jet)
delta_scale = np.log(r_jet / 2.0) - delta_shift


def shift_mom(mom):
    if len(mom)==3:
        """ Feature scaling momentum = (E, theta, phi) """
        e, theta, phi= mom
        e2     = (np.log(np.clip(e, epsilon, inf)) - e_shift) / e_scale
        theta2 = (np.log(np.clip(theta, epsilon, inf)) - mom_theta_shift) / mom_theta_scale
        phi2   = (phi - phi_shift) / phi_scale
        return np.asarray([e2, theta2, phi2])
    if len(mom)==4:
        """ Feature scaling momentum = (E, theta, phi, mass) """
        e, theta, phi, mass = mom
        e2     = (np.log(np.clip(e, epsilon, inf)) - e_shift) / e_scale
        theta2 = (np.log(np.clip(theta, epsilon, inf)) - mom_theta_shift) / mom_theta_scale
        phi2   = (phi - phi_shift) / phi_scale
        mass2  = (np.log(np.clip(mass, epsilon, inf)) - mass_shift) / mass_scale
        return np.asarray([e2, theta2, phi2, mass2])

def unshift_mom(mom2):
    if len(mom2)==3:
        """ Undo feature scaling momentum = (E, theta, phi) """
        e2, theta2, phi2, mass2 = mom2
        e     = np.exp(e2 * e_scale + e_shift)
        theta = np.exp(theta2 * mom_theta_scale + mom_theta_shift)
        phi   = phi2 * phi_scale + phi_shift
        return np.asarray([e, theta, phi])
    if len(mom2)==4:
        """ Undo feature scaling momentum = (E, theta, phi) """
        e2, theta2, phi2, mass2 = mom2
        e     = np.exp(e2 * e_scale + e_shift)
        theta = np.exp(theta2 * mom_theta_scale + mom_theta_shift)
        phi   = phi2 * phi_scale + phi_shift
        mass  = np.exp(mass2 * mass_scale + mass_shift)
        return np.asarray([e, theta, phi, mass])
    
def shift_branch(branch):
    """ Feature scaling branching = (z, theta, phi, delta) """
    z, theta, phi, delta = branch
    z2     = (np.log(np.clip(z, epsilon, inf)) - z_shift) / z_scale
    theta2 = (np.log(np.clip(theta, epsilon, inf)) - branch_theta_shift) / branch_theta_scale
    phi2   = (phi - phi_shift) / phi_scale
    delta2 = (np.log(np.clip(delta, epsilon, inf)) - delta_shift) / delta_scale
    return [z2, theta2, phi2, delta2]

def unshift_branch(branch2):
    """ Undo feature scaling branching = (z, theta, phi, delta) """
    z2, theta2, phi2, delta2 = branch2
    z     = np.exp(z2 * z_scale + z_shift)
    theta = np.exp(theta2 * branch_theta_scale + branch_theta_shift)
    phi   = phi2 * phi_scale + phi_shift
    delta = np.exp(delta2 * delta_scale + delta_shift)
    return [z, theta, phi, delta]
    
### Discretization

## branching -> i
def shifted_branching_to_ituple(shifted_branching, granularity):
    """ Convert branching to coordinates in (granularity, granularity, granularity, granularity) grid """
    z2, theta2, phi2, delta2 = shifted_branching # Each value is in range [0,1]
    width   = 1 / granularity
    i_z     = int(np.clip(z2     / width, 0, granularity-1))
    i_theta = int(np.clip(theta2 / width, 0, granularity-1))
    i_phi   = int(np.clip(phi2   / width, 0, granularity-1))
    i_delta = int(np.clip(delta2 / width, 0, granularity-1))
    ituple  = [i_z, i_theta, i_phi, i_delta]
    return ituple

def ituple_to_i(ituple, granularity):
    """ Convert from branching coordinate from (granularity, granularity, granularity, granularity) to (granularity**4)  """
    n = granularity
    i_z, i_theta, i_phi, i_delta = ituple
    i = i_z * n**3 + i_theta * n**2 + i_phi * n + i_delta
    return i
    
def branching_to_i(branching, granularity):
    """ Convert unshifted branching to index in (granularity**4) """
    branching2 = shift_branch(branching)
    ituple     = shifted_branching_to_ituple(branching2, granularity)
    return ituple_to_i(ituple, granularity)
    
def shifted_branching_to_i(branching2, granularity):
    """ Convert shifted branching to index in (granularity**4) """
    ituple     = shifted_branching_to_ituple(branching2, granularity)
    return ituple_to_i(ituple, granularity)    
    
## i -> branching
def i_to_ituple(i, granularity):
    """ Convert from index in (granularity**4) to index in (granularity, granularity, granularity, granularity) """
    n = granularity
    i_z, r     = np.divmod(i, n**3)
    i_theta, r = np.divmod(r, n**2)
    i_phi, r   = np.divmod(r, n)
    i_delta    = r
    ituple     = [i_z, i_theta, i_phi, i_delta]
    return ituple
    
def ituple_to_shifted_branching(ituple, granularity):
    """ Convert from  ituple in (granularity, granularity, granularity, granularity) to shifted branching """
    i_z, i_theta, i_phi, i_delta = ituple
    width   = 1 / granularity
    z2      = width * (i_z + 0.5)
    theta2  = width * (i_theta + 0.5)
    phi2    = width * (i_phi + 0.5)
    delta2  = width * (i_delta + 0.5)
    return z2, theta2, phi2, delta2

def i_to_shifted_branching(i, granularity):
    """ Convert index in (granularity**4) to shifted branching """
    ituple     = i_to_ituple(i, granularity)
    branching2 = ituple_to_shifted_branching(ituple, granularity)
    return branching2

def i_to_branching(i, granularity):
    """ Convert index in (granularity**4) to unshifted branching """
    ituple     = i_to_ituple(i, granularity)
    branching2 = ituple_to_shifted_branching(ituple, granularity)
    return unshift_branch(branching2)