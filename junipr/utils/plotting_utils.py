"""Plotting utilities for JUNIPR"""
from __future__ import absolute_import

import numpy as np
from matplotlib import pyplot as plt 
from junipr.config import *

__all__ = ['t_label', 'p_end', 'p_mother', 'p_z', 'p_theta', 'p_phi', 'p_delta',
           'PLOT_END_SETTINGS', 'PLOT_MOTHTER_SETTINGS', 'PLOT_BRANCH_SETTINGS'
          ]

############################
## Global config settings ##
############################


plt.rc('text', usetex=False)
plt.rc('font', family='serif', size=12)
plt.rc('font', size=12)
plt.rc('axes', labelsize=12)
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)
plt.rc('legend', fontsize=12, handlelength=0.7, handletextpad=0.4, borderaxespad=0.3, labelspacing=0.5, borderpad=0.4, frameon=False, framealpha=0.15, facecolor='gray', fancybox=False)
plt.rc('figure', titlesize=12)
plt.rc('figure', figsize=(4, 3))


def t_label(t):
    if t is None:
        return ""
    else:
        return "(t={})".format(t)
    
PLOT_END_SETTINGS = {'alpha':0.5, 'width':1}
PLOT_MOTHTER_SETTINGS = {'alpha':0.5, 'width':1}
PLOT_BRANCH_SETTINGS = {'alpha':0.5, 'align':'edge'}

#########
## End ##
#########

def p_end(endings_out, ending_counts_out):
    t_axis      = range(len(endings_out))
    numerator   = endings_out.flatten()
    denominator = np.clip(ending_counts_out, 0.1, np.inf).flatten()
    return t_axis, numerator/denominator
    
    
############
## Mother ##
############

def p_mother(mothers_out, mother_counts_out, t=None):
    t_axis = range(len(mothers_out))
    
    if t is not None:
        numerator   = mothers_out[t]
        denominator = np.clip(mother_counts_out[t], 0.1, np.inf)
        return t_axis, numerator/denominator
    else:
        # Average over all timesteps
        numerator   = np.sum(mothers_out, axis = 0)
        denominator = np.clip(np.sum(mother_counts_out, axis = 0), 0.1, np.inf)
        return t_axis, numerator/denominator
        
        
############
## Branch ##
############


def p_branch(i, branchings_out, branchings_counts_out, t=None):
    shift = [Z_SHIFT, BRANCH_THETA_SHIFT, PHI_SHIFT, DELTA_SHIFT][i]
    scale = [Z_SCALE, BRANCH_THETA_SCALE, PHI_SCALE, DELTA_SCALE][i]
    if i == 2:
        x_axis = np.linspace(0,1,11)*scale+shift
    else:
        x_axis = np.exp(np.linspace(0,1,11)*scale+shift)
        
    if i == 2:
        x_axis = np.linspace(0,1,11)*scale+shift
    else:
        x_axis = np.exp(np.linspace(0,1,11)*scale+shift)
        
    width  = np.diff(x_axis)
    avg_axis = [0,1,2,3]
    avg_axis.remove(i)
    avg_axis = tuple(avg_axis)
    
    if t is not None:
        numerator   = branchings_out[t]
        denominator = np.sum(numerator)
        return x_axis[:-1], numerator/denominator, width
    else:
        # Average over all timesteps
        t_average = np.sum(branchings_out, axis = 0)
        numerator   = t_average
        denominator = np.sum(numerator)
        return x_axis[:-1], numerator/denominator, width

def p_z(branchings_out, branchings_counts_out, t=None):
    return p_branch(0, branchings_out, branchings_counts_out, t=t)
    
def p_theta(branchings_out, branchings_counts_out, t=None):
    return p_branch(1, branchings_out, branchings_counts_out, t=t)
    
def p_phi(branchings_out, branchings_counts_out, t=None):
    return p_branch(2, branchings_out, branchings_counts_out, t=t)
    
def p_delta(branchings_out, branchings_counts_out, t=None):
    return p_branch(3, branchings_out, branchings_counts_out, t=t)
        

"""
#####################
## Mother vs angle ##
#####################

def p_mother_vs_angle(mothers_vs_angle, t=None):
    x_axis = np.linspace(r_sub, r_jet, mothers_vs_angle.shape[1] + 1)
    x_axis = (x_axis + np.diff(x_axis)[0])[:-1]
    if t is not None:
        numerator   = mothers_vs_angle[t]
        denominator = np.clip(np.sum(mothers_vs_angle[t]), 0.1, np.inf)
        return x_axis, numerator/denominator
    else:
        # Average over all timesteps
        numerator   = np.sum(mothers_vs_angle, axis = 0)
        denominator = np.clip(np.sum(numerator), 0.1, np.inf)
        return x_axis, numerator/denominator
"""        
        
        
        