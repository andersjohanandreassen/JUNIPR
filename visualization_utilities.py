from JUNIPR import *
from feature_scaling import *
from load_data import *
from matplotlib import pyplot as plt
import matplotlib.pylab as pl
from plotting_utilities import *
import numpy as np

###############
## Jet image ##
###############

def get_final_states(all_data):
    """Build the final states for a single jet (taken from read_in_jets and is not batched)"""
    
    [seed_momenta, daughters, mother_momenta, endings, ending_weights, mothers, mother_weights, sparse_branchings, sparse_branching_weights] = all_data
    # Build intermediate state:
    max_time = len(mother_momenta[0])
    
    # Add seed momentum
    intermediate_states = [unshift_mom(seed_momenta[0])]
    for t in range(max_time):
        # Add to intiermediate states array:
        if t<max_time-1:
            # Remove mother from intermediate state
            mother_index = mothers[0][t].argmax()
            del intermediate_states[mother_index]
            
            # Add daughters to intermediate state
            d1, d2 = [unshift_mom(d) for d in daughters[0][t].reshape(2,4)]
            intermediate_states.append(d1)
            intermediate_states.append(d2)
            intermediate_states.sort(key = lambda x: -x[0])          
    
    # intermediate_states at the last time step is the final states
    return intermediate_states


def plot_jet_image(jet_i, data_path, dim_mom = 4, dim_mother_out = 100, savefig = None):
    """ Make a plot of the jet image """
    
    # Get data for jet_i
    all_data_jet_i = read_in_jets(data_path, 1, skip_first = jet_i, dim_mom = dim_mom, dim_mother_out = dim_mother_out)
    
    # Get all the final states and their energies and angles
    final_states = np.asarray(get_final_states(all_data_jet_i))
    energy = final_states[:,0]
    theta = final_states[:,1]
    phi = final_states[:,2]
    
    # Plot settings
    colors =[pl.cm.jet(1-((np.log(e)-e_shift)/e_scale)) for e in energy]

    # Plot setup
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='polar')
    ax.scatter(phi, theta, c=colors, s=20)
    ax.set_rmax(0.6)
    ax.set_xticks([])
    ax.set_rticks([])
    ax.grid(False)
        
    if savefig is not None:
        plt.savefig(savefig)
        
    plt.show()
    
##############
## Jet tree ##
##############

# Helper functions
def get_dx_dy(theta, phi, theta0, r=1):
    phi = np.clip(phi %(2*np.pi), 1e-8, (2*np.pi))
    
    dx = r*np.cos(theta)
    dy = r*np.sin(theta)*np.sign(np.sin(phi))
    dxp, dyp =np.dot([[np.cos(theta0),-np.sin(theta0)],[np.sin(theta0),np.cos(theta0)]],[dx,dy]) # rotate relative to reference angle theta0
    
    return dxp, dyp

def get_angle(dx,dy):
    if (dx>0 and dy>0) or (dx>0 and dy<0):
        return np.arctan(dy/dx)
    if (dx<0 and dy<0) or (dx<0 and dy>0):
        return np.arctan(dy/dx)+np.pi
    else:
        return np.arctan(dy/dx)
    
    
def plot_jet_tree(jet_i, data_path, dim_mom = 4, dim_mother_out = 100, savefig = None, angle_scale_factor = 1, model_path_1 = None, model_path_2 = None):
    """ Plot the jet clustering tree. 
    If one model path is given, the P_jet is show at each node. 
    If two model paths are given, the likelihood ratio is shown at each node P_jet(model_1)/P_jet(model_2) 
    For jets with small jet radius and high multiplicity, angle_scale_factor can be used to plot the clustering tree with larger angles to reduce the overlap of lines and labels."""
    
    # plot setup:
    fig, ax = plt.subplots(1,figsize=(20,20))
    ax.axis('off')
    ax.set_aspect(1) # set equal aspect ratio
    r = 1 # length of each leg
    
    # Get data for jet_i
    all_data = read_in_jets(data_path, 1, skip_first = jet_i, dim_mom = dim_mom, dim_mother_out = dim_mother_out)
    [seed_momenta, daughters, mother_momenta, endings, ending_weights, mothers, mother_weights, sparse_branchings, sparse_branching_weights] = all_data
    zs, thetas, phis, deltas= np.asarray([i_to_branching(sparse_branching, 10) for sparse_branching in sparse_branchings[0][:-1]]).T[0]
    
    if model_path_1 is None and model_path_2 is not None:
        # if there is only one model given, we label it model_path_1 always! 
        model_path_1 = model_path_2
        model_path_2 = None
    
    if model_path_1 is not None: 
        # Get probabilities from model_1
        junipr = JUNIPR()
        junipr.load_model(model_path_1)
        e, m, b    = junipr.model.predict(x=[seed_momenta, daughters, mother_momenta, mother_weights])
        p_end_1    = 1-e.flatten()[:-1]
        p_mother_1 = m[0][mothers[0]]
        p_branch_1 = np.asarray([b[0,i,sb][0] for i, sb in enumerate(sparse_branchings[0][:-1])])
        
        labels = np.log10(p_end_1*p_mother_1*p_branch_1)
        
        p_last_end_1 = e.flatten()[-1]
    
    if model_path_2 is not None: 
        # Get probabilities from model_2
        junipr.load_model(model_path_2)
        e, m, b    = junipr.model.predict(x=[seed_momenta, daughters, mother_momenta, mother_weights])
        p_end_2    = 1-e.flatten()[:-1]
        p_mother_2 = m[0][mothers[0]]
        p_branch_2 = np.asarray([b[0,i,sb][0] for i, sb in enumerate(sparse_branchings[0][:-1])])
        
        labels_2 = np.log10(p_end_2*p_mother_2*p_branch_2)
        labels   = labels - labels_2
        
        p_last_end_2 = e.flatten()[-1]
    
    
    ### Step through the tree and build the intermediate state:
    
    max_time = len(mother_momenta[0])
    
    # Seed momentum
    # Plot initial particle
    x0 = 1
    y0 = 0
    theta0 = np.arctan(y0/x0)
    plt.plot([0, x0], [0, y0], linestyle='-',color=pl.cm.jet(1-((np.log(unshift_mom(seed_momenta[0])[0])-e_shift)/e_scale)))
    
    # intermediate_states is a tuple (momentum, (end_coordinate_x, end_coordinate_y), angle_relative_to_jet_axis)
    intermediate_states = [(unshift_mom(seed_momenta[0]), (x0, y0), theta0)]
    
    for t in range(max_time):
        # Add to intiermediate states array:
        if t<max_time-1:
            # Remove mother from intermediate state
            mother_index = mothers[0][t].argmax()
            
            # Extract end coordinates and angle from mother particle
            _, (x0, y0), theta0 = intermediate_states[mother_index]
            # delete mother from list of intermediate states
            del intermediate_states[mother_index]
            
            # Get coordinates from soft daughter
            dx2, dy2 = get_dx_dy(angle_scale_factor*thetas[t], phis[t], theta0, r=r)
            x2 = x0 + dx2
            y2 = y0 + dy2
            
            # Get coordinates from hard daughter
            dx1, dy1 = get_dx_dy(angle_scale_factor*deltas[t], phis[t]+np.pi, theta0, r=r)
            x1 = x0 + dx1
            y1 = y0 + dy1
            
            # get daughters momentum
            d1, d2 = [unshift_mom(d) for d in daughters[0][t].reshape(2,4)]
            
            
            # Get energies of daughters for coloring
            c2 = pl.cm.jet(1-((np.log(d2[0])-e_shift)/e_scale))
            c1 = pl.cm.jet(1-((np.log(d1[0])-e_shift)/e_scale))
            
            # Plot daughters
            plt.plot([x0, x1], [y0, y1], linestyle='-', color=c1)
            plt.plot([x0, x2], [y0, y2], linestyle='-', color=c2)
            
            # Add probability or likelihood ratio label
            if model_path_1 is not None:
                plt.text(x0+0.01,y0+0.01,'${:03.1f}$'.format(labels[t]),fontsize=13)
            
            # Add daughters to intermediate state
            intermediate_states.append((d1, (x1, y1), get_angle(dx1, dy1)))
            intermediate_states.append((d2, (x2, y2), get_angle(dx2, dy2)))
            intermediate_states.sort(key = lambda x: -x[0][0])
            
    if savefig is not None:
        plt.savefig(savefig)
            
    plt.show()