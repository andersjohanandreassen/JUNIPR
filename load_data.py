import os
import numpy as np
import tensorflow as tf
K = tf.keras
import pickle

from feature_scaling import *

###########################################################
# Begin Preprocessing

def preprocess_endings(endings_string):
    # Endings string gives the length of the jet
    # convert to categorical and change type to bool
    return np.expand_dims(np.asarray(K.utils.to_categorical(endings_string), dtype=np.bool)[0], axis=-1)
    
def preprocess_mothers(mothers_string, dim_mother_out=100):
    # mothers_string is a list of indicies as strings indicating which mother to branch (in an energy ordered list)
    mothers_int = np.asarray(mothers_string, dtype=np.int)
    # convert to categorical where the number of classes increases by one per time step as the number of candidates increases
    # change type to bool
    mothers = [np.asarray(K.utils.to_categorical(m, num_classes=t+1), dtype='bool') for t, m in enumerate(mothers_int)] + [[False]]
    return K.preprocessing.sequence.pad_sequences(mothers, dtype=np.bool, padding = 'post', maxlen=dim_mother_out)

def preprocess_mother_weights(mothers_string, dim_mother_out=100):
    # mothers_string is a list of indicies as strings indicating which mother to branch (in an energy ordered list)
    mothers_int = np.asarray(mothers_string, dtype=np.int)
    # create list of weights (dtype bool) for each timestep
    weights = [np.ones(t+1, dtype='bool') for t in range(len(mothers_int))] + [[False]]
    return K.preprocessing.sequence.pad_sequences(weights, padding='post', dtype=np.bool, maxlen=dim_mother_out)

def preprocess_daughters(daughters_string, dim_mom=4):
    # daughters string is a list of daughter momenta (as a string) - 8 numbers per timestep corresponding to the two daughters
    # Since first time step does not have daughters, we will pad by zeros. 
    
    # Convert string to float and reshape momenta.
    # Feature scale momenta
    daughters = np.asarray([shift_mom(mom) for mom in np.asarray(daughters_string, dtype=np.float).reshape(-1,4)[:,0:dim_mom]])
    return daughters.reshape(-1, dim_mom*2)

def preprocess_mother_momenta(mother_momenta_string, dim_mom=4):
    # mother momenta is a list of the momenta for the particle that will branch next (as a string) - 4 numbers
    # since no particle splits in the last time step, we pad with '-1'
    mother_momenta = [shift_mom(mom) for mom in np.asarray(mother_momenta_string, dtype=np.float).reshape(-1,4)[:,0:dim_mom]] + [np.array([-1]*dim_mom)]
    return np.asarray(mother_momenta)

def preprocess_branchings(branching_string, granularity = 10):
    # branchings is a list of strings giving the branching for each time step (z, theta, phi, delta)
    # Discretize branching to index in granularity**4 grid
    return [[branching_to_i(branch, granularity)] for branch in np.asarray(branching_string, dtype=np.float).reshape(-1,4)] + [[granularity**4]]

def preprocess_branching_weights(branching_string, granularity = 10):
    # branchings is a list of strings giving the branching for each time step (z, theta, phi, delta)
    # Discretize branching to index in granularity**4 grid
    return [[True] for branch in range(len(branching_string)//4)] + [[False]]

def preprocess_seed_momentum(seed_momentum_string, dim_mom = 4):
    # seed_momentum_string is a list of strings (px, py, pz, E)
    px, py, pz, E = np.asarray(seed_momentum_string, dtype=np.float)
    mass = np.sqrt(E**2-px**2-py**2-pz**2)
    return shift_mom([E, 0, 0, mass])[:dim_mom]

# End Preprocessing
#################################################################

#################################################################
################################################################# 

def read_in_jets(data_path, n_events, skip_first = 0, dim_mom = 4, dim_mother_out = 100):
    """ Read in jets from fastjet and convert them to numpy arrays"""
    
    with open(data_path, 'r') as f:
        data = [next(f).split() for x in range((n_events+skip_first)*7)]
        # jet_numbers    = [np.asarray(i[1:], dtype=np.int)   for i in data[skip_first*7::7]] # not used currently
        seed_momenta   = [preprocess_seed_momentum(i[1:], dim_mom) for i in data[skip_first*7+1::7]]
        endings        = [preprocess_endings(i[1:]) for i in data[skip_first*7+2::7]]
        ending_weights = [np.ones_like(i) for i in endings]
        
        mothers        = [preprocess_mothers(i[1:], dim_mother_out)  for i in data[skip_first*7+3::7]]
        mother_weights = [preprocess_mother_weights(i[1:], dim_mother_out)  for i in data[skip_first*7+3::7]]
        
        sparse_branchings        = [preprocess_branchings(i[1:]) for i in data[skip_first*7+4::7]]
        sparse_branching_weights = [preprocess_branching_weights(i[1:]) for i in data[skip_first*7+4::7]]
        
        daughters      = [preprocess_daughters(i[1:], dim_mom) for i in data[skip_first*7+5::7]]
        mother_momenta = [preprocess_mother_momenta(i[1:], dim_mom) for i in data[skip_first*7+6::7]]
        return [seed_momenta, daughters, mother_momenta, endings, ending_weights, mothers, mother_weights, sparse_branchings, sparse_branching_weights]
    
    
def pad_batched_data(data, i, granularity = 10):
    pad_values   = [-1, -1, False, False, False, False, granularity**4, False]
    dtypes       = [np.float32, np.float32, np.bool, np.bool, np.bool, np.bool, np.int32, np.bool]
    return [K.preprocessing.sequence.pad_sequences(batch, padding='post', dtype=dtypes[i], value=pad_values[i]) for batch in data]

def batch_data(data, batch_size, granularity = 10, dim_mom = 4):
    batched_data = [np.asarray(data[0]).reshape((-1, batch_size, 1, dim_mom))]
    for data_i, d in enumerate(data[1:]):
        # Batch and pad each of the data sets
        batched_data.append(pad_batched_data([d[batch_size*batch_i:batch_size*(batch_i+1)] for batch_i in range(len(d)//batch_size)], data_i, granularity=granularity))
    return batched_data

def load_data(data_path, n_events, batch_size, granularity, dim_mom = 4, dim_mother_out = 100, skip_first = 0, pickle_dir = './input_data/pickled', reload = False, verbose = True):
    """ Load data from fastjet and convert it into batched and padded numpy arrays """
    
    # Check if pickle_dir exists, or create it if not. 
    if not os.path.exists(pickle_dir):
        os.makedirs(pickle_dir)
    
    if skip_first>0:
        SKIP = "_SKIP"+str(skip_first)
    else:
        SKIP = ""
        
    data_basename = os.path.splitext(os.path.basename(data_path))[0]
    save_path = pickle_dir + '/'+ data_basename + '_N' + str(n_events) + SKIP + '_D' + str(dim_mom) + '_BS' + str(batch_size) +'_G' + str(granularity) + '_DM' + str(dim_mother_out) + '.pickled'
    # If not forced to reload data, see if pickled data alread exists
    if not reload and os.path.exists(save_path):
        if verbose:
            print('Getting pickled data from ' + save_path)
        with open(save_path, 'rb') as f:
                seed_momenta = pickle.load(f)
                daughters = pickle.load(f)
                mother_momenta = pickle.load(f)
                endings = pickle.load(f)
                ending_weights = pickle.load(f)
                mothers = pickle.load(f)
                mother_weights = pickle.load(f)
                sparse_branchings = pickle.load(f)
                sparse_branching_weights = pickle.load(f)
    else:
        if verbose:
            print('Loading Jets')
        # Load data
        data = read_in_jets(data_path, n_events, dim_mom = dim_mom, dim_mother_out = dim_mother_out)
        batched_data = batch_data(data, batch_size = batch_size, granularity = granularity, dim_mom = dim_mom)
        [seed_momenta, daughters, mother_momenta, endings, ending_weights, mothers, mother_weights, sparse_branchings, sparse_branching_weights] = batched_data
        
        # Pickle batched data
        with open(save_path, 'wb') as f:
            for item in [seed_momenta, daughters, mother_momenta, endings, ending_weights, mothers, mother_weights, sparse_branchings, sparse_branching_weights]:
                pickle.dump(item, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    return [seed_momenta, daughters, mother_momenta, endings, ending_weights, mothers, mother_weights, sparse_branchings, sparse_branching_weights]
