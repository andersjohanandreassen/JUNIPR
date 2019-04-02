"""Implementation of JUNIPR Network. """

__all__ = ['JUNIPR']

import tensorflow as tf 
tfK = tf.keras

import numpy as np
#from random import shuffle
from junipr.config import *


class JUNIPR:
    """ tf.keras implementation of JUNIPR from arXiv:1804.09720 using a Simple RNN """
    
    def __init__(self, 
                 label = None, 
                 dim_mom = 4, 
                 dim_end_hid = 10, dim_mother_hid = 10, dim_mother_out = 100, dim_branch_hid = 10, dim_RNN = 10, 
                 optimizer = None, 
                 RNN_type = None,
                 model_path = None):
        
        ### Define global parameters
        
        # Label
        if label is not None:
            self.label = '_' + label
        else:
            self.label = ""
                
        # Momenta
        self.dim_mom        = dim_mom
        self.dim_daughters  = 2*self.dim_mom
        self.dim_branch     = 4 # dimensionality of the branching coordinates x = (z, theta, phi, delta)
        
        # Network Parameters
        self.dim_end_hid    = dim_end_hid
        self.dim_mother_hid = dim_mother_hid
        self.dim_mother_out = DIM_M # This also sets the maximum shower length
        self.max_time       = DIM_M
        self.dim_branch_hid = dim_branch_hid
        self.dim_branch_out = GRANULARITY + 1 # +1 is for padded value
        self.dim_RNN        = dim_RNN
        
        # Masking values
        self.d_mask = D_PAD
        self.m_mask = M_PAD
        self.b_mask = B_PAD
        
        # Learning rate 
        self.lr = 1e-2
                
        # Optimizer
        if optimizer is None:
            self.optimizer = 'SGD'
        else:
            self.optimizer = optimizer
            
        # RNN_type
        self.RNN_type = RNN_type
        
        ### Get model
        self.model = self.model()
        
        if model_path is not None:
            self.load_model(model_path)
        
    def normalize_layer(self, x):
        """ Activation function that normalizes the output of a layer. 
        Assumes that all values are in the range [0,1] """
        total = tfK.backend.clip(tfK.backend.sum(x, axis=-1, keepdims=True), tfK.backend.epsilon()**2, 1)
        return x/total
            
    def categorical_crossentropy_mother(self, target, output):
        """ categorical_crossentropy where rows with just zeros are ignored """
        # Normalize output ignoring rows of just zeros
        sums = tfK.backend.sum(output, axis=-1, keepdims=True)
        sums = tfK.backend.clip(sums, tfK.backend.epsilon(), 1)
        output = output/sums
        # Explicitly calculate categorical_crossentropy
        output = tfK.backend.clip(output, tfK.backend.epsilon(), 1) # Clip to avoid nan from log(zero) from padding
        
        return -tfK.backend.sum(tf.cast(target, tf.float32)*tfK.backend.log(output), axis=-1) 
    
    def mask_input(self):
        # Masking
        self.masked_input_seed_momenta   = tfK.layers.Masking(mask_value=D_PAD, name = 'masked_input_seed_momenta' + self.label)(self.input_seed_momenta_exp)
        self.masked_input_daughters      = tfK.layers.Masking(mask_value=D_PAD, name = 'masked_input_daughter_momenta' + self.label)(self.input_daughters)
        self.masked_input_mother_momenta = tfK.layers.Masking(mask_value=M_PAD, name = 'masked_input_mother_momenta' + self.label)(self.input_mother_momenta)
        self.masked_input_branch_z       = tfK.layers.Masking(mask_value=B_PAD, name = 'masked_input_branchings_z' + self.label)(self.input_branch_z)
        self.masked_input_branch_t       = tfK.layers.Masking(mask_value=B_PAD, name = 'masked_input_branchings_t' + self.label)(self.input_branch_t)
        self.masked_input_branch_d       = tfK.layers.Masking(mask_value=B_PAD, name = 'masked_input_branchings_d' + self.label)(self.input_branch_d)
    
    def get_RNN(self):
        # Choose RNN cell
        if self.RNN_type == 'LSTM':
            self.RNN_cell = tfK.layers.LSTM
            print("Using LSTM Network")
        elif self.RNN_type == 'GRU':
            self.RNN_cell = tfK.layers.GRU
            print("Using GRU Network")
        else:
            self.RNN_cell = tfK.layers.SimpleRNN
            print("Using SimpleRNN Network")
        
        # Initialize RNN from seed momentum
        h_init_layer = tfK.layers.Dense(self.dim_RNN, activation = 'tanh', name = 'h_init' + self.label)
        h_init = h_init_layer(self.input_seed_momenta)
        rnn_0  = h_init_layer(self.masked_input_seed_momenta)
        if self.RNN_type == 'LSTM':
            c_init = tfK.layers.Dense(self.dim_RNN, activation = 'tanh', name = 'c_init' + self.label)(self.input_seed_momenta)
            initial_state = [h_init, c_init]
        else:
            initial_state = [h_init]

        # RNN for t>0
        rnn_t = self.RNN_cell(self.dim_RNN, name = 'RNN' + self.label, activation = 'tanh', return_sequences = True, bias_initializer = 'glorot_normal')(self.masked_input_daughters, initial_state = initial_state)
        
        RNN = tfK.layers.concatenate([rnn_0, rnn_t], axis=1, name = 'concatenate_RNN' + self.label)
        return RNN
    
    def build_end(self):
        self.end_hidden = tfK.layers.Dense(self.dim_end_hid, name = 'end_hidden_layer' + self.label, activation = 'relu')(self.RNN)
        self.end_output = tfK.layers.Dense(1,                name = 'endings' + self.label, activation = 'sigmoid')(self.end_hidden)
    
    def build_mother(self):
        self.mother_hidden = tfK.layers.Dense(self.dim_mother_hid, name = 'mother_hidden_layer' + self.label, activation = 'relu')(self.RNN)
        self.mother_unweighted_output = tfK.layers.Dense(self.dim_mother_out, name = 'mother_unweighted_output' + self.label, activation = 'softmax')(self.mother_hidden)
        self.mother_weighted_output = tfK.layers.multiply([self.input_mother_weights, self.mother_unweighted_output], name = 'multiply_weights' + self.label)
        self.mother_output = tfK.layers.Activation(self.normalize_layer, name = 'mothers' + self.label)(self.mother_weighted_output)
        
    def build_branch_X(self, concat_list, branch_label):
        branch_X_input  = tfK.layers.concatenate(concat_list, axis=-1 , name = 'concatinate_branch_'+branch_label+'_inputs' + self.label)
        branch_X_hidden = tfK.layers.Dense(self.dim_branch_hid, name = 'branch_'+branch_label+'_hidden_layer' + self.label, activation = 'relu')(branch_X_input)
        return tfK.layers.Dense(self.dim_branch_out, name = 'sparse_branchings_' +branch_label + self.label, activation = 'softmax')(branch_X_hidden)
    
    def model(self):
        """ Build model of JUNIPR """
        
        # Inputs to JUNIPR
        self.input_seed_momenta   = tfK.Input((self.dim_mom, ),            name = 'seed_momentum'    + self.label)
        self.input_daughters      = tfK.Input((None, self.dim_daughters),  name = 'daughter_momenta' + self.label)
        self.input_mother_momenta = tfK.Input((None, self.dim_mom),        name = 'mother_momenta'   + self.label)
        self.input_mother_weights = tfK.Input((None, self.dim_mother_out), name = 'mother_weights'   + self.label)
        self.input_branch_z       = tfK.Input((None, 1),                   name = 'branchings_z'     + self.label)
        self.input_branch_t       = tfK.Input((None, 1),                   name = 'branchings_t'     + self.label)
        self.input_branch_d       = tfK.Input((None, 1),                   name = 'branchings_d'     + self.label)
        
        _inputs = [self.input_seed_momenta, self.input_daughters, self.input_mother_momenta, self.input_mother_weights, self.input_branch_z, self.input_branch_t, self.input_branch_d]
        
        # Get copy of seed_momenta with same dimensions as RNN input
        self.input_seed_momenta_exp = tfK.layers.Lambda(lambda x: tfK.backend.expand_dims(x, axis = 1), name = 'input_seed_momenta_expand_dims' + self.label)(self.input_seed_momenta)
        
        self.mask_input()
        self.RNN = self.get_RNN()
        self.build_end()
        self.build_mother()
        
        # Branch networks
        self.branch_z_output = self.build_branch_X([self.RNN, 
                                                    self.masked_input_mother_momenta], 'z')
        self.branch_t_output = self.build_branch_X([self.RNN, 
                                                    self.masked_input_mother_momenta, 
                                                    self.masked_input_branch_z], 't')
        self.branch_d_output = self.build_branch_X([self.RNN, 
                                                    self.masked_input_mother_momenta, 
                                                    self.masked_input_branch_z, 
                                                    self.masked_input_branch_t], 'd')
        self.branch_p_output = self.build_branch_X([self.RNN, 
                                                    self.masked_input_mother_momenta, 
                                                    self.masked_input_branch_z, 
                                                    self.masked_input_branch_t, 
                                                    self.masked_input_branch_d], 'p')
        
        _outputs = [self.end_output, self.mother_output, self.branch_z_output, self.branch_t_output, self.branch_d_output, self.branch_p_output]
        
        return tfK.models.Model(inputs  = _inputs, 
                                outputs = _outputs, 
                                name = 'JUNIPR_' + self.label)

    def compile_model(self, learning_rate):
        self.lr = learning_rate
        
        # Select the optimizer to use
        if self.optimizer == 'Adam':
            the_optimizer = tfK.optimizers.Adam(lr = self.lr) 
            print("Compiling: Using Adam with learning rate" + str(self.lr) + ". Default value should be 1e-3.")
        elif self.optimizer == 'SGD':
            the_optimizer = tfK.optimizers.SGD(lr = self.lr)
            print("Compiling: Using SGD with learning rate " + str(self.lr))
        else:
            print("Unknown optimizer (" + self.optimizer + ") at JUNIPR.compile_model")
        
        self.model.compile(optimizer = the_optimizer, 
                                loss = ['binary_crossentropy', 
                                        self.categorical_crossentropy_mother, 
                                        'sparse_categorical_crossentropy',
                                        'sparse_categorical_crossentropy',
                                        'sparse_categorical_crossentropy',
                                        'sparse_categorical_crossentropy'])
    
"""
    def validate(self, data_path, n_events, predict, model_path = None, batch_size = 100, granularity = 10, label = '', skip_first = 0, in_pickle_dir='./input_data/pickled', out_pickle_dir='./output_data/pickled', reload = False, reload_input_data = False, verbose = True):
        # Check if pickle_dir exists, or create it if not. 
        if not os.path.exists(out_pickle_dir):
            os.makedirs(out_pickle_dir)
    
        if skip_first>0:
            SKIP = "_SKIP" + str(skip_first)
        else:
            SKIP = ""
        
        if model_path is None:
            model_basename = ""
        else:
            model_basename = os.path.basename(model_path)
            
        pickle_path = out_pickle_dir + '/' + label + model_basename + '_N' + str(n_events) + SKIP + '_D' + str(self.dim_mom) + '_BS' + str(batch_size) +'_G' + str(granularity) + '_DM' + str(self.dim_mother_out) + '_validate.pickled'
        
        # If not forced to reload data, see if pickled data alread exists
        if not reload and os.path.exists(pickle_path):
            if verbose:
                print('Getting pickled data from ' + pickle_path)
            with open(pickle_path, 'rb') as f:
                endings_out = pickle.load(f)
                ending_counts_out = pickle.load(f)
                mothers_out = pickle.load(f)
                mother_counts_out = pickle.load(f)
                branchings_z_out = pickle.load(f)
                branchings_t_out = pickle.load(f)
                branchings_p_out = pickle.load(f)
                branchings_d_out = pickle.load(f)
                branchings_counts_out = pickle.load(f)
                mother_vs_angle       = pickle.load(f)
                # Collect all outputs in one list
                outputs = [endings_out, ending_counts_out, mothers_out, mother_counts_out, branchings_z_out,branchings_t_out, branchings_p_out, branchings_d_out, branchings_counts_out, mother_vs_angle]
                # If predict, get probabilities
                if predict:
                    probabilities = pickle.load(f)
        
        else:
            if verbose:
                print('Did not find pickled data at ' + pickle_path + '. Validating model now.', flush = True)
            # Load model
            if model_path is not None:
                self.load_model(model_path)
        
            all_data = load_data(data_path, n_events = n_events, batch_size = batch_size, dim_mom = self.dim_mom, granularity = granularity, skip_first = skip_first, pickle_dir = in_pickle_dir, reload = reload_input_data, dim_mother_out = self.dim_mother_out)       
            [seed_momenta, daughters, mother_momenta, endings, ending_weights, mothers, mother_weights, branchings_z, branchings_t, branchings_p, branchings_d, sparse_branchings_z, sparse_branchings_t, sparse_branchings_p, sparse_branchings_d, branchings_weights] = all_data

        
            n_batches             = len(daughters)
        
            # Create ouput arrays
            endings_out           = np.zeros((self.max_time, 1))
            ending_counts_out     = np.zeros((self.max_time, 1))
        
            mothers_out           = np.zeros((self.max_time, self.dim_mother_out))
            mother_counts_out     = np.zeros((self.max_time, self.dim_mother_out))

            branchings_z_out        = np.zeros((self.max_time, self.granularity))
            branchings_t_out        = np.zeros((self.max_time, self.granularity))
            branchings_p_out        = np.zeros((self.max_time, self.granularity))
            branchings_d_out        = np.zeros((self.max_time, self.granularity))
            branchings_counts_out = np.zeros((self.max_time, 1))
            
            n_bins_mother_vs_angle = 10
            bin_edges_mother_vs_angle = np.linspace(r_sub, r_jet, n_bins_mother_vs_angle + 1)
            mother_vs_angle       = np.zeros((self.max_time, n_bins_mother_vs_angle))
        
            probabilities         = []
        
            # Organize outputs in one array
            outputs = [endings_out, ending_counts_out, mothers_out, mother_counts_out, branchings_z_out, branchings_t_out, branchings_p_out, branchings_d_out, branchings_counts_out, mother_vs_angle]

            # Predict on batches 
            for batch_i in range(n_batches):
                print_progress(batch_i, n_batches)
                if predict:
                    # Pretict on a single batch 
                    e, m, b_z, b_t, b_p, b_d = self.model.predict_on_batch(x = [seed_momenta[batch_i], daughters[batch_i], mother_momenta[batch_i], mother_weights[batch_i], branchings_z[batch_i], branchings_t[batch_i], branchings_d[batch_i]])
                else:
                    # Use one batch from input data
                    e   = endings[batch_i]
                    m   = mothers[batch_i]
                    b_z = sparse_branchings_z[batch_i]
                    b_t = sparse_branchings_t[batch_i]
                    b_p = sparse_branchings_p[batch_i]
                    b_d = sparse_branchings_d[batch_i]

                batch_length, max_time = e.shape[:2]
                for jet_i in range(batch_length):
                    # Initialize intermediate states array
                    intermediate_states = [unshift_mom(seed_momenta[batch_i][jet_i])]
                    
                    for t in range(max_time):

                        # Endings
                        if ending_weights[batch_i][jet_i, t] == 1:
                            endings_out[t]       += e[jet_i, t]
                            ending_counts_out[t] += 1

                        # Mothers
                        for mother_candidate_i in range(self.dim_mother_out):
                            if mother_weights[batch_i][jet_i, t][mother_candidate_i] == 1:
                                mothers_out[t, mother_candidate_i]       += m[jet_i, t, mother_candidate_i]
                                mother_counts_out[t, mother_candidate_i] += 1

                        # Branchings
                        if branchings_weights[batch_i][jet_i, t] == 1:
                            if predict:
                                branchings_z_out[t]                 += b_z[jet_i, t][:-1]
                                branchings_t_out[t]                 += b_t[jet_i, t][:-1]
                                branchings_p_out[t]                 += b_p[jet_i, t][:-1]
                                branchings_d_out[t]                 += b_d[jet_i, t][:-1]
                            else:
                                branchings_z_out[t, b_z[jet_i, t]]  += 1 # input data is sparse
                                branchings_t_out[t, b_t[jet_i, t]]  += 1 # input data is sparse
                                branchings_p_out[t, b_p[jet_i, t]]  += 1 # input data is sparse
                                branchings_d_out[t, b_d[jet_i, t]]  += 1 # input data is sparse

                            branchings_counts_out[t] += 1
                            
                        # Mothers vs angle
                        if predict:
                            ## Match probabilities and angles, and add up the probability per angle bin
                            for angle, prob in zip(np.asarray(intermediate_states)[:,1], m[jet_i][t][:t+1]):
                                mother_bin_index = get_bin_index(angle, bin_edges_mother_vs_angle)
                                mother_vs_angle[t, mother_bin_index] += prob
                        else:
                            mother_index = m[jet_i][t].argmax()
                            mother_vs_angle_bin_index = get_bin_index(intermediate_states[mother_index][1], bin_edges_mother_vs_angle)
                            mother_vs_angle[t, mother_vs_angle_bin_index] +=1
                        
                        # Add to intiermediate states array:
                        if t<max_time-1:
                            # Remove mother from intermediate state
                            mother_index = mothers[batch_i][jet_i][t].argmax()
                            del intermediate_states[mother_index]
                            
                            d1, d2 = [unshift_mom(d) for d in daughters[batch_i][jet_i][t].reshape(2,4)]
                            # Add daughters to intermediate state
                            intermediate_states.append(d1)
                            intermediate_states.append(d2)
                            intermediate_states.sort(key = lambda x: -x[0])
                        
                            

                    # Jet probabilities
                    if predict:
                        p_e = np.asarray(e[jet_i]*ending_weights[batch_i][jet_i]).flatten()
                        p_m = m[jet_i][mothers[batch_i][jet_i]]
                        branch_indices_z = tfK.utils.to_categorical(sparse_branchings_z[batch_i][jet_i], num_classes=granularity+1).astype('bool')
                        branch_indices_t = tfK.utils.to_categorical(sparse_branchings_t[batch_i][jet_i], num_classes=granularity+1).astype('bool')
                        branch_indices_p = tfK.utils.to_categorical(sparse_branchings_p[batch_i][jet_i], num_classes=granularity+1).astype('bool')
                        branch_indices_d = tfK.utils.to_categorical(sparse_branchings_d[batch_i][jet_i], num_classes=granularity+1).astype('bool')

                        p_b0 = (b_z[jet_i][branch_indices_z]*branchings_weights[batch_i][jet_i].T).flatten()
                        p_b1 = (b_t[jet_i][branch_indices_t]*branchings_weights[batch_i][jet_i].T).flatten()
                        p_b2 = (b_p[jet_i][branch_indices_p]*branchings_weights[batch_i][jet_i].T).flatten()
                        p_b3 = (b_d[jet_i][branch_indices_d]*branchings_weights[batch_i][jet_i].T).flatten()

                        length = len(p_m)
                        probs_e = np.append((1-p_e[:length]), p_e[length])
                        probs_m = p_m
                        probs_b0 = p_b0[:length]
                        probs_b1 = p_b1[:length]
                        probs_b2 = p_b2[:length]
                        probs_b3 = p_b3[:length]
                        probs_b = probs_b0*probs_b1*probs_b2*probs_b3
        
                        probslog10 = np.log10(1-p_e[:length], dtype=np.float64)+np.log10(p_m, dtype = np.float64)+np.log10(probs_b0, dtype = np.float64)+np.log10(probs_b1, dtype = np.float64)+np.log10(probs_b2, dtype = np.float64)+np.log10(probs_b3, dtype = np.float64)

                        probabilities.append(np.sum(np.append(probslog10, np.log10(p_e[length]))))
                        
            # Pickle outputs
            with open(pickle_path, 'wb') as f:
                for item in outputs:
                    pickle.dump(item, f, protocol=pickle.HIGHEST_PROTOCOL)
                if predict:
                    pickle.dump(np.asarray(probabilities), f, protocol=pickle.HIGHEST_PROTOCOL)

                            
        if predict:
            return outputs, np.asarray(probabilities)
        else:
            return outputs

    def load_model(self, path_to_saved_model):
        with tfK.utils.CustomObjectScope({'normalize_layer':self.normalize_layer,
                                        'binary_crossentropy_end': self.binary_crossentropy_end,
                                        'categorical_crossentropy_mother': self.categorical_crossentropy_mother,
                                        'sparse_categorical_crossentropy_branch': self.sparse_categorical_crossentropy_branch}):
            self.model = tfK.models.load_model(path_to_saved_model)


def get_bin_index(value, bin_edges):
    for i, edge in enumerate(np.asarray(bin_edges)[1:]):
        if(value < edge):
            return i
    return len(bin_edges)-2


"""
