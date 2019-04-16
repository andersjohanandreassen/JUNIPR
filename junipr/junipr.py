"""Implementation of JUNIPR Network. """

__all__ = ['JUNIPR']

from packaging import version
import tensorflow as tf 
if version.parse(tf.__version__)<version.parse("2.0.0-alpha0"):
    raise ImportError("Please upgrade to Tensorflow 2.0.0-alpha0 or newer")

tfK = tf.keras

import numpy as np
import os
from junipr.config import *


class JUNIPR:
    """ tf.keras implementation of JUNIPR from arXiv:1804.09720 using a Simple RNN """
    
    def __init__(self, 
                 label = None,
                 dim_mom = 4, 
                 dim_end_hid = 10, dim_mother_hid = 10, dim_mother_out = DIM_M, dim_branch_hid = 10, dim_RNN = 10, 
                 optimizer = None, 
                 RNN_type = None,
                 model_path = None,
                 verbose = True):
        
        ### Define global parameters
        self.verbose = verbose
        
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
            
        self.custom_objects = {'normalize_layer'                : self.normalize_layer,
                               'categorical_crossentropy_mother': self.categorical_crossentropy_mother,
                               'binary_crossentropy_end'        : self.binary_crossentropy_end,
                               'sparse_categorical_crossentropy': self.sparse_categorical_crossentropy
                              }
        
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
    
    def binary_crossentropy_end(self, target, output, weights):
        w = weights[:,:,0]
        t = target
        return w*tf.losses.binary_crossentropy(t, output)
    
    def sparse_categorical_crossentropy(self, target, output, weights):
        w = weights[:,:,0]
        return w*tf.losses.sparse_categorical_crossentropy(target, output)
    
    def mask_input(self):
        # Masking
        self.masked_input_seed_momenta   = tfK.layers.Masking(mask_value=D_PAD, name = 'masked_input_seed_momenta')(self.input_seed_momenta_exp)
        self.masked_input_daughters      = tfK.layers.Masking(mask_value=D_PAD, name = 'masked_input_daughter_momenta')(self.input_daughters)
        self.masked_input_mother_momenta = tfK.layers.Masking(mask_value=M_PAD, name = 'masked_input_mother_momenta')(self.input_mother_momenta)
        self.masked_input_branch_z       = tfK.layers.Masking(mask_value=B_PAD, name = 'masked_input_branchings_z')(self.input_branch_z)
        self.masked_input_branch_t       = tfK.layers.Masking(mask_value=B_PAD, name = 'masked_input_branchings_t')(self.input_branch_t)
        self.masked_input_branch_d       = tfK.layers.Masking(mask_value=B_PAD, name = 'masked_input_branchings_d')(self.input_branch_d)
    
    def get_RNN(self):
        # Choose RNN cell
        if self.RNN_type == 'LSTM':
            self.RNN_cell = tfK.layers.LSTM
            if self.verbose:
                print("Using LSTM Network")
        elif self.RNN_type == 'GRU':
            self.RNN_cell = tfK.layers.GRU
            if self.verbose:
                print("Using GRU Network")
        else:
            self.RNN_cell = tfK.layers.SimpleRNN
            if self.verbose:
                print("Using SimpleRNN Network")
        
        # Initialize RNN from seed momentum
        h_init_layer = tfK.layers.Dense(self.dim_RNN, activation = 'tanh', name = 'h_init')
        h_init = h_init_layer(self.input_seed_momenta)
        rnn_0  = h_init_layer(self.masked_input_seed_momenta)
        if self.RNN_type == 'LSTM':
            c_init = tfK.layers.Dense(self.dim_RNN, activation = 'tanh', name = 'c_init')(self.input_seed_momenta)
            initial_state = [h_init, c_init]
        else:
            initial_state = [h_init]

        # RNN for t>0
        rnn_t = self.RNN_cell(self.dim_RNN, name = 'RNN', activation = 'tanh', return_sequences = True, bias_initializer = 'glorot_normal')(self.masked_input_daughters, initial_state = initial_state)
        
        RNN = tfK.layers.concatenate([rnn_0, rnn_t], axis=1, name = 'concatenate_RNN')
        return RNN
    
    def build_end(self):
        self.end_hidden = tfK.layers.Dense(self.dim_end_hid, name = 'end_hidden_layer', activation = 'relu')(self.RNN)
        self.end_output = tfK.layers.Dense(1,                name = 'endings', activation = 'sigmoid')(self.end_hidden)
    
    def build_mother(self):
        self.mother_hidden = tfK.layers.Dense(self.dim_mother_hid, name = 'mother_hidden_layer', activation = 'relu')(self.RNN)
        self.mother_unweighted_output = tfK.layers.Dense(self.dim_mother_out, name = 'mother_unweighted_output', activation = 'softmax')(self.mother_hidden)
        self.mother_weighted_output = tfK.layers.multiply([self.input_mother_weights, self.mother_unweighted_output], name = 'multiply_weights')
        self.mother_output = tfK.layers.Activation(self.normalize_layer, name = 'mothers')(self.mother_weighted_output)
        
    def build_branch_X(self, concat_list, branch_label):
        branch_X_input  = tfK.layers.concatenate(concat_list, axis=-1 , name = 'concatinate_branch_'+branch_label+'_inputs')
        branch_X_hidden = tfK.layers.Dense(self.dim_branch_hid, name = 'branch_'+branch_label+'_hidden_layer', activation = 'relu')(branch_X_input)
        return tfK.layers.Dense(self.dim_branch_out, name = 'sparse_branchings_' +branch_label, activation = 'softmax')(branch_X_hidden)
    
    def model(self):
        """ Build model of JUNIPR """
        
        # Inputs to JUNIPR
        self.input_seed_momenta   = tfK.Input((self.dim_mom, ),            name = 'seed_momentum')
        self.input_daughters      = tfK.Input((None, self.dim_daughters),  name = 'daughter_momenta')
        self.input_mother_momenta = tfK.Input((None, self.dim_mom),        name = 'mother_momenta')
        self.input_mother_weights = tfK.Input((None, self.dim_mother_out), name = 'mother_weights')
        self.input_branch_z       = tfK.Input((None, 1),                   name = 'branchings_z')
        self.input_branch_t       = tfK.Input((None, 1),                   name = 'branchings_t')
        self.input_branch_d       = tfK.Input((None, 1),                   name = 'branchings_d')
        
        _inputs = [self.input_seed_momenta, self.input_daughters, self.input_mother_momenta, self.input_mother_weights, self.input_branch_z, self.input_branch_t, self.input_branch_d]
        
        # Get copy of seed_momenta with same dimensions as RNN input
        self.input_seed_momenta_exp = tfK.layers.Lambda(lambda x: tfK.backend.expand_dims(x, axis = 1), name = 'input_seed_momenta_expand_dims')(self.input_seed_momenta)
        
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
        
        _outputs = [self.end_output, self.mother_output, self.branch_z_output, self.branch_t_output, self.branch_p_output, self.branch_d_output]
        
        return tfK.models.Model(inputs  = _inputs, 
                                outputs = _outputs, 
                                name = 'JUNIPR' + self.label)

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
        
    def get_log_probability_model(self, sum_log_probabilities=True):
        # Get all inputs from JUNIPR
        _junipr_inputs = self.model.inputs
        
        # Add all other inputs needed
        input_endings             = tf.keras.Input((None, 1),     name = 'input_endings', dtype=tf.float32)
        input_ending_weights      = tf.keras.Input((None, 1),     name = 'input_ending_weights', dtype=tf.float32)
        input_mothers             = tf.keras.Input((None, DIM_M), name = 'input_mothers')
        input_sparse_branchings_z = tf.keras.Input((None, 1),     name = 'input_sparse_branchings_z', dtype=tf.int64)
        input_sparse_branchings_t = tf.keras.Input((None, 1),     name = 'input_sparse_branchings_t', dtype=tf.int64)
        input_sparse_branchings_p = tf.keras.Input((None, 1),     name = 'input_sparse_branchings_p', dtype=tf.int64)
        input_sparse_branchings_d = tf.keras.Input((None, 1),     name = 'input_sparse_branchings_d', dtype=tf.int64)
        input_branchings_weights  = tf.keras.Input((None, 1),     name = 'input_sparse_branchings_weights')
        
        _probability_inputs = [input_endings, 
                              input_ending_weights, 
                              input_mothers,
                              input_sparse_branchings_z,
                              input_sparse_branchings_t,
                              input_sparse_branchings_p,
                              input_sparse_branchings_d,
                              input_branchings_weights
                             ]
        # Collect all inputs
        _all_inputs = _junipr_inputs + _probability_inputs
        
        # Get output layers
        endings             = self.model.get_layer('endings').output
        mothers             = self.model.get_layer('mothers').output
        sparse_branchings_z = self.model.get_layer('sparse_branchings_z').output
        sparse_branchings_t = self.model.get_layer('sparse_branchings_t').output
        sparse_branchings_p = self.model.get_layer('sparse_branchings_p').output
        sparse_branchings_d = self.model.get_layer('sparse_branchings_d').output
        
        # Constrict log_probabilities
        log_P_end    = tf.keras.layers.Lambda(lambda x: -self.binary_crossentropy_end(x[0],x[1],x[2]), name='log_P_end')([input_endings, endings, input_ending_weights])
        log_P_mother = tf.keras.layers.Lambda(lambda x:-self.categorical_crossentropy_mother(x[0],x[1]), name='log_P_mother')([input_mothers, mothers])
        log_P_z      = tf.keras.layers.Lambda(lambda x:-self.sparse_categorical_crossentropy(x[0],x[1],x[2]), name='log_P_z')([input_sparse_branchings_z, sparse_branchings_z, input_branchings_weights])
        log_P_t      = tf.keras.layers.Lambda(lambda x:-self.sparse_categorical_crossentropy(x[0],x[1],x[2]), name='log_P_t')([input_sparse_branchings_t, sparse_branchings_t, input_branchings_weights])
        log_P_p      = tf.keras.layers.Lambda(lambda x:-self.sparse_categorical_crossentropy(x[0],x[1],x[2]), name='log_P_p')([input_sparse_branchings_p, sparse_branchings_p, input_branchings_weights])
        log_P_d      = tf.keras.layers.Lambda(lambda x:-self.sparse_categorical_crossentropy(x[0],x[1],x[2]), name='log_P_d')([input_sparse_branchings_d, sparse_branchings_d, input_branchings_weights])
        
        if sum_log_probabilities:
            log_P_end    = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=-1, keepdims=True), name='sum_log_P_end')(log_P_end)
            log_P_mother = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=-1, keepdims=True), name='sum_log_P_mother')(log_P_mother)
            log_P_z      = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=-1, keepdims=True), name='sum_log_P_z')(log_P_z)
            log_P_t      = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=-1, keepdims=True), name='sum_log_P_t')(log_P_t)
            log_P_p      = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=-1, keepdims=True), name='sum_log_P_p')(log_P_p)
            log_P_d      = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=-1, keepdims=True), name='sum_log_P_d')(log_P_d)
            
            log_P = tf.keras.layers.Lambda(lambda x:x[0]+x[1]+x[2]+x[3]+x[4]+x[5], name='sum_log_P')([log_P_end,log_P_mother,log_P_z,log_P_t,log_P_p,log_P_d])
            # construct model
            probability_model = tf.keras.models.Model(inputs = _all_inputs, 
                                                      outputs=[log_P],
                                                      name = 'log_probability' + self.label)
        else:
            # construct model
            probability_model = tf.keras.models.Model(inputs = _all_inputs, 
                                                      outputs=[log_P_end, log_P_mother, log_P_z, log_P_t, log_P_p, log_P_d],
                                                      name = 'log_probability' + self.label)
        
        return probability_model        
    
    
    def validate(self, dataset, predict = False, label=None, model = None, reload = False, save_dir = './data'):
        """
        Validate dataset. 
        The dataset should have padding set to TFR_PADDED_SHAPES_MAX and not be repeted or shuffled. 
        """        
        if label is not None:
            label = '_'+label

        if predict and model is None:
            model = self.model

        if predict:
            save_file = save_dir + '/JUNIPR_validation_data' + label +'.npz'
        else:
            save_file = save_dir + '/Pythia_validation_data' + label +'.npz'

        if not reload and os.path.exists(save_file):
            print("Loading data from " + save_file)
            return np.load(save_file)['val_data'][0]
        else:
            if not tf.executing_eagerly():
                print("Validation must run in eager mode.")
                print("Please restart tensorflow and run tf.enable_eager_execution() before running validation")
                return

            # Create output arrays
            endings           = np.zeros((DIM_M, 1))
            ending_counts     = np.zeros((DIM_M, 1))

            mothers           = np.zeros((DIM_M, DIM_M))
            mother_counts     = np.zeros((DIM_M, DIM_M))

            branchings_z      = np.zeros((DIM_M, GRANULARITY))
            branchings_t      = np.zeros((DIM_M, GRANULARITY))
            branchings_p      = np.zeros((DIM_M, GRANULARITY))
            branchings_d      = np.zeros((DIM_M, GRANULARITY))
            branchings_counts = np.zeros((DIM_M, 1))


            for inputs, outputs in dataset:
                if predict:
                    e, m, b_z, b_t, b_p, b_d = model.predict_on_batch(inputs)
                    e   = e*tf.cast(outputs['ending_weights'], tf.float32)
                    m   = m*tf.cast(inputs['mother_weights'], tf.float32)
                    b_z = b_z[...,:-1]*tf.cast(outputs['sparse_branching_weights'], tf.float32)
                    b_t = b_t[...,:-1]*tf.cast(outputs['sparse_branching_weights'], tf.float32)
                    b_p = b_p[...,:-1]*tf.cast(outputs['sparse_branching_weights'], tf.float32)
                    b_d = b_d[...,:-1]*tf.cast(outputs['sparse_branching_weights'], tf.float32)
                else:
                    e   = outputs['endings']
                    m   = outputs['mothers']
                    b_z = tf.one_hot(outputs['sparse_branchings_z'], depth=GRANULARITY)
                    b_t = tf.one_hot(outputs['sparse_branchings_t'], depth=GRANULARITY)
                    b_p = tf.one_hot(outputs['sparse_branchings_p'], depth=GRANULARITY)
                    b_d = tf.one_hot(outputs['sparse_branchings_d'], depth=GRANULARITY)


                endings           += np.sum(e                        , axis=0)
                ending_counts     += np.sum(outputs['ending_weights'], axis=0)

                mothers           += np.sum(m                        , axis=0)
                mother_counts     += np.sum(inputs['mother_weights'] , axis=0)

                branchings_z      += np.sum(b_z, axis=(0,1))
                branchings_t      += np.sum(b_t, axis=(0,1))
                branchings_p      += np.sum(b_p, axis=(0,1))
                branchings_d      += np.sum(b_d, axis=(0,1))
                branchings_counts += np.sum(outputs['sparse_branching_weights'], axis=0)

            val_data = {'endings':endings,
                        'ending_counts':ending_counts,
                        'mothers':mothers,
                        'mother_counts':mother_counts,
                        'branchings_z':branchings_z,
                        'branchings_t':branchings_t,
                        'branchings_p':branchings_p,
                        'branchings_d':branchings_d,
                        'branchings_counts':branchings_counts}
            np.savez(save_file, val_data=[val_data])

            return val_data

    def load_model(self, path_to_saved_model):
        self.model = tfK.models.load_model(path_to_saved_model, binary_crossentropy_end_objects=self.custom_objects)

"""
def get_bin_index(value, bin_edges):
    for i, edge in enumerate(np.asarray(bin_edges)[1:]):
        if(value < edge):
            return i
    return len(bin_edges)-2


"""
