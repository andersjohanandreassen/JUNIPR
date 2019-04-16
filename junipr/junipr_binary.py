"""Implementation of JUNIPR Binary. """

__all__ = ['JUNIPR_binary']

import tensorflow as tf 
tfK = tf.keras

import numpy as np
import os
from junipr.config import *
from .junipr import *

__all__ = ['JUNIPR_binary']


class JUNIPR_binary:
    """ tf.keras implementation of JUNIPR binary from arXiv:19XX.XXXXX """
    
    def __init__(self, model_path_0=None, model_path_1=None):
        self.model_path_0 = model_path_0
        self.model_path_1 = model_path_1
        
        self.model = self.model()
        
        self.model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy')

    def model(self):
        # Load two JUNIPR models
        junipr0 = JUNIPR(label='0', verbose=False)
        junipr1 = JUNIPR(label='1', verbose=False)
        
        # Load weights from model path if 
        if self.model_path_0 is not None:
            print("Loading model 0 from", self.model_path_0, flush=True)
            junipr0.load_model(self.model_path_0)
        if self.model_path_1 is not None:
            print("Loading model 1 from", self.model_path_1, flush=True)
            junipr1.load_model(self.model_path_1)
        
        pjunipr0 = junipr0.get_log_probability_model(True)
        pjunipr1 = junipr1.get_log_probability_model(True)
        
        input_seed_momenta        = tf.keras.Input((4, ),         name = 'seed_momentum'    )
        input_daughters           = tf.keras.Input((None, 8),     name = 'daughter_momenta' )
        input_mother_momenta      = tf.keras.Input((None, 4),     name = 'mother_momenta'   )
        input_mother_weights      = tf.keras.Input((None, DIM_M), name = 'mother_weights'   )
        input_branch_z            = tf.keras.Input((None, 1),     name = 'branchings_z'     )
        input_branch_t            = tf.keras.Input((None, 1),     name = 'branchings_t'     )
        input_branch_d            = tf.keras.Input((None, 1),     name = 'branchings_d'     )
        input_endings             = tf.keras.Input((None, 1),     name = 'input_endings')
        input_ending_weights      = tf.keras.Input((None, 1),     name = 'input_ending_weights')
        input_mothers             = tf.keras.Input((None, DIM_M), name = 'input_mothers')
        input_sparse_branchings_z = tf.keras.Input((None, 1),     name = 'input_sparse_branchings_z')
        input_sparse_branchings_t = tf.keras.Input((None, 1),     name = 'input_sparse_branchings_t', dtype=tf.int64)
        input_sparse_branchings_p = tf.keras.Input((None, 1),     name = 'input_sparse_branchings_p', dtype=tf.int64)
        input_sparse_branchings_d = tf.keras.Input((None, 1),     name = 'input_sparse_branchings_d', dtype=tf.int64)
        input_branchings_weights  = tf.keras.Input((None, 1),     name = 'input_sparse_branchings_weights')
        _all_inputs=[input_seed_momenta, input_daughters, input_mother_momenta, input_mother_weights, input_branch_z,
                     input_branch_t, input_branch_d, input_endings, input_ending_weights, input_mothers, input_sparse_branchings_z, 
                     input_sparse_branchings_t, input_sparse_branchings_p, input_sparse_branchings_d, input_branchings_weights]
        
        p0 = pjunipr0(inputs=_all_inputs)
        p1 = pjunipr1(inputs=_all_inputs)
        
        output_concat = tf.keras.layers.concatenate([p0,p1],  name='concat_p0_p1')
        output_layer  = tf.keras.layers.Activation('softmax', name='label')(output_concat)
        
        return tf.keras.models.Model(inputs=_all_inputs, outputs=output_layer)