"""Utilities for writing JuniprJets to a TFRecord."""

import tensorflow as tf
import numpy as np
import json
from .generic_utils import *
from junipr.config import *
from junipr.utils.feature_scaling import *
from junipr.utils.printing_utils import print_progress

## NEEDS MORE COMMENTS

__all__ = ['create_TFRecord']

## JUNIPR features

def label_feature(label):
    """ Create feature for label"""
    return _single_int64_feature(label)

def n_branchings_feature(n_branchings):
    """ Create feature for n_branchings"""
    return _single_int64_feature(n_branchings)

def seed_momentum_feature(seed_momentum):
    """ Create feature for seed momentum with feature scaling"""
    return _list_float_feature(shift_mom(seed_momentum))

def endings_feature(n_branchings):
    """ Create feature for endings"""
    return _list_int64_feature(tf.one_hot(n_branchings, depth=n_branchings+1))

def ending_weights_feature(n_branchings):
    """ Create feature for ending_weights"""
    return _list_int64_feature(tf.fill((n_branchings+1,1),1))

def daughter_momenta_feature(daugther_momenta):
    """ Create feature for daughter momenta"""
    return  _list_of_lists_float_feature([np.concatenate([shift_mom(d[0]), shift_mom(d[1])]) for d in  daugther_momenta]) 

def mother_momenta_feature(mother_momenta):
    """ Create feature for mother momenta"""
    return _list_of_lists_float_feature([shift_mom(mother_momentum) for mother_momentum in mother_momenta] + [[D_PAD]*4])

def mothers_feature(mothers_id_energy_order):
    """ Create feature for mothers (mother ids in energy ordering) """
    # Create sparse tensor
    indices     = [[i, val] for i, val in enumerate(mothers_id_energy_order)]
    values      = np.ones_like(mothers_id_energy_order)
    dense_shape = [len(mothers_id_energy_order)+1, max(mothers_id_energy_order)+1]
    
    sparse_tensor = tf.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)
    serialized_sparse_tensor = tf.io.serialize_sparse(sparse_tensor).numpy()
                                
    return _list_byte_string_feature(serialized_sparse_tensor)

def branchings_feature(branchings):
    """ Create feature for branchings """
    return _list_of_lists_float_feature([shift_branch(branching) for branching in branchings] + [[B_PAD]*4])

def sparse_branchings_feature(branchings):
    """ Create feature for sparse_branchings """
    shifted_branchings = np.asarray([shift_branch(branching) for branching in branchings])
    return _list_of_lists_int64_feature(list(np.clip(shifted_branchings*GRANULARITY, 0, GRANULARITY-1).astype(np.int32)) + [[GRANULARITY]*4])

def sparse_branching_weights_feature(n_branchings):
    """ Create feature for sparse_branching_weights"""
    return _list_int64_feature(tf.fill((n_branchings,1),1))

def CSJets_feature(CSJets):
    """ Create feature for CSJets"""
    return _list_of_lists_float_feature([shift_mom(momentum) for momentum in CSJets])

def CS_ID_intermediate_states_feature(CS_ID_intermediate_states):
    """ Create feature for CS_ID_intermediate_states"""
    return _list_of_lists_int64_feature(CS_ID_intermediate_states)


def get_sequence_example_object(data_element_dict):
    """ Creates a SequenceExample object from a dictionary for a single data element 
    data_element_dict is a dictionary for each element in .json file created by the fastjet code. 
    """
    # Context contains all scalar and list features
    context = tf.train.Features(
            feature=
            {
                'n_branchings'   : n_branchings_feature(  data_element_dict['n_branchings']),
                'seed_momentum'  : seed_momentum_feature( data_element_dict['seed_momentum']),
                'endings'        : endings_feature(       data_element_dict['n_branchings']),
                'ending_weights' : ending_weights_feature(data_element_dict['n_branchings']),
                'mothers'        : mothers_feature(       data_element_dict['mothers_id_energy_order']),
                'sparse_branching_weights' : sparse_branching_weights_feature(data_element_dict['n_branchings']),
                'label'   : label_feature(  data_element_dict['label']),
                
                #'CS_ID_mothers':       _list_int64_feature(data_element_dict['CS_ID_mothers']),
            }
    )
    
    # Feature_lists contains all lists of lists
    feature_lists = tf.train.FeatureLists(
            feature_list=
            {
                'branchings'       : branchings_feature(       data_element_dict['branchings']),
                'sparse_branchings': sparse_branchings_feature(data_element_dict['branchings']),     
                'mother_momenta'   : mother_momenta_feature(   data_element_dict['mother_momenta']),
                'daughter_momenta' : daughter_momenta_feature( data_element_dict['daughter_momenta']),
                
                #'CSJets'           : CSJets_feature(data_element_dict['CSJets']),
                #'CS_ID_intermediate_states': CS_ID_intermediate_states_feature(data_element_dict['CS_ID_intermediate_states']),
                #'CS_ID_daughters': _list_of_lists_int64_feature(data_element_dict['CS_ID_daughters']),
            }
    )
                
    sequence_example = tf.train.SequenceExample(context       = context,
                                                feature_lists = feature_lists)
    
    return sequence_example

def create_TFRecord(json_filename, tfrecord_filename, verbose = False):
    """ 
    Create TFRecord from json file. 
    The json file contains a list of JuniprJets. 
    Each JuniprJet is a dictionary containing the data for that single data element. 
    get_sequence_example_object is a function that takes one data_element_dict and returns a sequence_example object
    """
    
    # Load data array from json file
    with open(json_filename) as json_file:
        data_arr = json.load(json_file)['JuniprJets']
    
    # Write to TFRecord
    with tf.io.TFRecordWriter(tfrecord_filename) as tfwriter:
        # Iterate through all data elements in data array
        data_size = len(data_arr)
        if verbose:
            print("Writing " + json_filename + " to " + tfrecord_filename, flush=True)
        for i, data_element_dict in enumerate(data_arr):
            if verbose:
                print_progress(i, data_size)
                
            sequence_example = get_sequence_example_object(data_element_dict)

            # Append each example into tfrecord
            tfwriter.write(sequence_example.SerializeToString()) 