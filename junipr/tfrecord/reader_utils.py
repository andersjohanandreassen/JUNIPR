"""Utilities for reading TFRecords for JUNIPR."""

import tensorflow as tf
from junipr.config import *
import numpy as np

__all__ = ['parse_fn', 'get_TFR_PADDED_VALUES', 'get_TFR_PADDED_SHAPES', 'get_dataset','to_probability_inputs']

def deserialize_sparse(serialized_sparse_tensor, dtype):
    # tf.io.deserialize_many_sparse() requires the dimensions to be [N,3] so we add one dimension with expand_dims
    sst_exp_dim = tf.expand_dims(serialized_sparse_tensor, axis=0)
    # deserialize sparse tensor
    deserialized_sparse_tensor = tf.io.deserialize_many_sparse(sst_exp_dim, dtype=dtype)
    # convert from sparse to dense
    dense_tensor = tf.sparse.to_dense(deserialized_sparse_tensor)
    # remove extra dimenson [1, 3] -> [3]
    return tf.squeeze(dense_tensor, axis=0)


def parse_fn(data_record):
    """ Function to convert from TFRecord back to numbers, lists and arrays."""
    
    # Single values and lists are stored in context_features
    context_features = {
        'label'         : tf.io.FixedLenFeature([1], tf.int64),
        'n_branchings'  : tf.io.FixedLenFeature([1], tf.int64),
        'seed_momentum' : tf.io.FixedLenFeature([4], tf.float32),
        'endings'       : tf.io.VarLenFeature(tf.int64),
        'ending_weights': tf.io.VarLenFeature(tf.int64),
        'sparse_branching_weights': tf.io.VarLenFeature(tf.int64),
        'mothers'       : tf.io.FixedLenFeature([3], tf.string),
        
        #'CS_ID_mothers':    tf.VarLenFeature(tf.int64),
    }
    
    # List of lists is stored in sequence_features
    sequence_features = {
        'branchings'       : tf.io.VarLenFeature(tf.float32),
        'sparse_branchings': tf.io.VarLenFeature(tf.int64),
        'mother_momenta'   : tf.io.VarLenFeature(tf.float32),
        'daughter_momenta' : tf.io.VarLenFeature(tf.float32),
        
        'CSJets':       tf.io.VarLenFeature(tf.float32),
        'CS_ID_intermediate_states':       tf.io.VarLenFeature(tf.int64),
        #'CS_ID_daughters':       tf.VarLenFeature(tf.int64),
    }
    
    # Get single example
    context_data, sequence_data = tf.io.parse_single_sequence_example(data_record, 
                                                                       context_features = context_features, 
                                                                       sequence_features = sequence_features)
    
    ### All VarLenFeatures are stored as sparse tensors
    ### Convert them to dense tensor
   
    # context_features
    context_data['endings']        = tf.expand_dims(tf.sparse.to_dense(context_data['endings'], default_value=E_PAD), axis=-1)
    context_data['ending_weights'] = tf.expand_dims(tf.sparse.to_dense(context_data['ending_weights'], default_value=E_PAD), axis=-1)
    context_data['sparse_branching_weights'] = tf.expand_dims(tf.sparse.to_dense(context_data['sparse_branching_weights'], default_value=E_PAD), axis=-1)
    context_data['mothers']        = deserialize_sparse(context_data['mothers'], tf.int64)
    
    #context_data['CS_ID_mothers']        = tf.sparse.to_dense(context_data['CS_ID_mothers'], default_value=-1)
 
    #sequence_features
    sequence_data['branchings']        = tf.sparse.to_dense(sequence_data['branchings'],        default_value=B_PAD)
    sequence_data['sparse_branchings'] = tf.sparse.to_dense(sequence_data['sparse_branchings'], default_value=SB_PAD)
    sequence_data['mother_momenta']    = tf.sparse.to_dense(sequence_data['mother_momenta'],    default_value=M_PAD)
    sequence_data['daughter_momenta']  = tf.sparse.to_dense(sequence_data['daughter_momenta'],  default_value=D_PAD)
  
    sequence_data['CSJets']                          = tf.sparse.to_dense(sequence_data['CSJets'], default_value=-1)
    sequence_data['CS_ID_intermediate_states']       = tf.sparse.to_dense(sequence_data['CS_ID_intermediate_states'] , default_value=-1)
    #sequence_data['CS_ID_daughters']       = tf.sparse.to_dense(sequence_data['CS_ID_daughters'] , default_value=-1)

    # Construct mother weights
    mother_weights = tf.cast(tf.linalg.LinearOperatorLowerTriangular(tf.ones((context_data['n_branchings'][0]+1, DIM_M))).to_dense(), tf.float32)
    
    return {'label'                     : context_data['label'],
            'seed_momentum'             : context_data['seed_momentum'], 
            'daughter_momenta'          : sequence_data['daughter_momenta'],
            'mother_momenta'            : sequence_data['mother_momenta'],
            'mother_weights'            : mother_weights,
            'branchings_z'              : sequence_data['branchings'][:,0:1],
            'branchings_t'              : sequence_data['branchings'][:,1:2],
            'branchings_p'              : sequence_data['branchings'][:,2:3],
            'branchings_d'              : sequence_data['branchings'][:,3:],
            'CSJets'                    : sequence_data['CSJets'],
            'CS_ID_intermediate_states' : sequence_data['CS_ID_intermediate_states'],
           },{
            'endings'                   : context_data['endings'],
            'ending_weights'            : context_data['ending_weights'],
            'sparse_branching_weights'  : context_data['sparse_branching_weights'],
            'mothers'                   : context_data['mothers'],
            'sparse_branchings_z'       : sequence_data['sparse_branchings'][:,0:1],
            'sparse_branchings_t'       : sequence_data['sparse_branchings'][:,1:2],
            'sparse_branchings_p'       : sequence_data['sparse_branchings'][:,2:3],
            'sparse_branchings_d'       : sequence_data['sparse_branchings'][:,3:],
            }

def to_probability_inputs(inputs, outputs):
    
    return {**inputs, 
            'input_endings'                  : tf.cast(outputs['endings'],                  dtype=tf.float32),
            'input_ending_weights'           : tf.cast(outputs['ending_weights'],           dtype=tf.float32),
            'input_mothers'                  : tf.cast(outputs['mothers'],                  dtype=tf.float32),
            'input_sparse_branchings_z'      : tf.cast(outputs['sparse_branchings_z'],      dtype=tf.float32),
            'input_sparse_branchings_t'      : tf.cast(outputs['sparse_branchings_t'],      dtype=tf.float32),
            'input_sparse_branchings_p'      : tf.cast(outputs['sparse_branchings_p'],      dtype=tf.float32),
            'input_sparse_branchings_d'      : tf.cast(outputs['sparse_branchings_d'],      dtype=tf.float32),
            'input_sparse_branchings_weights': tf.cast(outputs['sparse_branching_weights'], dtype=tf.float32),            
           },{
            'label'                          : tf.cast(inputs['label'],                     dtype=tf.float32)
            }

def get_TFR_PADDED_SHAPES(max_padding = False):
    if max_padding:
        dim = DIM_M
        dim_d = DIM_M-1
    else:
        dim = None
        dim_d = None
    
    return ({'label'                     : [1],
             'seed_momentum'             : [4],
             'daughter_momenta'          : [dim_d, 8],
             'mother_momenta'            : [dim, 4],
             'mother_weights'            : [dim, DIM_M],
             'branchings_z'              : [dim,1],
             'branchings_t'              : [dim,1],
             'branchings_p'              : [dim,1],
             'branchings_d'              : [dim,1],
             'CSJets'                    : [2*DIM_M-1, 4],
             'CS_ID_intermediate_states' : [dim, dim], 
             },
            {'endings'                   : [dim, 1],
             'ending_weights'            : [dim, 1],
             'mothers'                   : [dim, DIM_M],
             'sparse_branchings_z'       : [dim,1],
             'sparse_branchings_t'       : [dim,1],
             'sparse_branchings_p'       : [dim,1],
             'sparse_branchings_d'       : [dim,1],
             'sparse_branching_weights'  : [dim, 1],
             })

def get_TFR_PADDED_VALUES():
    return ({'label'                     : np.int64(-1),
             'seed_momentum'             : np.float32(D_PAD),
             'daughter_momenta'          : np.float32(D_PAD),
             'mother_momenta'            : np.float32(M_PAD),
             'mother_weights'            : np.float32(0),
             'branchings_z'              : np.float32(B_PAD),
             'branchings_t'              : np.float32(B_PAD),
             'branchings_p'              : np.float32(B_PAD),
             'branchings_d'              : np.float32(B_PAD),
             'CSJets'                    : np.float32(D_PAD),
             'CS_ID_intermediate_states' : np.int64(-1), 
            },
            {'endings'                   : np.int64(0),
             'ending_weights'            : np.int64(0),
             'mothers'                   : np.int64(0),
             'sparse_branchings_z'       : np.int64(10),
             'sparse_branchings_t'       : np.int64(10),
             'sparse_branchings_p'       : np.int64(10),
             'sparse_branchings_d'       : np.int64(10),
             'sparse_branching_weights'  : np.int64(0),
             })

def get_dataset(tfrecord_files, 
                batch_size=None, 
                max_padding=False,
                repeat=True,
                shuffle=100,
                prefetch=1,
                probability_inputs=False,
                num_parallel_calls=None):
    """Get tf.data.Dataset for a tfrecord file or list of tfrecord files. 
    max_padding: Pad all inputs to max size (=DIM_M). This is used for the validation code
    repeat, shuffle, prefetch: settings for loading dataset
    probability_inputs: return dataset in format used for the probability model
    """
    
    if type(tfrecord_files)==str:
        dataset = tf.data.TFRecordDataset([tfrecord_files])
    elif type(tfrecord_files)==list:
        files = tf.data.Dataset.list_files(tfrecord_files, shuffle=False)
        dataset = files.interleave(tf.data.TFRecordDataset, cycle_length=len(tfrecord_files))
        
    dataset = dataset.map(parse_fn, num_parallel_calls=num_parallel_calls)
    
        
    if batch_size is not None:
        dataset = dataset.padded_batch(batch_size,
                                       padded_shapes  = get_TFR_PADDED_SHAPES(max_padding),
                                       padding_values = get_TFR_PADDED_VALUES())
    if probability_inputs:
        dataset = dataset.map(to_probability_inputs)
    if repeat:
        dataset = dataset.repeat()
    if shuffle>0:
        dataset = dataset.shuffle(buffer_size = shuffle)
    if prefetch>0:
        dataset = dataset.prefetch(prefetch)
    
    
    return dataset
