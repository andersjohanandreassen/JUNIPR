"""Utilities for reading TFRecords for JUNIPR."""

import tensorflow as tf
from junipr.config import *
import numpy as np

__all__ = ['parse_fn', 'get_TFR_PADDED_VALUES', 'get_TFR_PADDED_SHAPES', 'get_dataset']

def deserialize_sparse(serialized_sparse_tensor, dtype):
    # tf.io.deserialize_many_sparse() requires the dimensions to be [N,3] so we add one dimension with expand_dims
    sst_exp_dim = tf.expand_dims(serialized_sparse_tensor, axis=0)
    # deserialize sparse tensor
    deserialized_sparse_tensor = tf.io.deserialize_many_sparse(sst_exp_dim, dtype=dtype)
    # convert from sparse to dense
    dense_tensor = tf.sparse.to_dense(deserialized_sparse_tensor)
    # remove extra dimenson [1, 3] -> [3]
    return tf.squeeze(dense_tensor, axis=0)


def parse_fn(data_record, label=''):
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
        
        #'CSJets':       tf.VarLenFeature(tf.float32),
        #'CS_ID_intermediate_states':       tf.VarLenFeature(tf.int64),
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
  
    #sequence_data['CSJets']       = tf.sparse.to_dense(sequence_data['CSJets'], default_value=-1)
    #sequence_data['CS_ID_intermediate_states']       = tf.sparse.to_dense(sequence_data['CS_ID_intermediate_states'] , default_value=-1)
    #sequence_data['CS_ID_daughters']       = tf.sparse.to_dense(sequence_data['CS_ID_daughters'] , default_value=-1)

    # Construct mother weights
    mother_weights = tf.cast(tf.linalg.LinearOperatorLowerTriangular(tf.ones((context_data['n_branchings'][0]+1, DIM_M))).to_dense(), tf.float32)
    
    return {'label'                     : context_data['label'],
            'seed_momentum'    + label  : context_data['seed_momentum'], 
            'daughter_momenta' + label  : sequence_data['daughter_momenta'],
            'mother_momenta'   + label  : sequence_data['mother_momenta'],
            'mother_weights'   + label  : mother_weights,
            'branchings_z'     + label  : sequence_data['branchings'][:,0:1],
            'branchings_t'     + label  : sequence_data['branchings'][:,1:2],
            'branchings_p'     + label  : sequence_data['branchings'][:,2:3],
            'branchings_d'     + label  : sequence_data['branchings'][:,3:],
           },{
            'endings'                 + label : context_data['endings'],
            'ending_weights'          + label : context_data['ending_weights'],
            'sparse_branching_weights'+ label : context_data['sparse_branching_weights'],
            'mothers'                 + label : context_data['mothers'],
            'sparse_branchings_z'     + label : sequence_data['sparse_branchings'][:,0:1],
            'sparse_branchings_t'     + label : sequence_data['sparse_branchings'][:,1:2],
            'sparse_branchings_p'     + label : sequence_data['sparse_branchings'][:,2:3],
            'sparse_branchings_d'     + label : sequence_data['sparse_branchings'][:,3:],
            }

def get_TFR_PADDED_SHAPES(label='', max_padding = False):
    if max_padding:
        dim = DIM_M
        dim_d = DIM_M-1
    else:
        dim = None
        dim_d = None
    
    return ({'label'                  : [1],
             'seed_momentum'  + label : [4],
             'daughter_momenta'+label : [dim_d, 8],
             'mother_momenta' + label : [dim, 4],
             'mother_weights' + label : [dim, DIM_M],
             'branchings_z'   + label : [dim,1],
             'branchings_t'   + label : [dim,1],
             'branchings_p'   + label : [dim,1],
             'branchings_d'   + label : [dim,1],
             },
            {'endings'             + label : [dim, 1],
             'ending_weights'      + label : [dim, 1],
             'mothers'             + label : [dim, DIM_M],
             'sparse_branchings_z' + label : [dim,1],
             'sparse_branchings_t' + label : [dim,1],
             'sparse_branchings_p' + label : [dim,1],
             'sparse_branchings_d' + label : [dim,1],
             'sparse_branching_weights'+ label: [dim, 1],
             })

def get_TFR_PADDED_VALUES(label=''):
    return ({'label'                  : np.int64(-1),
             'seed_momentum'   + label: np.float32(D_PAD),
             'daughter_momenta'+ label: np.float32(D_PAD),
             'mother_momenta'  + label: np.float32(M_PAD),
             'mother_weights'  + label: np.float32(0),
             'branchings_z'    + label: np.float32(B_PAD),
             'branchings_t'    + label: np.float32(B_PAD),
             'branchings_p'    + label: np.float32(B_PAD),
             'branchings_d'    + label: np.float32(B_PAD),
            },
            {'endings'             + label: np.int64(0),
             'ending_weights'      + label: np.int64(0),
             'mothers'             + label: np.int64(0),
             'sparse_branchings_z' + label: np.int64(10),
             'sparse_branchings_t' + label: np.int64(10),
             'sparse_branchings_p' + label: np.int64(10),
             'sparse_branchings_d' + label: np.int64(10),
             'sparse_branching_weights'+ label: np.int64(0),
             })

def get_dataset(tfrecord_filename, batch_size=None, label='', max_padding=False):
    dataset = tf.data.TFRecordDataset([tfrecord_filename])
    dataset = dataset.map(parse_fn)
    if batch_size is not None:
        dataset = dataset.padded_batch(batch_size,
                                       padded_shapes  = get_TFR_PADDED_SHAPES(label, max_padding),
                                       padding_values = get_TFR_PADDED_VALUES(label))
    return dataset
