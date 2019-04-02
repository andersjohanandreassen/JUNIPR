import tensorflow as tf

### Helper functions for creating tf.train.Feature of different types

__all__ = ['_single_float_feature', '_single_int64_feature', '_single_byte_string_feature', 
           '_list_float_feature', '_list_int64_feature', '_list_byte_string_feature',
           '_list_of_lists_float_feature', '_list_of_lists_int64_feature', '_list_of_lists_byte_string_feature']

# Scalars
def _single_float_feature(value):
    """ Creates a feature object from a single float"""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
def _single_int64_feature(value):
    """ Creates a feature object from a single Int64"""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _single_byte_string_feature(value):
    """ Creates a feature object from a single byte string"""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# Lists
def _list_float_feature(lst):
    """ Creates a feature object from a list of floats"""
    return tf.train.Feature(float_list=tf.train.FloatList(value=lst))
def _list_int64_feature(lst):
    """ Creates a feature object from a list of Int64s"""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=lst))
def _list_byte_string_feature(lst):
    """ Creates a feature object from a list of byte strings"""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=lst))

# List of lists
def _list_of_lists_float_feature(list_of_lists):
    """ Creates a FeatureList object from a list of lists of floats"""
    list_of_features = [_list_float_feature(lst) for lst in list_of_lists]
    return tf.train.FeatureList(feature=list_of_features)
def _list_of_lists_int64_feature(list_of_lists):
    """ Creates a FeatureList object from a list of lists of Int64s"""
    list_of_features = [_list_int64_feature(lst) for lst in list_of_lists]
    return tf.train.FeatureList(feature=list_of_features)
def _list_of_lists_byte_string_feature(list_of_lists):
    """ Creates a FeatureList object from a list of lists of byte strings"""
    list_of_features = [_list_byte_string_feature(lst) for lst in list_of_lists]
    return tf.train.FeatureList(feature=list_of_features)

