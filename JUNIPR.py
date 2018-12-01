import tensorflow as tf 
tfK = tf.keras

import numpy as np
from load_data import *
from random import shuffle


def print_progress(step_i, n_steps, n_print_outs=10):
    if step_i != 0 and step_i%(n_steps//n_print_outs) ==0:
        print("-- Step", step_i, "of", n_steps, flush=True)

class JUNIPR:
    """ tf.keras implementation of JUNIPR from arXiv:1804.09720 using a Simple RNN """
    
    def __init__(self, 
                 granularity=10, 
                 label = None, 
                 dim_mom = 4, 
                 dim_end_hid = 100, dim_mother_hid = 100, dim_mother_out = 100, dim_branch_hid = 100, dim_RNN = 100, 
                 normalize_by_length = False, 
                 optimizer = None, 
                 RNN_type = None,
                 tensorflow_warning_verbosity = False):
        
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
        self.granularity    = granularity
        self.dim_end_hid    = dim_end_hid
        self.dim_mother_hid = dim_mother_hid
        self.dim_mother_out = dim_mother_out # This also sets the maximum shower length
        self.max_time       = self.dim_mother_out
        self.dim_branch_hid = dim_branch_hid
        self.dim_branch_out = self.granularity**self.dim_branch + 1 # +1 is for padded value
        self.dim_RNN        = dim_RNN
        
        # Masking values
        self.d_mask = -1
        self.m_mask = -1
        self.b_mask = self.granularity**self.dim_branch
        
        # Learning rate 
        self.lr = 1e-2
        
        # Normalization
        self.normalize_by_length = normalize_by_length
        
        # Optimizer
        if optimizer is None:
            self.optimizer = 'SGD'
        else:
            self.optimizer = optimizer
            
        # RNN_type
        self.RNN_type = RNN_type
        
        ### Get model
        self.model = self.model()
        
        if tensorflow_warning_verbosity is False:
            # Remove tensorflow warnings that have no effect on performance. Should be removed in future. 
            tf.logging.set_verbosity(tf.logging.ERROR)

        
    def normalize_layer(self, x):
        """ Activation function that normalizes the output of a layer. 
        Assumes that all values are in the range [0,1] """
        total = tfK.backend.clip(tfK.backend.sum(x, axis=-1, keepdims=True), tfK.backend.epsilon(), 1)
        return x/total
    
    def binary_crossentropy_end(self, target, output):
        if self.normalize_by_length:
            # Default Keras behavior is to normalize loss by length of sequence
            return tfK.losses.binary_crossentropy(target, output)
        else:
            # Multiply by length of sequence to remove default normalization
            N_branchings = tf.cast(tfK.backend.expand_dims(tfK.backend.argmax(target, axis = 1), axis = -1), tf.float32) + 1
            batch_scaling = tfK.backend.mean(N_branchings)*N_branchings**(-1.) # Undo batch normalization where keras does a weighted average
            return tfK.backend.mean(batch_scaling*N_branchings*tfK.backend.binary_crossentropy(target, output), axis = -1)
        
    def categorical_crossentropy_mother(self, target, output):
        """ categorical_crossentropy where rows with just zeros are ignored """
        # Normalize output ignoring rows of just zeros
        sums = tfK.backend.sum(output, axis=-1, keepdims=True)
        sums = tfK.backend.clip(sums, tfK.backend.epsilon(), 1)
        output = output/sums
        # Explicitly calculate categorical_crossentropy
        output = tfK.backend.clip(output, tfK.backend.epsilon(), 1) # Clip to avoid nan from log(zero) from padding
        
        if self.normalize_by_length:
            # Default Keras behavior is to normalize loss by length of sequence
            return -tfK.backend.sum(target*tfK.backend.log(output), axis=-1)
        else:
            # Multiply by length of sequence to remove default normalization
            N_branchings = tfK.backend.sum(target, axis = (1,2), keepdims = True)+1 # length of each jet
            batch_scaling = tfK.backend.mean(N_branchings)*N_branchings**(-1.) # Undo batch normalization where keras does a weighted average
            return -tfK.backend.sum(N_branchings*batch_scaling*target*tfK.backend.log(output), axis=-1)
        
    def sparse_categorical_crossentropy_branch(self, target, output):
        if self.normalize_by_length:
            # Default Keras behavior is to normalize loss by length of sequence
            return tfK.losses.sparse_categorical_crossentropy(target, output)
        else:
            # Multiply by length of sequence to remove default normalization
            #N_branchings = tf.cast(tfK.backend.expand_dims(tfK.backend.argmax(target, axis = 1), axis = -1), tf.float32)
            N_branchings = tf.cast(tfK.backend.argmax(target, axis = 1), tf.float32)
            N_branchings = tfK.backend.maximum(N_branchings, tfK.backend.ones_like(N_branchings)) # tfK.backend.maximum neccesary to deal with jets of length 1 (zero splittings)
            batch_scaling = tfK.backend.mean(N_branchings)*N_branchings**(-1.) # Undo batch normalization where keras does a weighted average
            return batch_scaling*N_branchings*tfK.losses.sparse_categorical_crossentropy(target, output)
            
    def model(self):
        """ Build model of JUNIPR """
        
        # Inputs to JUNIPR
        input_seed_momenta     = tfK.Input((self.dim_mom, ),        name = 'input_seed_momenta' + self.label)
        input_daughters        = tfK.Input((None, self.dim_daughters),  name = 'input_daughters' + self.label)
        input_mother_momenta   = tfK.Input((None, self.dim_mom),        name = 'input_mother_momenta' + self.label)
        input_mother_weights   = tfK.Input((None, self.dim_mother_out), name = 'input_mother_weights' + self.label)
        
        # Get copy of seed_momenta with same dimensions as RNN input
        input_seed_momenta_exp = tfK.layers.Lambda(lambda x: tfK.backend.expand_dims(x, axis = 1), name = 'input_seed_momenta_expand_dims' + self.label)(input_seed_momenta)
        
        # Masking
        masked_input_seed_momenta   = tfK.layers.Masking(mask_value=self.d_mask, name = 'masked_input_seed_momenta' + self.label)(input_seed_momenta_exp)
        masked_input_daughters      = tfK.layers.Masking(mask_value=self.d_mask, name = 'masked_input_daughters' + self.label)(input_daughters)
        masked_input_mother_momenta = tfK.layers.Masking(mask_value=self.m_mask, name = 'masked_input_mother_momenta' + self.label)(input_mother_momenta)
        
        # RNN cell
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
        if self.RNN_type == 'LSTM':
            h_init_layer = tfK.layers.Dense(self.dim_RNN, activation = 'tanh', name = 'h_init' + self.label)
            h_init = h_init_layer(input_seed_momenta)
            c_init = tfK.layers.Dense(self.dim_RNN, activation = 'tanh', name = 'c_init' + self.label)(input_seed_momenta)
            initial_state = [h_init, c_init]
            rnn_0  = h_init_layer(masked_input_seed_momenta)
        else:
            h_init_layer = tfK.layers.Dense(self.dim_RNN, activation = 'tanh', name = 'h_init' + self.label)
            h_init = h_init_layer(input_seed_momenta)
            initial_state = [h_init]
            rnn_0  = h_init_layer(masked_input_seed_momenta)

        # RNN for t>0
        rnn_t = self.RNN_cell(self.dim_RNN, name = 'RNN' + self.label, activation = 'tanh', return_sequences = True, bias_initializer = 'glorot_normal')(masked_input_daughters, initial_state = initial_state)
        
        RNN = tfK.layers.concatenate([rnn_0, rnn_t], axis=1, name = 'concatenate_RNN' + self.label)
        
        # End
        end_hidden = tfK.layers.Dense(self.dim_end_hid, name = 'end_hidden_layer' + self.label, activation = 'relu')(RNN)
        end_output = tfK.layers.Dense(1,                name = 'end_output_layer' + self.label, activation = 'sigmoid')(end_hidden)
        
        # Mother
        mother_hidden = tfK.layers.Dense(self.dim_mother_hid, name = 'mother_hidden_layer' + self.label, activation = 'relu')(RNN)
        mother_unweighted_output = tfK.layers.Dense(self.dim_mother_out, name = 'mother_unweighted_output' + self.label, activation = 'softmax')(mother_hidden)
        mother_weighted_output = tfK.layers.multiply([input_mother_weights, mother_unweighted_output], name = 'multiply_weights' + self.label)
        mother_output = tfK.layers.Activation(self.normalize_layer, name = 'Normalize' + self.label)(mother_weighted_output)
        
        # Branch 
        branch_input  = tfK.layers.concatenate([RNN, masked_input_mother_momenta], axis=-1 , name = 'concatinate_branch_inputs' + self.label)
        branch_hidden = tfK.layers.Dense(self.dim_branch_hid, name = 'branch_hidden_layer' + self.label, activation = 'relu')(branch_input)
        branch_output = tfK.layers.Dense(self.dim_branch_out, name = 'branch_output_layer' + self.label, activation = 'softmax')(branch_hidden)
        
        return tfK.models.Model(
            inputs  = [input_seed_momenta, input_daughters, input_mother_momenta, input_mother_weights], 
            outputs = [end_output, mother_output, branch_output], 
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
                                loss = [self.binary_crossentropy_end, 
                                        self.categorical_crossentropy_mother, 
                                        self.sparse_categorical_crossentropy_branch])
    
    def train(self, data_path, n_events, granularity, epochs, learning_rates, batch_sizes, label, log_dir = './logs', save_dir='./saved_models', pickle_dir='./input_data/pickled', start_from_scratch = False):
        
        if start_from_scratch: # Decide if training should pick up from where it left off if same model was trained earlier, or if it should start from scratch
            open_file_mode = 'w' # truncate existing log file if it exists
        else:
            open_file_mode = 'a' # append to existing log file
        
        n_phases = len(epochs)
        train_file = open(log_dir + '/' + label+'_train_loss.log', open_file_mode)
        test_file  = open(log_dir + '/' + label+'_test_loss.log', open_file_mode)
        
        total_training_step = 0
        
        for n_epochs, learning_rate, batch_size in zip(epochs, learning_rates, batch_sizes):
            print('Loading data')
            [seed_momentum, daughters, mother_momenta, endings, ending_weights, mothers, mother_weights, sparse_branchings, sparse_branchings_weights] = load_data(data_path, n_events, batch_size, granularity, dim_mom = self.dim_mom, verbose=False, pickle_dir = pickle_dir, dim_mother_out = self.dim_mother_out)
            self.compile_model(learning_rate)
            n_train    = int((n_events//batch_size)*0.9)
            n_validate = int((n_events//batch_size)*0.1)
            
            for n in range(n_epochs):
                save_file_name = save_dir + '/' + label + "_weights_BS{}_LR{}_E{}".format(batch_size, learning_rate, n)
                
                if (start_from_scratch == False and os.path.exists(save_file_name)):
                    print("Loading model: " + save_file_name)
                    self.load_model(save_file_name)
                    total_training_step += n_train
                else:                
                    print("Starting epoch {} with learning_rate {} and batch size {}".format(n, learning_rate, batch_size))
                    train_loss = []
                    test_loss  = []

                    training_indices = list(range(0, n_train))
                    shuffle(training_indices) # shuffle training set for each epoch

                    for training_step, i in enumerate(training_indices):
                        print_progress(training_step, n_train)
                            
                        train_loss.append([total_training_step] + list(self.model.train_on_batch(
                            x = [seed_momentum[i], daughters[i], mother_momenta[i], mother_weights[i]],
                            y = [endings[i], mothers[i], sparse_branchings[i]])))
                        total_training_step += 1
                    for i in range(n_train, n_train+n_validate):
                        test_loss.append(self.model.test_on_batch(
                            x = [seed_momentum[i], daughters[i], mother_momenta[i], mother_weights[i]],
                            y = [endings[i], mothers[i], sparse_branchings[i]]))
                    # Write loss to file for each epoch
                    [train_file.write("{} {} {} {} {} \n".format(*line)) for line in train_loss]
                    test_file.write("{} {} {} {} {} \n".format(total_training_step, *np.average(test_loss, axis = 0)))
                    # Flush output losses
                    train_file.flush()
                    test_file.flush()

                    # Save model
                    tfK.models.save_model(self.model, save_file_name)
                
        train_file.close()
        test_file.close()
                       
    def validate(self, data_path, n_events, predict, model_path = None, batch_size = 100, granularity = 10, label = '', skip_first = 0, in_pickle_dir='./input_data/pickled', out_pickle_dir='./output_data/pickled', reload = False, reload_input_data = False, verbose = True):
        # Check if pickle_dir exists, or create it if not. 
        if not os.path.exists(out_pickle_dir):
            os.makedirs(out_pickle_dir)
    
        if skip_first>0:
            SKIP = "_SKIP" + str(skip_first)
        else:
            SKIP = ""
        
        if model_path is None:
            model_basename = "DataSet"
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
                branchings_out = pickle.load(f)
                branchings_counts_out = pickle.load(f)
                # Collect all outputs in one list
                outputs = [endings_out, ending_counts_out, mothers_out, mother_counts_out, branchings_out, branchings_counts_out]
                # If predict, get probabilities
                if predict:
                    probabilities = pickle.load(f)
        
        else:
            # Load model
            if model_path is not None:
                self.load_model(model_path)
        
            all_data = load_data(data_path, n_events = n_events, batch_size = batch_size, dim_mom = self.dim_mom, granularity = granularity, skip_first = skip_first, pickle_dir = in_pickle_dir, reload = reload_input_data, dim_mother_out = self.dim_mother_out)       
            [seed_momentum, daughters, mother_momenta, endings, ending_weights, mothers, mother_weights, sparse_branchings, sparse_branchings_weights] = all_data

        
            n_batches             = len(daughters)
        
            # Create ouput arrays
            endings_out           = np.zeros((self.max_time, 1))
            ending_counts_out     = np.zeros((self.max_time, 1))
        
            mothers_out           = np.zeros((self.max_time, self.dim_mother_out))
            mother_counts_out     = np.zeros((self.max_time, self.dim_mother_out))

            branchings_out        = np.zeros((self.max_time, self.dim_branch_out))
            branchings_counts_out = np.zeros((self.max_time, 1))
        
            probabilities         = []
        
            # Organize outputs in one array
            outputs = [endings_out, ending_counts_out, mothers_out, mother_counts_out, branchings_out, branchings_counts_out]

            # Predict on batches 
            for batch_i in range(n_batches):
                print_progress(batch_i, n_batches)
                if predict:
                    # Pretict on a single batch 
                    e, m, b = self.model.predict_on_batch(x = [seed_momentum[batch_i], daughters[batch_i], mother_momenta[batch_i], mother_weights[batch_i]])
                else:
                    # Use one batch from input data
                    e = endings[batch_i]
                    m = mothers[batch_i]
                    b = sparse_branchings[batch_i]

                batch_length, max_time = e.shape[:2]
                for jet_i in range(batch_length):
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
                        if sparse_branchings_weights[batch_i][jet_i, t] == 1:
                            if predict:
                                branchings_out[t]               += b[jet_i, t]
                            else:
                                branchings_out[t, b[jet_i, t]]  += 1 # input data is sparse

                            branchings_counts_out[t] += 1

                    # Jet probabilities
                    if predict:
                        p_e = np.asarray(e[jet_i]*ending_weights[batch_i][jet_i]).flatten()
                        p_m = m[jet_i][mothers[batch_i][jet_i]]
                        branch_indices = tfK.utils.to_categorical(sparse_branchings[batch_i][jet_i]).astype('bool')
                        p_b = (b[jet_i][branch_indices]*sparse_branchings_weights[batch_i][jet_i].T).flatten()

                        length = len(p_m)
                        probs = np.append((1-p_e[:length])*p_m*p_b[:length], p_e[length])
                        probabilities.append(np.prod(probs, dtype=np.float64))
                        
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



