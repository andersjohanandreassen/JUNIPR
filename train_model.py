import numpy as np
from JUNIPR import *

data_path = './input_data/JUNIPR_format_jets.dat'
data_label = 'quarks'

n_events    = 2*10**4
granularity = 10
optimizer   = 'SGD'
RNN_type    = None # None == SimpleRNN

junipr = JUNIPR(label = data_label, optimizer = optimizer, RNN_type = RNN_type)
 
epochs = [5, 5, 5, 5, 5, 5]
leaning_rates = [1e-2, 1e-3, 1e-4, 1e-3, 1e-4, 1e-5]
batch_sizes = [10, 10, 10, 100, 100, 100]
label = "JUNIPR_test"
junipr.train(data_path, n_events, granularity, epochs, leaning_rates, batch_sizes, label)
