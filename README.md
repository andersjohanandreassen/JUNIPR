# JUNIPR

** This code is in a public beta. Parts of the code is still in development and new features will be added in the near future. ** 

JUNIPR is a tf.keras implementation of the JUNIPR model introduced in arXiv:1804.09720

The code is written with tensorflow version 1.11. Compatibility with earlier versions is not guaranteed. 

Another README file is included in the folder ./fastjet/ with some more details as how to convert jets from e.g. Pythia to the input format needed by JUNIPR. 
A fastjet code doing the conversion is also included. 

To explore the input format to JUNIPR, please use the explore_data.ipynb notebook. 

The model itself is defined in JUNIPR.py

Sample data (20000 jets) is included in the ./input_data/ folder to make it easy to test out the code. 

A pre-trained model is included in ./saved_models/ and can be studied using the ./validate_model.ipynb notebook. 

You can try to train the model using the ./train_model.py python script. It will train using the included sample data from ./input_data/, and should not require any other configuration other than having tensorflow installed. 
During training you can monitor the losses using the notebook ./monitor_training.ipynb.