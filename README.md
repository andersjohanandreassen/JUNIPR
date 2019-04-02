# JUNIPR

**This code is in a public beta. Parts of the code is still in development and new features will be added in the near future.** 

## Introduction

JUNIPR is a TF2.0.0-alpha0 implementation of the JUNIPR model introduced in arXiv:1804.09720

An old version written in TF1.* is stored in branch JUNIPR_TFv1. 

## Import data
### Convert from fastjet to JuniprJets.json
To convert your data to JuniprJets format, please use the code provided in the fastjet directory. 
Compile `jets_to_JUNIPR.cc` with [fastjet][1] and run to create a .json file with JuniprJet-format. 
A sample of jets are provided in `fastjet/raw_jets` and a converted sample is provided in `data/json`. 

### Convert from json to TFRecord
Use the `create_TFRecord` in `junipr.tfrecord.writer_utils` to convert the json file to a TFRecord. 

An example is provided in `Create_TFRecord.ipynb`. Note that this must run with eager execution. 

A sample TFRecord is provided in `data/tfrecord`

## Build and Train Model
The JUNIPR model is defined in `junipr.junipr`. 

We provide an example of how to train a model in `JUNIPR_demo.ipynb`.


[1] http://fastjet.fr