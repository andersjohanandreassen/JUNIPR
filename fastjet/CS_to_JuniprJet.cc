#include "fastjet/ClusterSequence.hh"
#include <iostream>
#include <stdio.h>
#include <cmath>
#include <algorithm>
#include "utils.cc"
#include "JuniprJet.cc"
using namespace fastjet;
using namespace std;

vector<double> get_branching(vector<PseudoJet> const &PJ_CSJets, int hist_id, vector<int> current_daughters){
    // Calculate branching (z, theta, phi, delta)
    
    PseudoJet mother        = PJ_CSJets[hist_id];
    PseudoJet hard_daughter = PJ_CSJets[current_daughters[0]];
    PseudoJet soft_daughter = PJ_CSJets[current_daughters[1]];
    

    double z     = soft_daughter.e()/mother.e(); // z is defined as the energy fraction of the soft daughter. 
    double theta = get_thetaPJ(soft_daughter, mother);
    double phi   = get_phiPJ(soft_daughter, mother);
    double delta = get_thetaPJ(hard_daughter, mother);

    vector<double> branching;
    branching.push_back(z);
    branching.push_back(theta);
    branching.push_back(phi);
    branching.push_back(delta);

    return branching;
}

void remove_hist_id(vector<int> &current_intermediate_state, int hist_id){
    // remove hist_id from current_intermediate_state
    current_intermediate_state.erase(remove(current_intermediate_state.begin(), current_intermediate_state.end(), hist_id), current_intermediate_state.end());
}

void add_hist_id(vector<int> &current_intermediate_state, int hist_id, vector< vector<double> > const &CSJets){
    // Add hist_id to current_intermediate_state in energy order
    
    // Keep track if value has been inserted
    bool inserted_new_value = false;
    
    for (int i = 0; i<current_intermediate_state.size(); i++){
        // Check if particle at hist_id has higher energy than particle i in current_intermediate_state
        if(CSJets[current_intermediate_state[i]][0] < CSJets[hist_id][0]){
            current_intermediate_state.insert(current_intermediate_state.begin()+i, hist_id);
            inserted_new_value = true;
            break;
        }
    }
    // If not inserted in between previous values of current_intermediate_state, add to the end. 
    if (!inserted_new_value){
        current_intermediate_state.push_back(hist_id);
    }
    
}

JuniprJet cluster_sequence_to_JuniprJet(ClusterSequence cs, int label){
    // History: Contains history_elements that lists parents and children at every step in the clusterin history
    // During clustering two parents are merged together to one child
    vector<ClusterSequence::history_element> hist = cs.history();
    
    // CSJets(ClusteringSequenceJets): List of all jets during clustering history. 
    // The first N jets are the final states. They appear in the order they were when the ClusterSequence was called.
    // As pseudojets are clustered to new pseudojets, the combined new pseudojet is added to the end of the CSJets list. 
    // There is a one-to-one correspondance between the index in the jet history (hist) and the jets (CSJets). 
    // Note that hist has one extra element compared to the jet list (correspoding to the end of clustering)
    vector<PseudoJet> PJ_CSJets = cs.jets();

    // The total jet momentum (=jet_axis) is in the last element of PJ_CSJets  
    PseudoJet jet_axis = PJ_CSJets[PJ_CSJets.size()-1];  

    // Convert CSJets from PseudoJet to format (E, theta, phi, mass)
    vector< vector<double> > CSJets;
    for (int i = 0; i<PJ_CSJets.size(); i++){
        CSJets.push_back(PJ_to_ETPM(PJ_CSJets[i], jet_axis));
    }
    
    // Build intermediate states and list of mothers and daughters.
    // The indices in the intermediate states correspond to the location of the four momentum in CSJets
    vector< vector<int> > intermediate_states;
    vector<int> current_intermediate_state;
    vector<int> mothers;
    vector< vector<int> > daughters;
    
    // Branchings (z, theta, phi, delta) for each local branching
    vector< vector<double> > branchings;
    
    // Loop backwards through the clustering history to decluster jet
    // Last history element at hist[hist.size()-1] indicates end of clustering, so we skip it.  
    current_intermediate_state.push_back(hist.size()-2);// Add seed particle

    // Copy current_intermediate_state into intermediate_states
    intermediate_states.push_back(vector<int>(current_intermediate_state));
    
    for(int hist_id = hist.size()-2; hist_id>=0; hist_id--){
        if(hist[hist_id].parent1 == -2 || hist[hist_id].parent2 == -2){
            // Final state particle
            // Do nothing. 
        }else{
            // Intermediate PseudoJet
            
            // Remove mother from current intermediate states
            remove_hist_id(current_intermediate_state, hist_id); 
            
            // Add index of mother to mothers
            mothers.push_back(hist_id);

            // Add daughters to current intermediate states
            add_hist_id(current_intermediate_state, hist[hist_id].parent1, CSJets);
            add_hist_id(current_intermediate_state, hist[hist_id].parent2, CSJets);
            
            // Add daughters to list of daughter indices
            // Order them so that the most energetic is first
            vector<int> current_daughters;
            if(CSJets[hist[hist_id].parent1][0]>CSJets[hist[hist_id].parent2][0]){
                current_daughters.push_back(hist[hist_id].parent1);
                current_daughters.push_back(hist[hist_id].parent2);
            }else{
                current_daughters.push_back(hist[hist_id].parent2);
                current_daughters.push_back(hist[hist_id].parent1);
            }
            daughters.push_back(current_daughters);
          
            // Copy into intermediate_states
            intermediate_states.push_back(vector<int>(current_intermediate_state)); 
            
            // Add branching
            branchings.push_back(get_branching(PJ_CSJets, hist_id, current_daughters));
        }
    }  
    
    JuniprJet jet(CSJets, 
                  intermediate_states,
                  mothers, 
                  daughters,
                  branchings,
                  label);
    return jet;
}

