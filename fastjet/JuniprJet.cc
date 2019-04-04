#include <iostream>

using namespace std;

class JuniprJet
{
    public:
        JuniprJet(vector< vector<double> > CSJets_arg,
                  vector< vector<int> >    intermediate_states_arg, 
                  vector<int>              mothers_arg, 
                  vector< vector<int> >    daughters_arg,
                  vector< vector<double> > branchings_arg,
                  int label_arg);
        
        // Variables set by constructor:
        
        // CSJets contains the ClusteringSequenceJets. Each element is a 4-momentum (E, theta, phi, mass) 
        vector< vector<double> > CSJets;
        
        //// intermediate_states, mother and daugthers contain indices to the entries in CSJets
        // intermediate_states[i] is a list of the intermediate states at step i in energy order (high to low)
        vector< vector<int> >    intermediate_states;
        // mothers[i] is the index to the decaying mother at time step i
        vector<int>              mothers;
        // daughters[i] is the index to the two daughters at time step i. Most energetic daughter first. 
        vector< vector<int> >    daughters;
        // branchings[i] is the branchings (z, theta, phi, delta) at time step i. 
        vector< vector<double> > branchings;
    
        // mother_id_energy_order[i] is the id of the mother that branches in the list of energy ordered intermediate states at time step i
        vector<int> mother_id_energy_order;
    
        int label; // label for used for classification target value
        
        int multiplicity; // number of final state particles
        int n_branchings; // number of branchings/
        
        // Seed_momentum = sum of all final state momenta
        vector<double>                     seed_momentum;
        // daughter_momenta[i] contains the two 4-momenta (E, theta, phi, mass) of the daughters produced at time step i.
        vector< vector< vector<double> > > daughter_momenta;
        // mother_momenta[i] contains the 4-momenta (E, theta, phi, mass) of the mother that branches at time step i.
        vector< vector<double> >           mother_momenta;
    
        void write_to_json(ostream& outfile);
        
        
    private:
        void set_multiplicity();
        void set_n_branchings();
        void set_seed_momentum();
        void set_mother_momenta();
        void set_daughter_momenta();
        void set_mother_id_energy_order();    
    
        // Printing functions in json format
        void json_momentum(ostream& outfile, vector<double> momentum);
        void json_vector_int(ostream& outfile, vector<int> vector_int);
        void json_vector_double(ostream& outfile, vector<double> vector_double);
        
        void json_label(ostream& outfile);    
        void json_multiplicity(ostream& outfile);
        void json_n_branchings(ostream& outfile);
        void json_seed_momentum(ostream& outfile);
        void json_CSJets(ostream& outfile);
        void json_intermediate_states(ostream& outfile);
        void json_mothers(ostream& outfile);
        void json_mother_id_energy_order(ostream& outfile);
        void json_daughters(ostream& outfile);
        void json_branchings(ostream& outfile);
        void json_mother_momenta(ostream& outfile);
        void json_daughter_momenta(ostream& outfile);
    
};



/////////////////////////////

JuniprJet::JuniprJet(vector< vector<double> > CSJets_arg,
                     vector< vector<int> >    intermediate_states_arg, 
                     vector<int>              mothers_arg, 
                     vector< vector<int> >    daughters_arg,
                     vector< vector<double> > branchings_arg,
                     int label_arg){

    label               = label_arg;
    CSJets              = CSJets_arg;
    intermediate_states = intermediate_states_arg; 
    mothers             = mothers_arg;
    daughters           = daughters_arg;
    branchings          = branchings_arg;
    
    set_multiplicity();
    set_n_branchings();
    set_seed_momentum();
    set_mother_momenta();
    set_daughter_momenta();
    set_mother_id_energy_order();
}

void JuniprJet::set_multiplicity(){
    multiplicity = (CSJets.size()+1)/2;
}

void JuniprJet::set_n_branchings(){
    n_branchings = (CSJets.size()-1)/2;
}

void JuniprJet::set_seed_momentum(){
    seed_momentum = CSJets[mothers[0]];
}

void JuniprJet::set_mother_momenta(){
    for(int i = 0; i< mothers.size(); i++){
        mother_momenta.push_back(CSJets[mothers[i]]);
    }
}

void JuniprJet::set_daughter_momenta(){
    for(int i = 0; i< daughters.size(); i++){
        vector<double> daughter1 = CSJets[daughters[i][0]];
        vector<double> daughter2 = CSJets[daughters[i][1]];
        
        vector< vector<double> > two_daughters;
        two_daughters.push_back(daughter1);
        two_daughters.push_back(daughter2);
        daughter_momenta.push_back(two_daughters);
    }
}

void JuniprJet::set_mother_id_energy_order(){
    for (int i = 0; i< mothers.size(); i++){
        for (int j = 0; j< intermediate_states[i].size(); j++){
            if(mothers[i] == intermediate_states[i][j]){
                mother_id_energy_order.push_back(j);
                break;
            }
        }
    }
    if(mothers.size() !=mother_id_energy_order.size()){
        cout << "Error in set_mother_id_energy_order(): one or more mother_ids not found in intermediate_states" << endl;
    }
}

// Printing functions in json format

void JuniprJet::write_to_json(ostream& outfile){
    
    outfile << "{\n\t" ; 
    
    json_label(outfile);
        outfile << ",\n\t";
    json_multiplicity(outfile);
        outfile << ",\n\t";
    json_n_branchings(outfile);
        outfile << ",\n\t";
    json_seed_momentum(outfile);
        outfile << ",\n\t";
    json_CSJets(outfile);
        outfile << ",\n\t";
    json_intermediate_states(outfile);
        outfile << ",\n\t";
    json_mother_id_energy_order(outfile);
         outfile << ",\n\t";
    json_mothers(outfile);
        outfile << ",\n\t";
    json_daughters(outfile);
        outfile << ",\n\t";
    json_branchings(outfile);
        outfile << ",\n\t";
    json_mother_momenta(outfile);
            outfile << ",\n\t";
    json_daughter_momenta(outfile);
        outfile << "\n}" ; 
}

void JuniprJet::json_label(ostream& outfile){
    outfile << "\"label\" : " << label ;
}

void JuniprJet::json_multiplicity(ostream& outfile){
    outfile << "\"multiplicity\" : " << multiplicity ;
}

void JuniprJet::json_n_branchings(ostream& outfile){
    outfile << "\"n_branchings\" : " << n_branchings ;
}

void JuniprJet::json_momentum(ostream& outfile, vector<double> momentum){
    outfile << "[ " 
            << momentum[0] << " , "
            << momentum[1] << " , "
            << momentum[2] << " , "
            << momentum[3] << " ]";
}

void JuniprJet::json_vector_double(ostream& outfile, vector<double> vector_double){
    outfile << "[ ";
    
    for(int i = 0; i<vector_double.size(); i++){
        outfile << vector_double[i];
        if(i!=vector_double.size()-1){
            outfile << " , ";
        }      
    }
    outfile << " ]";
}

void JuniprJet::json_vector_int(ostream& outfile, vector<int> vector_int){
    outfile << "[ ";
    
    for(int i = 0; i<vector_int.size(); i++){
        outfile << vector_int[i];
        if(i!=vector_int.size()-1){
            outfile << " , ";
        }      
    }
    outfile << " ]";
}

void JuniprJet::json_seed_momentum(ostream& outfile){
    outfile << "\"seed_momentum\" :" ;
    json_momentum(outfile, seed_momentum);
}

void JuniprJet::json_CSJets(ostream& outfile){
    outfile << "\"CSJets\" : [\n\t\t";
    for (int i = 0; i<CSJets.size(); i++){
        json_momentum(outfile, CSJets[i]);
        if(i!=CSJets.size()-1){
            outfile << ",\n\t\t";
        }
    }
    outfile << "]";
}

void JuniprJet::json_intermediate_states(ostream& outfile){
    outfile << "\"CS_ID_intermediate_states\": [\n\t\t";
    for (int i = 0; i< intermediate_states.size(); i++){
        json_vector_int(outfile, intermediate_states[i]);
        if(i!=intermediate_states.size()-1){
            outfile << ",\n\t\t";
        }
    }
    outfile << "]";
    
}

void JuniprJet::json_mothers(ostream& outfile){
    outfile << "\"CS_ID_mothers\": ";
    json_vector_int(outfile, mothers);    
}

void JuniprJet::json_mother_id_energy_order(ostream& outfile){
    outfile << "\"mothers_id_energy_order\": ";
    json_vector_int(outfile, mother_id_energy_order);    
}

void JuniprJet::json_daughters(ostream& outfile){
    outfile << "\"CS_ID_daughters\": [\n\t\t";
    for (int i = 0; i< daughters.size(); i++){
        json_vector_int(outfile, daughters[i]);
        if(i!=daughters.size()-1){
            outfile << ",\n\t\t";
        }
    }
    outfile << "]";
    
}

void JuniprJet::json_branchings(ostream& outfile){
    outfile << "\"branchings\" : [\n\t\t";
    for (int i = 0; i<branchings.size(); i++){
        json_vector_double(outfile, branchings[i]); 
        if(i!=branchings.size()-1){
            outfile << ",\n\t\t";
        }
    }
    outfile << "]";
}

void JuniprJet::json_mother_momenta(ostream& outfile){
    outfile << "\"mother_momenta\" : [\n\t\t";
    for (int i = 0; i<mother_momenta.size(); i++){
        json_momentum(outfile, mother_momenta[i]);
        if(i!=mother_momenta.size()-1){
            outfile << ",\n\t\t";
        }
    }
    outfile << "]";
}

void JuniprJet::json_daughter_momenta(ostream& outfile){
    outfile << "\"daughter_momenta\" : [\n\t\t";
    for (int i = 0; i<daughter_momenta.size(); i++){
        outfile << "[";
        json_momentum(outfile, daughter_momenta[i][0]);
        outfile << " , ";
        json_momentum(outfile, daughter_momenta[i][1]);
        outfile << " ]";
     
        if(i!=daughter_momenta.size()-1){
            outfile << ",\n\t\t";
        }
    }
    outfile << "]";
}