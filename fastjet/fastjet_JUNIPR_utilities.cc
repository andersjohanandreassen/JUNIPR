#include "fastjet/ClusterSequence.hh"
#include <iostream>
#include <stdio.h>
#include <cmath>
#include <algorithm>
using namespace fastjet;
using namespace std;

vector<double> PJ_3mom(PseudoJet PJ){
  // Convert PseudoJet to a 3-momentum vector<double>

  vector<double> v;
  v.push_back(PJ.px());
  v.push_back(PJ.py());
  v.push_back(PJ.pz());

  return v;
}

double dot3(vector<double> v1, vector<double> v2){
  // Inner product between two 3-vectors 

  return v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2] ;
}

vector<double> hat(vector<double> v){
  // Calculate unit vector of a 3-vector. 

  if(v.size()!=3){
    cout << "Error: Wrong length in hat(vector<double> v)" << endl;
  }

  vector<double> hv;
  double norm = sqrt(dot3(v, v));
  hv.push_back(v[0]/norm);
  hv.push_back(v[1]/norm);
  hv.push_back(v[2]/norm);

  return hv;
}

double get_theta(vector<double> v1, vector<double> v2){
  // Angle between 3-vectors

  double v1_norm  = sqrt(dot3(v1, v1));
  double v2_norm  = sqrt(dot3(v2, v2));    
  double cosTheta = dot3(v1, v2)/(v1_norm*v2_norm);
  
  if(abs(cosTheta-1)<1e-15){
    return 0;
  }else if(abs(cosTheta+1)<1e-15){
    return M_PI;
  }else{
    return acos(cosTheta);
  }
}

double get_thetaPJ(PseudoJet PJ1, PseudoJet PJ2){
  // Angle between 3-momentum of two PseudoJets
  return get_theta(PJ_3mom(PJ1), PJ_3mom(PJ2));
}

vector<double> cross(vector<double> v1, vector<double> v2){
  // Calculates cross product of two 3-vectors

  if(v1.size()!=3 || v2.size()!=3){
    cout << "Error: Wrong length in cross product" << endl;
  }

  vector<double> c;
  c.push_back(v1[1]*v2[2] - v1[2]*v2[1]);
  c.push_back(v1[2]*v2[0] - v1[0]*v2[2]);
  c.push_back(v1[0]*v2[1] - v1[1]*v2[0]);
  return c;
}

double get_phiPJ(PseudoJet p, PseudoJet ref){
  // Calculate phi
  
  // Get 3-momentum
  vector<double> p3 = PJ_3mom(p);
  vector<double> ref3 = PJ_3mom(ref);
  
  if(abs(get_theta(p3, ref3))< 1e-15){
    return 0.0;
  }
    
  // Define x,y,z directions by a convention
  // Define cartesian y-direction
  vector<double> cart_y;
  cart_y.push_back(0);
  cart_y.push_back(1);
  cart_y.push_back(0);
  
  vector<double> e_z = hat(ref3); // z is the direction of the reference vector
  vector<double> e_x = hat(cross(cart_y, e_z));
  vector<double> e_y = hat(cross(e_z, e_x));

  // Project p into xy plane
  vector<double> pxy;
  pxy.push_back(p3[0]-e_z[0]*dot3(e_z, p3));
  pxy.push_back(p3[1]-e_z[1]*dot3(e_z, p3));
  pxy.push_back(p3[2]-e_z[2]*dot3(e_z, p3));

  // Determine azimuthal angle in xy plane
  double phi_y = get_theta(pxy, e_y);
  double phi   = get_theta(pxy, e_x);

  if(phi_y > M_PI/2.){
    phi = 2*M_PI - phi;
  }
  return phi;

}

vector<double> get_branching(PseudoJet mother, PseudoJet daughter1, PseudoJet daughter2){
  // Calculate branching (z, theta, phi, delta)
  
  PseudoJet soft_daughter, hard_daughter;
  if(daughter1.e()>daughter2.e()){
    soft_daughter = daughter2;
    hard_daughter = daughter1;
  }else{
    soft_daughter = daughter1;
    hard_daughter = daughter2;
  }

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

int get_mother(vector<int> intermediate_states, vector<PseudoJet> jets, int mother_index){
  // Index of the next mother to branch in an energy ordered list

  // Mother's energy
  double E = jets[mother_index].e();
  int counter = 0;
  
  // count the number of intermediate states that have higher energy than the chosen mother
  for(int i=0; i< intermediate_states.size(); i++){
    if(jets[intermediate_states[i]].e()>E){
      counter++;
    }
  }

  return counter;

}

vector<double> XYZE_to_ETPM(PseudoJet PJ, PseudoJet ref){
  // Convert PseudoJet to (Energy, theta, phi, mass) where angles are relative to a reference vector
  double E     = PJ.e();
  double theta = get_thetaPJ(PJ, ref);
  double phi   = get_phiPJ(PJ, ref);
  double mass  = PJ.m();

  vector<double> ETPM;
  ETPM.push_back(E);
  ETPM.push_back(theta);
  ETPM.push_back(phi);
  ETPM.push_back(mass);

  return ETPM;
}

vector<double> get_mother_momenta(PseudoJet mother, PseudoJet ref){
  // return momenta of mother as (Energy, theta, phi, mass) relative to a reference vector
  return XYZE_to_ETPM(mother, ref);
}

vector<double> get_daughter_momenta(PseudoJet daughter1, PseudoJet daughter2, PseudoJet ref){
  // return momenta of daughters as (Energy, theta, phi, mass) relative to a reference vector
  // A pair of daughters will be returned (most energetic always first).
  PseudoJet soft_daughter, hard_daughter;
  if(daughter1.e()>daughter2.e()){
    soft_daughter = daughter2;
    hard_daughter = daughter1;
  }else{
    soft_daughter = daughter1;
    hard_daughter = daughter2;
  }
  
  vector<double> soft_ETPM = XYZE_to_ETPM(soft_daughter, ref);
  vector<double> hard_ETPM = XYZE_to_ETPM(hard_daughter, ref);
  
  // Append soft daougher to the hard one
  for(int i=0; i<soft_ETPM.size(); i++){
    hard_ETPM.push_back(soft_ETPM[i]);
  }
  
  return hard_ETPM;
}

void convert_cluster_sequence_to_JUNIPR(ClusterSequence cs, ostream& outfile){
  // History: Contains history_elements that lists parents and children at every step in the clusterin history
  // During clustering two parents are merged together to one child
  vector<ClusterSequence::history_element> hist = cs.history();
  
  // Jets: List of all jets during clustering history. 
  // The first N jets are the final states. 
  // As pseudojets are clustered to new pseudojets, the combined new pseudojet is added to the jet list. 
  // There is a one-to-one correspondance between the index in the jet history (hist) and the jets (my_jets). Note that hist has one extra element compared to the jet list (correspoding to the end of clustering?)
  vector<PseudoJet> jets = cs.jets();

  // Since we are declustering I will refer to the fastjet-children as mothers nad fastjet-parents as daughters. This is consistent with the usage in JUNIPR paper. 
  
  // Define input vecotrs
  vector< vector<double> > mother_momenta;
  vector< vector<double> > daughter_momenta;

  // Define output vectors
  int end = 0;
  vector<int> mothers;
  vector< vector<double> > branchings;


  // Loop backwards through the clustering history to decluster jet
  // Last history element at hist[hist.size()-1] indicates end of clustering, so we skip it.  
  vector<int> intermediate_states;
  intermediate_states.push_back(hist.size()-2);// Add seed particle

  PseudoJet jet_axis = jets[hist.size()-2];

  for(int hist_id = hist.size()-2; hist_id>=0; hist_id--){
    if(hist[hist_id].parent1 == -2 || hist[hist_id].parent2 == -2){
      // Final state particle
    }else{
      // Intermediate PseudoJet
      PseudoJet mother    = jets[hist_id];
      PseudoJet daughter1 = jets[hist[hist_id].parent1];
      PseudoJet daughter2 = jets[hist[hist_id].parent2];

      // Remove mother from current intermediate states
      intermediate_states.erase(remove(intermediate_states.begin(), intermediate_states.end(), hist_id), intermediate_states.end());  

      // Add daughters to current intermediate states
      intermediate_states.push_back(hist[hist_id].parent1);
      intermediate_states.push_back(hist[hist_id].parent2);

      // JUNIPR Inputs
      daughter_momenta.push_back(get_daughter_momenta(daughter1, daughter2, jet_axis));
      mother_momenta.push_back(get_mother_momenta(mother, jet_axis));

      // JUNIPR Outputs
      end++;
      mothers.push_back(get_mother(intermediate_states, jets, hist_id));
      branchings.push_back(get_branching(mother, daughter1, daughter2));

    }
  }

  // Print Jet to file

  PseudoJet seed = jets[hist.size()-2]; 
  double pz = sqrt(pow(seed.px(),2) + pow(seed.py(),2) + pow(seed.pz(),2)); // rotate to point in z-direction

  // Seed momentum (oriented to z direction)
  outfile << "O " << 0  << " " << 0 << " "  << pz << " "  << seed.e() << endl;
   
  // Ending
  outfile << "E " << end << endl;

  // Mothers
  outfile << "M ";
  for(int i =0; i<mothers.size(); i++){
    outfile << mothers[i] << " ";
  }
  outfile << endl;
  
  // Branchings
  outfile << "B " ;
  for(int i =0; i<branchings.size(); i++){
    for(int j=0; j<4; j++){
      outfile << branchings[i][j] << " ";
    }
  }
  outfile << endl;

  // Daughters
  outfile << "D " ;
  for(int i =0; i<daughter_momenta.size(); i++){
    for(int j=0; j<8; j++){
      outfile << daughter_momenta[i][j] << " ";
    }
  }
  outfile << endl;
 
  // Mother momenta
  outfile << "P " ;
  for(int i =0; i<mother_momenta.size(); i++){
    for(int j=0; j<4; j++){
      outfile << mother_momenta[i][j] << " ";
    }
  }
  outfile << endl; 
}
