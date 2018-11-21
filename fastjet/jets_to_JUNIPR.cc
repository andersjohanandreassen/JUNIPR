// compile:
// g++ jets_to_JUNIPR.cc -o jets_to_JUNIPR `../../../Programs/fastjet-install/bin/fastjet-config --cxxflags --libs --plugins`
// run:
// ./jets_to_JUNIPR input_directory input_file output_directory recluster_def

#include "fastjet/ClusterSequence.hh"
#include <iostream>
#include <fstream>
#include <sstream>
#include "fastjet_JUNIPR_utilities.cc"

using namespace fastjet;
using namespace std;

#define NEVENTS   1000000

int main(int argc, char *argv[]) {

  if (argc != 5) {    
    cout << "Wrong number of arguments" << endl;
    return 0;
  }
  
  // User input
  string input_directory = argv[1];
  string input_file = argv[2];
  string output_directory = argv[3];
  double recluster_def = atof(argv[4]);

  // Input, output
  stringstream s_infile;
  s_infile << input_directory << "/" << input_file;
  ifstream infile;
  infile.open( s_infile.str().c_str() );

  stringstream s_outfile;
  s_outfile << output_directory << "/JUNIPR_format_" << input_file ;
  ofstream outfile;
  outfile.open( s_outfile.str().c_str());
  

  // Read jets from input file
  // Jets begin with J <jet number> N <number of subjets>
  // Following N lines are px py pz e for subjets
  string dummy;
  int ijet;
  int nsubjets;

  // Loop over jets
  int jet_counter = 0;
  while ((infile >> dummy) && (infile >> ijet) && (ijet <= NEVENTS)) {

    if (jet_counter % 10000 == 0) cout << "processing jet " << jet_counter << endl;

    infile >> dummy >> nsubjets;

    vector<PseudoJet> particles;

    // Particle loop
    for (int i = 0; i < nsubjets; i++) {

      double px, py, pz, e;
      infile >> px >> py >> pz >> e;

      PseudoJet particle(px, py, pz, e);
      particles.push_back(particle);

    }  // end loop (particles)
    
    // Recluster jet constituents
    JetDefinition reclust_def(ee_genkt_algorithm, 7, recluster_def); // radius unused
    ClusterSequence reclust_seq(particles, reclust_def);
    vector<PseudoJet> reclust_jets = reclust_seq.exclusive_jets(1);
    if (jet_counter == 0)
      cout << "Reclustered with " << reclust_def.description() << endl;
    if(reclust_jets[0].constituents().size() != nsubjets){
      cout << "Mismatch between nsubjets and reclustered jet" << endl;
    }
    outfile << "J " << jet_counter << endl;
    convert_cluster_sequence_to_JUNIPR(reclust_seq, outfile);
  
    jet_counter++;
} // end loop (jets)



  infile.close();
  outfile.close();
  return 0;

} // end function (main)

