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

vector<double> PJ_to_ETPM(PseudoJet PJ, PseudoJet ref){
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