#include "bond_axis.h"
#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "neighbor.h"
#include "domain.h"

#include <cmath>
#include <cstring>
#include <iostream>
#include "atom_vec_ellipsoid.h"
#include "math_extra.h"
#include "neigh_list.h"
#include <vector>
#include <stdexcept>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

BondAxis::BondAxis(LAMMPS *_lmp) : Bond(_lmp)
{ 
  kr = nullptr;
  kR = nullptr;
  kphi = nullptr;
  kalpha = nullptr;
  R = nullptr;
  r0 = nullptr;
  comm_forward = 7;
}

/* ---------------------------------------------------------------------- */

BondAxis::~BondAxis()
{
  if (allocated && !copymode) {
    memory->destroy(setflag);
    memory->destroy(kr);
    memory->destroy(kR);
    memory->destroy(kalpha);
    memory->destroy(kphi);
    memory->destroy(R);
    memory->destroy(r0);
    
  }
}

/* ---------------------------------------------------------------------- */

void BondAxis::compute(int eflag, int vflag)
{
  int i1, i2, n, type;
  double delx, dely, delz, ebond, fbond, a0, alpha, dUda0, a0_0;
  double force_kphi_x_i1_i2, force_kphi_y_i1_i2, force_kphi_z_i1_i2; 
  double force_kphi_x_i2_i1, force_kphi_y_i2_i1, force_kphi_z_i2_i1; 
  double c11, c22, c33, c12, c13, c23, d12, d13, d23 , a3, dUda3 , phi, f ,g, a3_0, c11_0, c22_0, c33_0, c12_0, c13_0, c23_0;
  double da3df1_x, da3df1_y, da3df1_z;  
  double da3dr_x, da3dr_y, da3dr_z;
  double da3df2_x, da3df2_y, da3df2_z;
  double d_mag, dx, dy, dz;
  double r, delx_norm, dely_norm, delz_norm;
  double a[3][3], b[3][3];
  double *iquat, *jquat;
  AtomVecEllipsoid::Bonus *bonus = avec->bonus;
  int *ellipsoid = atom->ellipsoid;

  fbond = 0.0;
  ebond = 0.0;
  ev_init(eflag, vflag);

  double **x = atom->x;
  double **forc = atom->f;
  double **tor = atom->torque;
  int **bondlist = neighbor->bondlist;
  int nbondlist = neighbor->nbondlist;
  int nlocal = atom->nlocal;
  int newton_bond = force->newton_bond;
  tagint *tag = atom->tag;
  int **nspecial = atom->nspecial;
  tagint **special = atom->special;

  double small = 1e-30; // Is an angle in radians

  bigint nellipsoids = atom->nellipsoids;
  bigint natoms = atom->natoms;

  //comm_forward = 100*(natoms + nellipsoids);  
  comm->forward_comm(this); 
  MPI_Barrier(MPI_COMM_WORLD);

  // kR

  dUda0 = 0.0;
  alpha = 0.0;
  double force_kR_x_i1 = 0.0;
  double force_kR_y_i1 = 0.0;
  double force_kR_z_i1 = 0.0;

  double force_kR_x_i2 = 0.0;
  double force_kR_y_i2 = 0.0;
  double force_kR_z_i2 = 0.0;

  double torque_gx_i1 = 0.0;
  double torque_gy_i1 = 0.0;
  double torque_gz_i1 = 0.0;

  double torque_gx_i2 = 0.0;
  double torque_gy_i2 = 0.0;
  double torque_gz_i2 = 0.0;

  std::vector<double> g_i1_cross_dr(3);
  std::vector<double> g_i2_cross_dr(3);

  std::vector<double> g_i1_cross_g_i2(3);
  std::vector<double> g_i2_cross_g_i1(3);


  
  //phi
  dUda3 = 0.0;
  phi = 0.0;
  double force_kphi_x_i1 = 0.0;
  double force_kphi_y_i1 = 0.0;
  double force_kphi_z_i1 = 0.0;

  double force_kphi_x_i2 = 0.0;
  double force_kphi_y_i2 = 0.0;
  double force_kphi_z_i2 = 0.0;

  double torque_fx_phi_i1 = 0.0;
  double torque_fy_phi_i1 = 0.0;
  double torque_fz_phi_i1 = 0.0;

  double torque_fx_phi_i2 = 0.0;
  double torque_fy_phi_i2 = 0.0;
  double torque_fz_phi_i2 = 0.0;


  std::vector<double> f_i1_cross_f_i2(3);
  std::vector<double> f_i1_cross_dr_norm(3);
  std::vector<double> f_i2_cross_f_i1(3);
  std::vector<double> f_i2_cross_dr_norm(3);

  std::vector<double> da3df1(3);
  std::vector<double> da3df2(3);

  std::vector<double> f_i1_cross_da3df1(3);
  std::vector<double> f_i2_cross_da3df2(3);

  for (n = 0; n < nbondlist; n++) {

    i1 = bondlist[n][0];
    i2 = bondlist[n][1];
    type = bondlist[n][2];

    iquat = bonus[ellipsoid[i1]].quat;
    MathExtra::quat_to_mat(iquat,a);

    jquat = bonus[ellipsoid[i2]].quat;
    MathExtra::quat_to_mat(jquat,b);

    //std::cout << "i1 :" << i1 << std::endl;
    //std::cout << "i2 :" << i1 << std::endl;
    //std::cout << "type :" << type << std::endl;
    //std::cout << "tag[i1] :" << tag[i1] << std::endl;
    //std::cout << "tag[i2] :" << tag[i2] << std::endl;
    //std::cout << "ellipsoid[i1] :" << ellipsoid[i1] << std::endl;
    //std::cout << "ellipsoid[tag[i1]] :" << ellipsoid[tag[i1]] << std::endl;
    //std::cout << "ellipsoid[i2] :" << ellipsoid[i2] << std::endl;
    //std::cout << "ellipsoid[tag[i2]] :" << ellipsoid[tag[i2]] << std::endl;


    double gx_i1 = a[0][0];
    double gy_i1 = a[1][0];
    double gz_i1 = a[2][0];

    double hx_i1 = a[0][1];
    double hy_i1 = a[1][1];
    double hz_i1 = a[2][1];

    double fx_i1 = a[0][2];
    double fy_i1 = a[1][2];
    double fz_i1 = a[2][2];

    double gx_i2 = b[0][0];
    double gy_i2 = b[1][0];
    double gz_i2 = b[2][0];

    double hx_i2 = b[0][1];
    double hy_i2 = b[1][1];
    double hz_i2 = b[2][1];

    double fx_i2 = b[0][2];
    double fy_i2 = b[1][2];
    double fz_i2 = b[2][2];


    // Normalize vector components of the first vector in i1
    double g_mag_i1 = sqrt(gx_i1 * gx_i1 + gy_i1 * gy_i1 + gz_i1 * gz_i1);
    double h_mag_i1 = sqrt(hx_i1 * hx_i1 + hy_i1 * hy_i1 + hz_i1 * hz_i1);
    double f_mag_i1 = sqrt(fx_i1 * fx_i1 + fy_i1 * fy_i1 + fz_i1 * fz_i1);

    gx_i1 /= g_mag_i1;
    gy_i1 /= g_mag_i1;
    gz_i1 /= g_mag_i1;

    hx_i1 /= h_mag_i1;
    hy_i1 /= h_mag_i1;
    hz_i1 /= h_mag_i1;

    fx_i1 /= f_mag_i1;
    fy_i1 /= f_mag_i1;
    fz_i1 /= f_mag_i1;

    // Normalize vector components of the second vector in i2
    double g_mag_i2 = sqrt(gx_i2 * gx_i2 + gy_i2 * gy_i2 + gz_i2 * gz_i2);
    double h_mag_i2 = sqrt(hx_i2 * hx_i2 + hy_i2 * hy_i2 + hz_i2 * hz_i2);
    double f_mag_i2 = sqrt(fx_i2 * fx_i2 + fy_i2 * fy_i2 + fz_i2 * fz_i2);

    gx_i2 /= g_mag_i2;
    gy_i2 /= g_mag_i2;
    gz_i2 /= g_mag_i2;

    hx_i2 /= h_mag_i2;
    hy_i2 /= h_mag_i2;
    hz_i2 /= h_mag_i2;

    fx_i2 /= f_mag_i2;
    fy_i2 /= f_mag_i2;
    fz_i2 /= f_mag_i2;

    delx = x[i2][0] - x[i1][0];
    dely = x[i2][1] - x[i1][1];
    delz = x[i2][2] - x[i1][2];   

    r = sqrt(delx * delx + dely * dely + delz * delz);
    delx_norm = delx/r;
    dely_norm = dely/r;
    delz_norm = delz/r;

    std::vector<double> g_i1 = {gx_i1, gy_i1, gz_i1};
    std::vector<double> h_i1 = {hx_i1, hy_i1, hz_i1};
    std::vector<double> f_i1 = {fx_i1, fy_i1, fz_i1};

    std::vector<double> g_i2 = {gx_i2, gy_i2, gz_i2};
    std::vector<double> h_i2 = {hx_i2, hy_i2, hz_i2};
    std::vector<double> f_i2 = {fx_i2, fy_i2, fz_i2};

    std::vector<double> dr = {delx , dely , delz };
    std::vector<double> dr_norm = {delx_norm , dely_norm , delz_norm };


    dx = delx - R[type]*(gx_i1 + gx_i2);
    dy = dely - R[type]*(gy_i1 + gy_i2);
    dz = delz - R[type]*(gz_i1 + gz_i2);

    std::vector<double> d = {dx, dy, dz};
    d_mag = magnitude(d);

    g_i1_cross_dr = crossProduct(g_i1,dr); 
    g_i2_cross_dr = crossProduct(g_i2,dr); 
    g_i1_cross_g_i2 = crossProduct(g_i1,g_i2);
    g_i2_cross_g_i1 = crossProduct(g_i2,g_i1);
    a0_0 =  dotProduct(g_i1, g_i2);
    if(a0_0 > 1.0){
      a0 = 1.0;
    }else if(a0_0 < -1.0){
      a0 = -1.0;
    }else{
      a0 = a0_0;
    }
    //a0 = roundToPrecision(a0_0, 5);
    alpha = acos(a0);
    
    c22 = dotProduct(dr,dr);
    c12 = dotProduct(f_i1,dr);
    c13 = dotProduct(f_i1,f_i2);
    c23 = dotProduct(dr,f_i2);
    //c22 = roundToPrecision(c22_0, 2);
    //c12 = roundToPrecision(c12_0, 2);
    //c13 = roundToPrecision(c13_0, 2);
    //c23 = roundToPrecision(c23_0, 2);
    d12 = c22 - c12*c12;
    d23 = c22 - c23*c23;

    if(d12*d23 > 0.0){
      a3_0 = (c13*c22 - c12*c23)/(sqrt(d12*d23));
      if(a3_0 > 1.0){
        a3 = 1.0;
      }else if(a3_0 < -1.0){
        a3 = -1.0;
      }else{
        a3 = a3_0;
      }
      //a3 = roundToPrecision(a3_0, 5);
      phi = acos(a3);
    }

    //std::cout << "d12 :" << d12 << std::endl;
    //std::cout << "d23 :" << d23 << std::endl;
    //std::cout << "d12*d23 :" << d12*d23 << std::endl;
    //std::cout << "c12 :" << c12 << std::endl;
    //std::cout << "c23 :" << c23 << std::endl;
    //std::cout << "a3 :" << a3 << std::endl;
    //std::cout << "alpha :" << alpha << std::endl;
    //std::cout << "f_i1[0] :" << f_i1[0] << std::endl; 
    //std::cout << "f_i1[1] :" << f_i1[1] << std::endl; 
    //std::cout << "f_i1[2] :" << f_i1[2] << std::endl; 
    //std::cout << "phi :" << phi << std::endl;
    //std::cout << "f_i2[0] :" << f_i2[0] << std::endl; 
    //std::cout << "f_i2[1] :" << f_i2[1] << std::endl; 
    //std::cout << "f_i2[2] :" << f_i2[2] << std::endl; 

    double frx_i1 = kr[type]*(r-r0[type])*delx_norm;
    double fry_i1 = kr[type]*(r-r0[type])*dely_norm;
    double frz_i1 = kr[type]*(r-r0[type])*delz_norm;

    double frx_i2 = -kr[type]*(r-r0[type])*delx_norm;
    double fry_i2 = -kr[type]*(r-r0[type])*dely_norm;
    double frz_i2 = -kr[type]*(r-r0[type])*delz_norm;

    // kR

    dUda0 = -kalpha[type];
        
    force_kR_x_i2 = -kR[type]*(delx - R[type]*(gx_i1 + gx_i2)); 
    force_kR_y_i2 = -kR[type]*(dely - R[type]*(gy_i1 + gy_i2)); 
    force_kR_z_i2 = -kR[type]*(delz - R[type]*(gz_i1 + gz_i2)); 

    force_kR_x_i1 = -force_kR_x_i2;
    force_kR_y_i1 = -force_kR_y_i2;
    force_kR_z_i1 = -force_kR_z_i2;  


    torque_gx_i1 = kR[type]*R[type]*(g_i1_cross_dr[0] -R[type]*g_i1_cross_g_i2[0]) - dUda0*g_i1_cross_g_i2[0];
    torque_gy_i1 = kR[type]*R[type]*(g_i1_cross_dr[1] -R[type]*g_i1_cross_g_i2[1]) - dUda0*g_i1_cross_g_i2[1];
    torque_gz_i1 = kR[type]*R[type]*(g_i1_cross_dr[2] -R[type]*g_i1_cross_g_i2[2]) - dUda0*g_i1_cross_g_i2[2];

    torque_gx_i2 = kR[type]*R[type]*(g_i2_cross_dr[0] -R[type]*g_i2_cross_g_i1[0]) - dUda0*g_i2_cross_g_i1[0];
    torque_gy_i2 = kR[type]*R[type]*(g_i2_cross_dr[1] -R[type]*g_i2_cross_g_i1[1]) - dUda0*g_i2_cross_g_i1[1];
    torque_gz_i2 = kR[type]*R[type]*(g_i2_cross_dr[2] -R[type]*g_i2_cross_g_i1[2]) - dUda0*g_i2_cross_g_i1[2];

    //std::cout << "torque_gy_i1 :" << torque_gy_i1 << std::endl; 
    //std::cout << "g_i1[0] :" << g_i1[0] << std::endl;   
    //std::cout << "g_i1[1] :" << g_i1[1] << std::endl;  
    //std::cout << "g_i1[2] :" << g_i1[2] << std::endl; 
    //std::cout << "torque_gy_i2 :" << torque_gy_i2 << std::endl;  
    //std::cout << "g_i2[0] :" << g_i2[0] << std::endl; 
    //std::cout << "g_i2[1] :" << g_i2[1] << std::endl; 
    //std::cout << "g_i2[2] :" << g_i2[2] << std::endl; 
    //std::cout << "dUda0 :" << dUda0 << std::endl; 
    //std::cout << "g_i1_cross_g_i2[1] :" << g_i1_cross_g_i2[1] << std::endl; 
    //std::cout << "g_i2_cross_g_i1[1] :" << g_i2_cross_g_i1[1] << std::endl; 

    if (newton_bond || i1 < nlocal) {

      tor[i1][0] +=  torque_gx_i1;
      tor[i1][1] +=  torque_gy_i1;
      tor[i1][2] +=  torque_gz_i1;
    }

    if (newton_bond || i2 < nlocal) {
          
      tor[i2][0] +=  torque_gx_i2;
      tor[i2][1] +=  torque_gy_i2;
      tor[i2][2] +=  torque_gz_i2;
    }  

    //phi

    dUda3 = -kphi[type];  

    if (d12*d23 > 0.0){
   
      da3dr_x = (((2*c13*dr[0]-c23*f_i1[0]-c12*f_i2[0])/sqrt(d12*d23)) - ((c13*c22-c12*c23)/(d12*d23))*(( (dr[0]-c12*f_i1[0])*d23 + d12*(dr[0]-c23*f_i2[0]))/sqrt(d12*d23)));
      da3dr_y = (((2*c13*dr[1]-c23*f_i1[1]-c12*f_i2[1])/sqrt(d12*d23)) - ((c13*c22-c12*c23)/(d12*d23))*(( (dr[1]-c12*f_i1[1])*d23 + d12*(dr[1]-c23*f_i2[1]))/sqrt(d12*d23)));
      da3dr_z = (((2*c13*dr[2]-c23*f_i1[2]-c12*f_i2[2])/sqrt(d12*d23)) - ((c13*c22-c12*c23)/(d12*d23))*(( (dr[2]-c12*f_i1[2])*d23 + d12*(dr[2]-c23*f_i2[2]))/sqrt(d12*d23)));

      da3df1_x = ((( c22*(f_i2[0]-c13*f_i1[0])-c23*(dr[0]-c12*f_i1[0]))/sqrt(d12*d23)) + ((c13*c22-c12*c23)/(d12*d23))*((c12*(dr[0]-c12*f_i1[0])*d23)/sqrt(d12*d23)));
      da3df1_y = ((( c22*(f_i2[1]-c13*f_i1[1])-c23*(dr[1]-c12*f_i1[1]))/sqrt(d12*d23)) + ((c13*c22-c12*c23)/(d12*d23))*((c12*(dr[1]-c12*f_i1[1])*d23)/sqrt(d12*d23)));
      da3df1_z = ((( c22*(f_i2[2]-c13*f_i1[2])-c23*(dr[2]-c12*f_i1[2]))/sqrt(d12*d23)) + ((c13*c22-c12*c23)/(d12*d23))*((c12*(dr[2]-c12*f_i1[2])*d23)/sqrt(d12*d23)));

      da3df2_x = ((( c22*(f_i1[0]-c13*f_i2[0])-c12*(dr[0]-c23*f_i2[0]))/sqrt(d12*d23)) + ((c13*c22-c12*c23)/(d12*d23))*((c23*(dr[0]-c23*f_i2[0])*d12)/sqrt(d12*d23)));
      da3df2_y = ((( c22*(f_i1[1]-c13*f_i2[1])-c12*(dr[1]-c23*f_i2[1]))/sqrt(d12*d23)) + ((c13*c22-c12*c23)/(d12*d23))*((c23*(dr[1]-c23*f_i2[1])*d12)/sqrt(d12*d23)));
      da3df2_z = ((( c22*(f_i1[2]-c13*f_i2[2])-c12*(dr[2]-c23*f_i2[2]))/sqrt(d12*d23)) + ((c13*c22-c12*c23)/(d12*d23))*((c23*(dr[2]-c23*f_i2[2])*d12)/sqrt(d12*d23)));
      
      da3df1 = {da3df1_x , da3df1_y , da3df1_z };
      da3df2 = {da3df2_x , da3df2_y , da3df2_z };

      f_i1_cross_da3df1 = crossProduct(f_i1,da3df1);
      f_i2_cross_da3df2 = crossProduct(f_i2,da3df2);

      force_kphi_x_i2 = -dUda3*da3dr_x;
      force_kphi_y_i2 = -dUda3*da3dr_y;
      force_kphi_z_i2 = -dUda3*da3dr_z;
    
      force_kphi_x_i1 = - force_kphi_x_i2;
      force_kphi_y_i1 = - force_kphi_y_i2;
      force_kphi_z_i1 = - force_kphi_z_i2;

      torque_fx_phi_i1 = -dUda3*f_i1_cross_da3df1[0];
      torque_fy_phi_i1 = -dUda3*f_i1_cross_da3df1[1];
      torque_fz_phi_i1 = -dUda3*f_i1_cross_da3df1[2];

      torque_fx_phi_i2 = -dUda3*f_i2_cross_da3df2[0];
      torque_fy_phi_i2 = -dUda3*f_i2_cross_da3df2[1];
      torque_fz_phi_i2 = -dUda3*f_i2_cross_da3df2[2];

    } 

      //std::vector<double> fc = {force_kphi_x_i2 ,force_kphi_y_i2 ,force_kphi_z_i2 }; 
      //std::vector<double> vec = crossProduct(dr,fc);
      //std::cout << "a2_i1 :" << a2_i1 << std::endl;
      //std::cout << "a2_i2 :" << a2_i2 << std::endl;
      //std::cout << "n1_dot_n2:" << n1_dot_n2 << std::endl;
      //std::cout << "n1_mag:" << n1_mag << std::endl;
      //std::cout << "n2_mag:" << n2_mag << std::endl;
      //std::cout << "phi :" << phi << std::endl;
      //std::cout << "force_kphi_x_i1 :" << force_kphi_x_i1 << std::endl;
      //std::cout << "force_kphi_x_i2 :" << force_kphi_x_i2 << std::endl;
      //std::cout << "torque_fx_phi_i1 :" << torque_fx_phi_i1 << std::endl;
      //std::cout << "torque_fx_phi_i2 :" << torque_fx_phi_i2 << std::endl;
      //std::cout << "rxF :" << vec[0] + torque_fx_phi_i1 + torque_fx_phi_i2 << std::endl;
      

    if (newton_bond || i1 < nlocal) {
      tor[i1][0] += torque_fx_phi_i1 ;
      tor[i1][1] += torque_fy_phi_i1 ;
      tor[i1][2] += torque_fz_phi_i1 ;
    }

    if (newton_bond || i2 < nlocal) {
      tor[i2][0] +=  torque_fx_phi_i2  ;
      tor[i2][1] +=  torque_fy_phi_i2  ;
      tor[i2][2] +=  torque_fz_phi_i2  ;
    }  

    // apply force to each of 2 atoms 

    if (newton_bond || i1 < nlocal) {
      forc[i1][0] +=  frx_i1 + force_kR_x_i1 + force_kphi_x_i1 ;
      forc[i1][1] +=  fry_i1 + force_kR_y_i1 + force_kphi_y_i1 ;
      forc[i1][2] +=  frz_i1 + force_kR_z_i1 + force_kphi_z_i1 ;
    }  
    if (newton_bond || i2 < nlocal) {
      forc[i2][0] +=  frx_i2 + force_kR_x_i2 + force_kphi_x_i2 ;
      forc[i2][1] +=  fry_i2 + force_kR_y_i2 + force_kphi_y_i2 ;
      forc[i2][2] +=  frz_i2 + force_kR_z_i2 + force_kphi_z_i2 ;
    }

    double fbondx = frx_i2 + force_kR_x_i2 + force_kphi_x_i2;
    double fbondy = fry_i2 + force_kR_y_i2 + force_kphi_y_i2;
    double fbondz = frz_i2 + force_kR_z_i2 + force_kphi_z_i2;
    
    //comm->forward_comm(this);
    //MPI_Barrier(MPI_COMM_WORLD);

    //int sign = ((r-r0[type]) >= 0) ? 1 : -1;
    //fbond = sqrt(fbondx*fbondx + fbondy*fbondy + fbondz*fbondz)/r;
   
    if (eflag) {
      ebond = 0.25*kr[type]*r*r + 0.25*kR[type]*d_mag*d_mag + 0.5*kalpha[type]*(1.0-cos(alpha)) + 0.5*kphi[type]*(1-cos(phi));
      //std::cout << "ebond : " << ebond << std::endl;
      ev_tally_bond_axis(i1, i2, nlocal, newton_bond, ebond, fbondx, fbondy, fbondz, delx, dely, delz);
    }

    if (evflag ) {
      ebond = 0.25*kr[type]*r*r + 0.25*kR[type]*d_mag*d_mag + 0.5*kalpha[type]*(1.0-cos(alpha)) + 0.5*kphi[type]*(1-cos(phi));
      ev_tally_bond_axis(i1, i2, nlocal, newton_bond, ebond, fbondx, fbondy, fbondz, delx, dely, delz);
    } 

  } 
}

// Function to calculate the dot product of two vectors
double BondAxis::dotProduct(const std::vector<double>& v1, const std::vector<double>& v2) {
    if (v1.size() != v2.size()) {
        throw std::invalid_argument("Vectors must be of the same size");
    }

    double result = 0.0;
    for (size_t i = 0; i < v1.size(); ++i) {
        result += v1[i] * v2[i];
    }
    return result;
}

// Function to calculate the cross product of two 3D vectors
std::vector<double> BondAxis::crossProduct(const std::vector<double>& v1, const std::vector<double>& v2) {
    if (v1.size() != 3 || v2.size() != 3) {
        throw std::invalid_argument("Vectors must be 3-dimensional");
    }

    std::vector<double> result(3);
    result[0] = v1[1] * v2[2] - v1[2] * v2[1];
    result[1] = v1[2] * v2[0] - v1[0] * v2[2];
    result[2] = v1[0] * v2[1] - v1[1] * v2[0];

    return result;
}

// Function to calculate the magnitude of a vector
double BondAxis::magnitude(const std::vector<double>& v) {
    double sum_of_squares = 0.0;
    for (double component : v) {
        sum_of_squares += component * component;
    }
    return std::sqrt(sum_of_squares);
}

/* ---------------------------------------------------------------------- */

void BondAxis::allocate()
{
  allocated = 1;
  const int np1 = atom->nbondtypes + 1;

  memory->create(kr, np1, "bond:kr");
  memory->create(kR, np1, "bond:kR");
  memory->create(kalpha, np1, "bond:kalpha");
  memory->create(kphi, np1, "bond:kphi");
  memory->create(R, np1, "bond:R");
  memory->create(r0, np1, "bond:r0");

  memory->create(setflag, np1, "bond:setflag");
  for (int i = 1; i < np1; i++) setflag[i] = 0;

  avec = dynamic_cast<AtomVecEllipsoid *>(atom->style_match("ellipsoid"));
  if (!avec) error->all(FLERR,"Pair bond_axis requires atom style ellipsoid");

}

/* ----------------------------------------------------------------------
   set coeffs for one or more types
------------------------------------------------------------------------- */

void BondAxis::coeff(int narg, char **arg)
{
 
  if (narg != 7) error->all(FLERR, "Incorrect args for bond coefficients");
  if (!allocated) allocate();

  int ilo, ihi;
  utils::bounds(FLERR, arg[0], 1, atom->nbondtypes, ilo, ihi, error);
  double k_r = utils::numeric(FLERR, arg[1], false, lmp);
  double k_R = utils::numeric(FLERR, arg[2], false, lmp);
  double k_alpha = utils::numeric(FLERR, arg[3], false, lmp);
  double k_phi = utils::numeric(FLERR, arg[4], false, lmp);
  double r_0 = utils::numeric(FLERR, arg[5], false, lmp);
  double R_0 = utils::numeric(FLERR, arg[6], false, lmp);


  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    kr[i] = k_r;
    kR[i] = k_R;
    kalpha[i] = k_alpha;
    kphi[i] = k_phi;
    R[i] = R_0;
    r0[i] = r_0;
    setflag[i] = 1;
    count++;
  }

  if (count == 0) error->all(FLERR, "Incorrect args for bond coefficients");
}


/* ----------------------------------------------------------------------
   return an equilbrium bond length
------------------------------------------------------------------------- */

double BondAxis::equilibrium_distance(int i)
{   
  return r0[i];
}

int BondAxis::pack_forward_comm(int n, int *list, double *buf, int pbc_flag, int *pbc)
{
    int i,j,m;
    double *quat;
    int **nspecial = atom->nspecial;
    tagint **special = atom->special;
    bigint natoms = atom->natoms;
    int *ellipsoid = atom->ellipsoid;
    AtomVecEllipsoid::Bonus *bonus = avec->bonus;
    int **bondlist = neighbor->bondlist;
    m = 0;
    for (i = 0; i < n; i++) {
        j = list[i];
        if (ellipsoid[j] >= 0){
            buf[m++] = ubuf(nspecial[j][0]).d;
            buf[m++] = ubuf(special[j][0]).d;
            buf[m++] = ubuf(special[j][1]).d;
            quat = bonus[ellipsoid[j]].quat;
            buf[m++] = quat[0];
            buf[m++] = quat[1];
            buf[m++] = quat[2];
            buf[m++] = quat[3];
            //buf[m++] = ubuf(ellipsoid[j]).d;
        }
    }
  return m;
}

void BondAxis::unpack_forward_comm(int n, int first, double *buf)
{
    int i,m,last;
    double *quat;
    int **nspecial = atom->nspecial;
    tagint **special = atom->special;
    int *ellipsoid = atom->ellipsoid;
    AtomVecEllipsoid::Bonus *bonus = avec->bonus;
    int **bondlist = neighbor->bondlist;
    m = 0;
    last = first + n;
    for (i = first; i < last; i++){
        if (ellipsoid[i] >=0 ){
            nspecial[i][0] = (int) ubuf((buf[m++])).i;
            special[i][0] = (int) ubuf((buf[m++])).i;
            special[i][1] = (int) ubuf((buf[m++])).i;
            quat = bonus[ellipsoid[i]].quat;
            quat[0] = buf[m++];
            quat[1] = buf[m++];
            quat[2] = buf[m++];
            quat[3] = buf[m++];
            //ellipsoid[i] = (int) ubuf((buf[m++])).i;
        }

    } 
}
/* ----------------------------------------------------------------------
   proc 0 writes out coeffs to restart file
------------------------------------------------------------------------- */

void BondAxis::write_restart(FILE *fp)
{
  fwrite(&kr[1], sizeof(double), atom->nbondtypes, fp);
  fwrite(&kR[1], sizeof(double), atom->nbondtypes, fp);
  fwrite(&kalpha[1], sizeof(double), atom->nbondtypes, fp);
  fwrite(&kphi[1], sizeof(double), atom->nbondtypes, fp);
  fwrite(&R[1], sizeof(double), atom->nbondtypes, fp);
  fwrite(&r0[1], sizeof(double), atom->nbondtypes, fp);
}

/* ----------------------------------------------------------------------
   proc 0 reads coeffs from restart file, bcasts them
------------------------------------------------------------------------- */

void BondAxis::read_restart(FILE *fp)
{
  allocate();

  if (comm->me == 0) {
    utils::sfread(FLERR, &kr[1], sizeof(double), atom->nbondtypes, fp, nullptr, error);
    utils::sfread(FLERR, &kR[1], sizeof(double), atom->nbondtypes, fp, nullptr, error);
    utils::sfread(FLERR, &kalpha[1], sizeof(double), atom->nbondtypes, fp, nullptr, error);
    utils::sfread(FLERR, &kphi[1], sizeof(double), atom->nbondtypes, fp, nullptr, error);
    utils::sfread(FLERR, &R[1], sizeof(double), atom->nbondtypes, fp, nullptr, error);
    utils::sfread(FLERR, &r0[1], sizeof(double), atom->nbondtypes, fp, nullptr, error);
  }

  MPI_Bcast(&kr[1], atom->nbondtypes, MPI_DOUBLE, 0, world);
  MPI_Bcast(&kR[1], atom->nbondtypes, MPI_DOUBLE, 0, world);
  MPI_Bcast(&kalpha[1], atom->nbondtypes, MPI_DOUBLE, 0, world);
  MPI_Bcast(&kphi[1], atom->nbondtypes, MPI_DOUBLE, 0, world);
  MPI_Bcast(&R[1], atom->nbondtypes, MPI_DOUBLE, 0, world);
  MPI_Bcast(&r0[1], atom->nbondtypes, MPI_DOUBLE, 0, world);

  for (int i = 1; i <= atom->nbondtypes; i++) setflag[i] = 1;
}

/* ----------------------------------------------------------------------
   proc 0 writes to data file
------------------------------------------------------------------------- */

void BondAxis::write_data(FILE *fp)
{
  for (int i = 1; i <= atom->nbondtypes; i++) fprintf(fp, "%d  %g %g %g %g  %g %g \n", i, kr[i], kR[i], kphi[i], kalpha[i],R[i], r0[i]);
}

/* ---------------------------------------------------------------------- */

double BondAxis::single(int type, double /*rsq*/, int i1, int i2, double &/*fforce*/)
{
  double a1[3][3], b1[3][3];
  double *iquat, *jquat;
  double **x = atom->x; 
  AtomVecEllipsoid::Bonus *bonus = avec->bonus;
  int *ellipsoid = atom->ellipsoid;
  double delx ,dely, delz, energy, alpha, a0, a3, a0_0, a3_0;
  double c11, c22, c33, c12, c13, c23, d12, d13, d23 , dUda3 , phi, f ,g;
  double d_mag, dx, dy, dz;
  double r, delx_norm, dely_norm, delz_norm;

  double small = 1e-16;

  iquat = bonus[ellipsoid[i1]].quat;
  MathExtra::quat_to_mat(iquat,a1);

  jquat = bonus[ellipsoid[i2]].quat;
  MathExtra::quat_to_mat(iquat,b1);

  double gx_i1 = a1[0][0];
  double gy_i1 = a1[1][0];
  double gz_i1 = a1[2][0];

  double hx_i1 = a1[0][1];
  double hy_i1 = a1[1][1];
  double hz_i1 = a1[2][1];

  double fx_i1 = a1[0][2];
  double fy_i1 = a1[1][2];
  double fz_i1 = a1[2][2];

  double gx_i2 = b1[0][0];
  double gy_i2 = b1[1][0];
  double gz_i2 = b1[2][0];

  double hx_i2 = b1[0][1];
  double hy_i2 = b1[1][1];
  double hz_i2 = b1[2][1];

  double fx_i2 = b1[0][2];
  double fy_i2 = b1[1][2];
  double fz_i2 = b1[2][2];


    // Normalize vector components of the first vector in i1
  double g_mag_i1 = sqrt(gx_i1 * gx_i1 + gy_i1 * gy_i1 + gz_i1 * gz_i1);
  double h_mag_i1 = sqrt(hx_i1 * hx_i1 + hy_i1 * hy_i1 + hz_i1 * hz_i1);
  double f_mag_i1 = sqrt(fx_i1 * fx_i1 + fy_i1 * fy_i1 + fz_i1 * fz_i1);

  gx_i1 /= g_mag_i1;
  gy_i1 /= g_mag_i1;
  gz_i1 /= g_mag_i1;

  hx_i1 /= h_mag_i1;
  hy_i1 /= h_mag_i1;
  hz_i1 /= h_mag_i1;

  fx_i1 /= f_mag_i1;
  fy_i1 /= f_mag_i1;
  fz_i1 /= f_mag_i1;

  // Normalize vector components of the second vector in i2
  double g_mag_i2 = sqrt(gx_i2 * gx_i2 + gy_i2 * gy_i2 + gz_i2 * gz_i2);
  double h_mag_i2 = sqrt(hx_i2 * hx_i2 + hy_i2 * hy_i2 + hz_i2 * hz_i2);
  double f_mag_i2 = sqrt(fx_i2 * fx_i2 + fy_i2 * fy_i2 + fz_i2 * fz_i2);

  gx_i2 /= g_mag_i2;
  gy_i2 /= g_mag_i2;
  gz_i2 /= g_mag_i2;

  hx_i2 /= h_mag_i2;
  hy_i2 /= h_mag_i2;
  hz_i2 /= h_mag_i2;

  fx_i2 /= f_mag_i2;
  fy_i2 /= f_mag_i2;
  fz_i2 /= f_mag_i2;

  std::vector<double> g_i1 = {gx_i1, gy_i1, gz_i1};
  std::vector<double> h_i1 = {hx_i1, hy_i1, hz_i1};
  std::vector<double> f_i1 = {fx_i1, fy_i1, fz_i1};

  std::vector<double> g_i2 = {gx_i2, gy_i2, gz_i2};
  std::vector<double> h_i2 = {hx_i2, hy_i2, hz_i2};
  std::vector<double> f_i2 = {fx_i2, fy_i2, fz_i2};

  delx = x[i2][0] - x[i1][0];
  dely = x[i2][1] - x[i1][1];
  delz = x[i2][2] - x[i1][2];

  r = sqrt(delx * delx + dely * dely + delz * delz);
  delx_norm = delx/r;
  dely_norm = dely/r;
  delz_norm = delz/r;

  domain->minimum_image(delx, dely, delz);

  dx = delx - R[type]*(gx_i1 + gx_i2);
  dy = dely - R[type]*(gy_i1 + gy_i2);
  dz = delz - R[type]*(gz_i1 + gz_i2);

  std::vector<double> d = {dx, dy, dz};
  d_mag = magnitude(d);

  std::vector<double> dr = {delx , dely , delz };
  std::vector<double> dr_norm = {delx_norm , dely_norm , delz_norm };

  a0_0 =  dotProduct(g_i1, g_i2);

  if(a0_0 > 1.0){
    a0 = 1.0;
    }else if(a0_0 < -1.0){
      a0 = -1.0;
    }else{
    a0 = a0_0;
  }
  //a0 = roundToPrecision(a0_0, 5);
  alpha = acos(a0);
    
  c22 = dotProduct(dr,dr);
  c12 = dotProduct(f_i1,dr);
  c13 = dotProduct(f_i1,f_i2);
  c23 = dotProduct(dr,f_i2);
  d12 = c22 - c12*c12;
  d23 = c22 - c23*c23;

  a3_0 = -(c13*c22 - c12*c23)/(sqrt(d12*d23));
  a3 = roundToPrecision(a3_0, 2);
  
  energy = 0.5*kr[type]*r*r + 0.5*kR[type]*d_mag*d_mag + kalpha[type]*(1.0-cos(alpha)) + kphi[type]*(1-cos(phi));

  return energy;
}

/* ---------------------------------------------------------------------- */


/* ----------------------------------------------------------------------
   return ptr to internal members upon request
------------------------------------------------------------------------ */

void *BondAxis::extract(const char *str, int &dim)
{
  dim = 1;
  if (strcmp(str, "kr") == 0) return (void *) kr;
  if (strcmp(str, "kR") == 0) return (void *) kR;
  if (strcmp(str, "kalpha") == 0) return (void *) kalpha;
  if (strcmp(str, "kphi") == 0) return (void *) kphi;
  if (strcmp(str, "R") == 0) return (void *) R;
  if (strcmp(str, "r0") == 0) return (void *) r0;
  return nullptr;
}

double BondAxis::roundToPrecision(double num, int precision) {
    double scale = std::pow(10.0, precision);
    return std::round(num * scale) / scale;
}
