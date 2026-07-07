#include "pair_electron_hopping.h"
#include "atom.h"
#include "force.h"
#include "memory.h"
#include "error.h"
#include "math_extra.h"
#include "update.h"
#include "neigh_list.h" // Include neighbor list header
#include "neighbor.h"
#include <cmath>
#include <iostream>
#include "comm.h"
#include "domain.h"
#include "atom_vec_ellipsoid.h"

using namespace LAMMPS_NS;



ElectronHopping::ElectronHopping(LAMMPS *lmp) : Pair(lmp) {
    
    kb_constant = 8.617333262145e-5;
    temp = 0.0;
    n_exp_count = 0;
    //hybrid_overlay = nullptr;
}

ElectronHopping::~ElectronHopping() {
    if (allocated) {
        memory->destroy(setflag);
        memory->destroy(force_constant_1);
        memory->destroy(force_constant_2);
        memory->destroy(force_constant_3);
        memory->destroy(cut);
        memory->destroy(cutsq);
    }
}

void ElectronHopping::allocate() {
    // Debugging: Print input values
    //int me;
    //MPI_Comm_rank(world, &me);
    allocated = 1;
    int n = atom->ntypes + 1;
    memory->create(force_constant_1, n, n, "pair:force_constant_1");
    memory->create(force_constant_2, n, n, "pair:force_constant_2");
    memory->create(force_constant_3, n, n, "pair:force_constant_3");
    memory->create(cut, n, n, "pair:cut");
    memory->create(cutsq, n, n, "pair:cutsq");
    memory->create(setflag, n, n, "pair:setflag");
    for (int i = 1; i < n; i++){
        for (int j = i; j < n; j++){
            setflag[i][j] = 0;
        }    
    }
}    
//void ElectronHopping::init_list(int, char **) {}


void ElectronHopping::settings(int narg, char **arg) {

    if (narg != 1) error->all(FLERR, "Illegal pair_style command");
    cut_global = utils::numeric(FLERR, arg[0], false, lmp);

    if (allocated) {
    int i, j;
    for (i = 1; i <= atom->ntypes; i++)
      for (j = i; j <= atom->ntypes; j++)
        if (setflag[i][j]) cut[i][j] = (cut_global);
    }

}

void ElectronHopping::coeff(int narg, char **arg) {

    if (narg != 8) error->all(FLERR, "Illegal pair_coeff command"); 
    if (!allocated) allocate();

    int ilo, ihi, jlo, jhi;
    utils::bounds(FLERR, arg[0], 1, atom->ntypes, ilo, ihi, error);
    utils::bounds(FLERR, arg[1], 1, atom->ntypes, jlo, jhi, error);

    double force_const_1 = utils::numeric(FLERR, arg[2], false, lmp);
    double force_const_2 = utils::numeric(FLERR, arg[3], false, lmp);
    double force_const_3 = utils::numeric(FLERR, arg[4], false, lmp);
    double cutoff = utils::numeric(FLERR, arg[5], false, lmp);
    double kb_constant_in = utils::numeric(FLERR, arg[6], false, lmp);
    double temp_in= utils::numeric(FLERR, arg[7], false, lmp);
    //std::cout << "kb_constant: " << kb_constant << std::endl;
    //std::cout << "temp: " << temp << std::endl;
    
    int count = 0;
    for (int i = ilo; i <= ihi; i++) {
        for (int j = MAX(jlo, i); j <= jhi; j++) {
            force_constant_1[i][j] = force_const_1;
            force_constant_2[i][j] = force_const_2;
            force_constant_3[i][j] = force_const_3;
            cutsq[i][j] = (cutoff)  * (cutoff);
            cut[i][j] = cutoff;
            setflag[i][j] = 1;
            setflag[j][j] = 1; 
            count++;
        }
    }
    kb_constant = kb_constant_in;
    temp = temp_in;
    if (count == 0) error->all(FLERR, "Incorrect args for pair coefficients");
}

double ElectronHopping::init_one(int i, int j)
{
    if (setflag[i][j] == 0) {
        cut[i][j] = mix_distance(cut[i][i], cut[j][j]);
        force_constant_1[i][j] = mix_energy(force_constant_1[i][i], force_constant_1[j][j], cut[i][i], cut[j][j]);
        force_constant_2[i][j] = mix_energy(force_constant_2[i][i], force_constant_2[j][j], cut[i][i], cut[j][j]);
        force_constant_3[i][j] = mix_energy(force_constant_3[i][i], force_constant_3[j][j], cut[i][i], cut[j][j]);
    }
    force_constant_1[j][i] = force_constant_1[i][j];
    force_constant_2[j][i] = force_constant_2[i][j];
    force_constant_3[j][i] = force_constant_3[i][j];
    cut[j][i] = cut[i][j];
    cutsq[j][i] = cutsq[i][j]; // Ensure symmetry for cutsq
    return cut[i][j];
}


void ElectronHopping::compute(int eflag, int vflag) {

    int i, j, ii, jj, inum, jnum, itype, jtype, allnum;
    double xtmp, ytmp, ztmp, fxtmp, fytmp, fztmp, force_x_pair, force_y_pair, force_z_pair;
    double delx, dely, delz, rsq, factor_lj, k_force_1, k_force_2, dot_f_left_center, dot_f_center_right, dot_f_center_other;
    int *ilist, *jlist, *numneigh, **firstneigh;
    double *iquat1, *iquat2, *iquat3;
    double m1[3][3], m2[3][3], m3[3][3];
    int atom_k_left, atom_k_center, atom_k_right, atom_k_other;
    double torque_gi_x, torque_gi_y, torque_gi_z;
    double torque_hi_x, torque_hi_y, torque_hi_z;
    double torque_fi_x, torque_fi_y, torque_fi_z;
    double torque_fi_x_left, torque_fi_y_left, torque_fi_z_left;
    double torque_fi_x_right, torque_fi_y_right, torque_fi_z_right;
    double torque_fi_x_other, torque_fi_y_other, torque_fi_z_other;
    double gx_center, gy_center, gz_center, hx_center, hy_center, hz_center, fx_center, fy_center, fz_center;
    double fx_left, fy_left, fz_left, fx_right, fy_right, fz_right;
    double g_mag_center, h_mag_center, f_mag_center, f_mag_left, f_mag_right;
    double a0,a1,a2;
    double fx_other, fy_other, fz_other;
    double f_mag_other;

    double evdwl = 0.0;
    ev_init(eflag, vflag);

    double **x = atom->x;
    double **forc = atom->f;
    int *type = atom->type;
    int nlocal = atom->nlocal;
    double *special_lj = force->special_lj;
    int newton_pair = force->newton_pair;
    double **tor = atom->torque;

    inum = list->inum;
    ilist = list->ilist;
    numneigh = list->numneigh;
    firstneigh = list->firstneigh;
    int **nspecial = atom->nspecial;
    tagint **special = atom->special;
    tagint *tag = atom->tag;

    bigint nellipsoids = atom->nellipsoids;
    bigint natoms = atom->natoms;
    AtomVecEllipsoid::Bonus *bonus = avec->bonus;
    int *ellipsoid = atom->ellipsoid;



    for (ii = 0; ii < inum; ii++){
        i = ilist[ii];
        xtmp = x[i][0];
        ytmp = x[i][1];
        ztmp = x[i][2];
        itype = type[i];
        jlist = firstneigh[i];
        jnum = numneigh[i];
        fxtmp = fytmp = fztmp = 0.0;
        double energy = 0.0;
        double fpair = 0.0;

        if (setflag[itype][itype]){
            double Z = partition_function(tag[i]); 
            //std::cout << "Z : "<< Z <<std::endl;         
            for (jj = 0; jj < jnum; jj++){
                j = jlist[jj];
                jtype = type[j];
                if (itype!=jtype){
                    factor_lj = special_lj[sbmask(j)];
                    j &= NEIGHMASK;
                    delx = xtmp - x[j][0];
                    dely = ytmp - x[j][1];
                    delz = ztmp - x[j][2];
                    rsq = delx * delx + dely * dely + delz * delz;
                    double distance = sqrt(rsq);
                    const double rcut = sqrt(cutsq[itype][jtype]);
                    double stretch_sq = (distance*distance);
                    double delx_norm = delx/distance;
                    double dely_norm = dely/distance;
                    double delz_norm = delz/distance;

                    if (stretch_sq < pow(rcut,2)){
                        int atom_id_left = special[j][0];
                        int atom_id_center = tag[j];
                        int atom_id_right = special[j][1];
                        //std::cout << "atom_id_left : "<< atom_id_left << ": atom_id_center :" << atom_id_center << ": atom_id_right :"<<atom_id_right << ": nspecial :"<< nspecial[j][0] <<std::endl;
                        //std::cout << "tag[i] : "<< tag[i] <<std::endl;
                        //std::cout << "tag[j] : "<< tag[j] <<std::endl;
                        if(nspecial[j][0]==1){

                            atom_k_other = atom->map(special[j][0]);
                            atom_k_center = atom->map(atom_id_center);

                            iquat1 = bonus[ellipsoid[atom_k_other]].quat;
                            MathExtra::quat_to_mat_trans(iquat1,m1);
                            iquat2 = bonus[ellipsoid[atom_k_center]].quat;
                            MathExtra::quat_to_mat_trans(iquat2,m2);
                                
                            gx_center = m2[0][0];
                            gy_center = m2[1][0];
                            gz_center = m2[2][0];

                            hx_center = m2[0][1];
                            hy_center = m2[1][1];
                            hz_center = m2[2][1];
                        
                            fx_other = m1[0][2];
                            fy_other = m1[1][2];
                            fz_other = m1[2][2];

                            fx_center = m2[0][2];
                            fy_center = m2[1][2];
                            fz_center = m2[2][2];

                            g_mag_center = sqrt(gx_center * gx_center + gy_center * gy_center + gz_center * gz_center);
                            h_mag_center = sqrt(hx_center * hx_center + hy_center * hy_center + hz_center * hz_center);

                            f_mag_other = sqrt(fx_other * fx_other + fy_other * fy_other +  fz_other * fz_other);
                            f_mag_center = sqrt(fx_center * fx_center + fy_center * fy_center+ fz_center * fz_center);

                            gx_center /= g_mag_center;
                            gy_center /= g_mag_center;
                            gz_center /= g_mag_center;

                            hx_center /= h_mag_center;
                            hy_center /= h_mag_center;
                            hz_center /= h_mag_center;
                    
                            fx_other /= f_mag_other;
                            fy_other /= f_mag_other;
                            fz_other /= f_mag_other;

                            fx_center /= f_mag_center;
                            fy_center /= f_mag_center;
                            fz_center /= f_mag_center;


                            dot_f_center_other = (fx_center*fx_other + fy_center*fy_other + fz_center*fz_other);

                            k_force_2 = force_constant_2[itype][jtype];
                            k_force_1 = force_constant_1[itype][jtype] - force_constant_3[itype][jtype]*(dot_f_center_other); 

                            double x_other = x[atom_k_other][0];
                            double y_other = x[atom_k_other][1];
                            double z_other = x[atom_k_other][2];

                            double x_center = x[atom_k_center][0];
                            double y_center = x[atom_k_center][1];
                            double z_center = x[atom_k_center][2];

                            double V1_x = xtmp - x_center;
                            double V1_y = ytmp - y_center;
                            double V1_z = ztmp - z_center ;

                            double V2_x = x_other - x_center;
                            double V2_y = y_other - y_center;
                            double V2_z = z_other - z_center;

                            double magnitude_V2 = sqrt(V2_x * V2_x + V2_y * V2_y + V2_z * V2_z);
                            V2_x /= magnitude_V2;
                            V2_y /= magnitude_V2;
                            V2_z /= magnitude_V2;

                            double dot_product = V1_x * V2_x + V1_y * V2_y + V1_z * V2_z;

                            a0 = delx*gx_center + dely*gy_center + delz*gz_center;
                            a1 = delx*hx_center + dely*hy_center + delz*hz_center;
                            a2 = delx*fx_center + dely*fy_center + delz*fz_center;

                            if ( dot_product <= 0.0 ){
                                k_force_1 = force_constant_2[itype][jtype];
                            } 

                            if (k_force_1 >= 0 && k_force_2 >=0){

                                energy = factor_lj*0.5*k_force_1*(a0*a0 + a1*a1) + 0.5*k_force_2*a2*a2;
                                force_x_pair = -exp(-energy/(kb_constant*temp))*(1/Z)*(k_force_1*(a0*gx_center + a1*hx_center)+k_force_2*a2*fx_center);
                                force_y_pair = -exp(-energy/(kb_constant*temp))*(1/Z)*(k_force_1*(a0*gy_center + a1*hy_center)+k_force_2*a2*fy_center);
                                force_z_pair = -exp(-energy/(kb_constant*temp))*(1/Z)*(k_force_1*(a0*gz_center + a1*hz_center)+k_force_2*a2*fz_center);

                                torque_gi_x = -exp(-energy/(kb_constant*temp))*(1/Z)*k_force_1*a0*(hy_center*delz - hz_center*dely);
                                torque_gi_y = -exp(-energy/(kb_constant*temp))*(1/Z)*k_force_1*a0*(hz_center*delx - hx_center*delz);
                                torque_gi_z = -exp(-energy/(kb_constant*temp))*(1/Z)*k_force_1*a0*(hx_center*dely - hy_center*delx);

                                torque_hi_x = -exp(-energy/(kb_constant*temp))*(1/Z)*k_force_1*a1*(hy_center*delz - hz_center*dely);
                                torque_hi_y = -exp(-energy/(kb_constant*temp))*(1/Z)*k_force_1*a1*(hz_center*delx - hx_center*delz);
                                torque_hi_z = -exp(-energy/(kb_constant*temp))*(1/Z)*k_force_1*a1*(hx_center*dely - hy_center*delx);

                                torque_fi_x = exp(-energy/(kb_constant*temp))*(1/Z)*((0.5*force_constant_3[itype][jtype]*(fy_center*fz_other - fz_center*fy_other))*(a0*a0 + a1*a1) - force_constant_2[itype][jtype]*a2*(fy_center*delz - fz_center*dely));
                                torque_fi_x = exp(-energy/(kb_constant*temp))*(1/Z)*((0.5*force_constant_3[itype][jtype]*(fz_center*fx_other - fx_center*fz_other))*(a0*a0 + a1*a1) - force_constant_2[itype][jtype]*a2*(fz_center*delx - fx_center*delz));
                                torque_fi_x = exp(-energy/(kb_constant*temp))*(1/Z)*((0.5*force_constant_3[itype][jtype]*(fx_center*fy_other - fy_center*fx_other))*(a0*a0 + a1*a1) - force_constant_2[itype][jtype]*a2*(fx_center*dely - fy_center*delx));

                                torque_fi_x_other = exp(-energy/(kb_constant*temp))*(1/Z)*(0.5*force_constant_3[itype][jtype]*(fy_other*fz_center-fz_other*fy_center)*(a0*a0 + a1*a1));
                                torque_fi_y_other = exp(-energy/(kb_constant*temp))*(1/Z)*(0.5*force_constant_3[itype][jtype]*(fz_other*fx_center-fx_other*fz_center)*(a0*a0 + a1*a1));
                                torque_fi_z_other = exp(-energy/(kb_constant*temp))*(1/Z)*(0.5*force_constant_3[itype][jtype]*(fx_other*fy_center-fy_other*fx_center)*(a0*a0 + a1*a1));

                                forc[i][0] += force_x_pair;
                                forc[i][1] += force_y_pair;
                                forc[i][2] += force_z_pair;
                                
                                
                                if (newton_pair || j < nlocal) {
                                    forc[j][0] -= force_x_pair;
                                    forc[j][1] -= force_y_pair;
                                    forc[j][2] -= force_z_pair;

                                    tor[atom_k_center][0] += torque_gi_x + torque_hi_x + torque_fi_x; 
                                    tor[atom_k_center][1] += torque_gi_y + torque_hi_y + torque_fi_y;
                                    tor[atom_k_center][2] += torque_gi_z + torque_hi_z + torque_fi_z;

                                    tor[atom_k_other][0] += torque_fi_x_other;
                                    tor[atom_k_other][1] += torque_fi_y_other;
                                    tor[atom_k_other][2] += torque_fi_z_other;
                                }
                                //if(eflag) evdwl =  (-kb_constant * temp * log(partition_function(i)));
                                if (evflag) {
                                    double n_bond_count = n_exp_count;
                                    double factor =   -2*(factor_lj/n_bond_count)*kb_constant * temp * log(Z);
                                    //if(n_bond_count>1){
                                    //std::cout << "factor_lj : " << factor_lj << std::endl;
                                    //std::cout << "n_bond_count: " << n_bond_count << std::endl;
                                    //std::cout << "energy: " << factor << std::endl;
                                    //std::cout << "Z : " << Z << std::endl;
                                    //ev_tally(i, j, nlocal, newton_pair, factor, 0.0, fpair, delx, dely, delz);
                                    ev_tally_xyz_full(i,factor,0.0,force_x_pair,force_y_pair,force_z_pair,delx, dely, delz);
                                }
                            }
                        }
                        if(nspecial[j][0]==2){

                            atom_k_left = atom->map(atom_id_left);
                            atom_k_center = atom->map(atom_id_center);
                            atom_k_right = atom->map(atom_id_right);

                            iquat1 = bonus[ellipsoid[atom_k_left]].quat;
                            MathExtra::quat_to_mat_trans(iquat1,m1);
                            iquat2 = bonus[ellipsoid[atom_k_center ]].quat;
                            MathExtra::quat_to_mat_trans(iquat2,m2);
                            iquat3 = bonus[ellipsoid[atom_k_right]].quat;
                            MathExtra::quat_to_mat_trans(iquat3,m3);

                            gx_center = m2[0][0];
                            gy_center = m2[1][0];
                            gz_center = m2[2][0];

                            hx_center = m2[0][1];
                            hy_center = m2[1][1];
                            hz_center = m2[2][1];

                            fx_left = m1[0][2];
                            fy_left = m1[1][2];
                            fz_left = m1[2][2];

                            fx_center = m2[0][2];
                            fy_center = m2[1][2];
                            fz_center = m2[2][2];

                            fx_right = m3[0][2];
                            fy_right = m3[1][2];
                            fz_right = m3[2][2];

                            g_mag_center = sqrt(gx_center * gx_center + gy_center * gy_center + gz_center * gz_center);
                            h_mag_center = sqrt(hx_center * hx_center + hy_center * hy_center + hz_center * hz_center);

                            f_mag_left = sqrt(fx_left * fx_left + fy_left * fy_left+ fz_left * fz_left);
                            f_mag_center = sqrt(fx_center * fx_center + fy_center * fy_center+ fz_center * fz_center);
                            f_mag_right = sqrt(fx_right * fx_right + fy_right * fy_right + fz_right * fz_right);

                            gx_center /= g_mag_center;
                            gy_center /= g_mag_center;
                            gz_center /= g_mag_center;

                            hx_center /= h_mag_center;
                            hy_center /= h_mag_center;
                            hz_center /= h_mag_center;
                    
                            fx_left /= f_mag_left;
                            fy_left /= f_mag_left;
                            fz_left /= f_mag_left;

                            fx_center /= f_mag_center;
                            fy_center /= f_mag_center;
                            fz_center /= f_mag_center;

                            fx_right /= f_mag_right;
                            fy_right /= f_mag_right;
                            fz_right /= f_mag_right;

                            a0 = delx*gx_center + dely*gy_center + delz*gz_center;
                            a1 = delx*hx_center + dely*hy_center + delz*hz_center;
                            a2 = delx*fx_center + dely*fy_center + delz*fz_center;

                            dot_f_left_center = (fx_left*fx_center + fy_left*fy_center + fz_left*fz_center);
                            dot_f_center_right = (fx_center*fx_right + fy_center*fy_right + fz_center*fz_right);

                            k_force_2 = force_constant_2[itype][jtype];
                            k_force_1 = force_constant_1[itype][jtype] - 0.5*force_constant_3[itype][jtype]*(dot_f_left_center + dot_f_center_right); 
                                                                                             
                            if (k_force_1 >= 0 && k_force_2 >= 0){

                                energy = factor_lj*0.5*k_force_1*(a0*a0 + a1*a1) + 0.5*k_force_2*a2*a2;
                                force_x_pair = -exp(-energy/(kb_constant*temp))*(1/Z)*(k_force_1*(a0*gx_center + a1*hx_center)+k_force_2*a2*fx_center);
                                force_y_pair = -exp(-energy/(kb_constant*temp))*(1/Z)*(k_force_1*(a0*gy_center + a1*hy_center)+k_force_2*a2*fy_center);
                                force_z_pair = -exp(-energy/(kb_constant*temp))*(1/Z)*(k_force_1*(a0*gz_center + a1*hz_center)+k_force_2*a2*fz_center);

                                torque_gi_x = -exp(-energy/(kb_constant*temp))*(1/Z)*k_force_1*a0*(hy_center*delz - hz_center*dely);
                                torque_gi_y = -exp(-energy/(kb_constant*temp))*(1/Z)*k_force_1*a0*(hz_center*delx - hx_center*delz);
                                torque_gi_z = -exp(-energy/(kb_constant*temp))*(1/Z)*k_force_1*a0*(hx_center*dely - hy_center*delx);

                                torque_hi_x = -exp(-energy/(kb_constant*temp))*(1/Z)*k_force_1*a1*(hy_center*delz - hz_center*dely);
                                torque_hi_y = -exp(-energy/(kb_constant*temp))*(1/Z)*k_force_1*a1*(hz_center*delx - hx_center*delz);
                                torque_hi_z = -exp(-energy/(kb_constant*temp))*(1/Z)*k_force_1*a1*(hx_center*dely - hy_center*delx);

                                torque_fi_x = exp(-energy/(kb_constant*temp))*(1/Z)*((0.25*force_constant_3[itype][jtype]*(fy_center*fz_left - fz_center*fy_left) + (fy_center*fz_right - fz_center*fy_right))*(a0*a0 + a1*a1) - force_constant_2[itype][jtype]*a2*(fy_center*delz - fz_center*dely));
                                torque_fi_y = exp(-energy/(kb_constant*temp))*(1/Z)*((0.25*force_constant_3[itype][jtype]*(fz_center*fx_left - fx_center*fz_left) + (fz_center*fx_right - fx_center*fz_right))*(a0*a0 + a1*a1) - force_constant_2[itype][jtype]*a2*(fz_center*delx - fx_center*delz));    
                                torque_fi_z = exp(-energy/(kb_constant*temp))*(1/Z)*((0.25*force_constant_3[itype][jtype]*(fx_center*fy_left - fy_center*fx_left) + (fx_center*fy_right - fy_center*fx_right))*(a0*a0 + a1*a1) - force_constant_2[itype][jtype]*a2*(fx_center*dely - fy_center*delx));

                                torque_fi_x_left = exp(-energy/(kb_constant*temp))*(1/Z)*(0.25*force_constant_3[itype][jtype]*(fy_left*fz_center-fz_left*fy_center)*(a0*a0 + a1*a1));
                                torque_fi_y_left = exp(-energy/(kb_constant*temp))*(1/Z)*(0.25*force_constant_3[itype][jtype]*(fz_left*fx_center-fx_left*fz_center)*(a0*a0 + a1*a1));
                                torque_fi_z_left = exp(-energy/(kb_constant*temp))*(1/Z)*(0.25*force_constant_3[itype][jtype]*(fx_left*fy_center-fy_left*fx_center)*(a0*a0 + a1*a1));

                                torque_fi_x_right = exp(-energy/(kb_constant*temp))*(1/Z)*(0.25*force_constant_3[itype][jtype]*(fy_right*fz_center-fz_right*fy_center)*(a0*a0 + a1*a1));
                                torque_fi_y_right = exp(-energy/(kb_constant*temp))*(1/Z)*(0.25*force_constant_3[itype][jtype]*(fz_right*fx_center-fx_right*fz_center)*(a0*a0 + a1*a1));
                                torque_fi_z_right = exp(-energy/(kb_constant*temp))*(1/Z)*(0.25*force_constant_3[itype][jtype]*(fx_right*fy_center-fy_right*fx_center)*(a0*a0 + a1*a1));

                                forc[i][0] += force_x_pair;
                                forc[i][1] += force_y_pair;
                                forc[i][2] += force_z_pair;

                                //std::cout << "tag[i] : "<< tag[i] <<std::endl;
                                //std::cout << "force_x_pair : "<< force_x_pair <<std::endl;
                                //std::cout << "force_y_pair : "<< force_y_pair <<std::endl;
                                //std::cout << "force_z_pair : "<< force_z_pair <<std::endl;

                                
                                if (newton_pair || j < nlocal) {
                                    forc[j][0] -= force_x_pair;
                                    forc[j][1] -= force_y_pair;
                                    forc[j][2] -= force_z_pair;

                                    tor[atom_k_center][0] += torque_gi_x + torque_hi_x + torque_fi_x; 
                                    tor[atom_k_center][1] += torque_gi_y + torque_hi_y + torque_fi_y;
                                    tor[atom_k_center][2] += torque_gi_z + torque_hi_z + torque_fi_z;

                                    tor[atom_k_left][0] += torque_fi_x_left;
                                    tor[atom_k_left][1] += torque_fi_y_left;
                                    tor[atom_k_left][2] += torque_fi_z_left;

                                    tor[atom_k_right][0] += torque_fi_x_right;
                                    tor[atom_k_right][1] += torque_fi_y_right;
                                    tor[atom_k_right][2] += torque_fi_z_right;
                                }
                                //if(eflag) evdwl =  (-kb_constant * temp * log(partition_function(i)));
                                if (evflag) {
                                    double n_bond_count = n_exp_count;
                                    double factor =   -2*(factor_lj/n_bond_count)*kb_constant * temp * log(Z);
                                    //if(n_bond_count>1){
                                    //std::cout << "factor_lj : " << factor_lj << std::endl;
                                    //std::cout << "n_bond_count: " << n_bond_count << std::endl;
                                    //std::cout << "energy: " << factor << std::endl;
                                    //std::cout << "Z : " << Z << std::endl;
                                    //ev_tally(i, j, nlocal, newton_pair, factor, 0.0, fpair, delx, dely, delz);
                                    ev_tally_xyz_full(i,factor,0.0,force_x_pair,force_y_pair,force_z_pair,delx, dely, delz);
                                }
                            }
                        }                           
                    }
                  }
            }          
        }
    }
    if (vflag_fdotr) virial_fdotr_compute();
}

void ElectronHopping::init_style()
{
    avec = dynamic_cast<AtomVecEllipsoid *>(atom->style_match("ellipsoid"));
    if (!avec) error->all(FLERR, "Pair electron_hopping requires atom style ellipsoid");
    comm_forward = 4; // Each atom sends 4 doubles
    int list_style = NeighConst::REQ_FULL;
    neighbor->add_request(this, list_style);
}


double ElectronHopping::partition_function(int i) {

    int j, ii, jj, inum, jnum, itype, jtype ;
    double xi, yi, zi;
    double delx, dely, delz, rsq, factor_lj, k_force_1, k_force_2, dot_f_left_center, dot_f_center_right, dot_f_center_other;
    int *ilist, *jlist, *numneigh, **firstneigh;
    double *iquat1, *iquat2, *iquat3;
    double m1[3][3], m2[3][3], m3[3][3];
    int atom_k_left, atom_k_center, atom_k_right, atom_k_other;
    double gx_center, gy_center, gz_center, hx_center, hy_center, hz_center, fx_center, fy_center, fz_center;
    double fx_left, fy_left, fz_left, fx_right, fy_right, fz_right;
    double g_mag_center, h_mag_center, f_mag_center, f_mag_left, f_mag_right;
    double a0,a1,a2;
    double fx_other, fy_other, fz_other;
    double f_mag_other;
    
    double **x = atom->x;
    double **forc = atom->f;
    int *type = atom->type;
    int nlocal = atom->nlocal;
    double *special_lj = force->special_lj;
    int newton_pair = force->newton_pair;
    AtomVecEllipsoid::Bonus *bonus = avec->bonus;
    int *ellipsoid = atom->ellipsoid;


    inum = list->inum;
    ilist = list->ilist;
    numneigh = list->numneigh;
    firstneigh = list->firstneigh;
    int **nspecial = atom->nspecial;
    tagint **special = atom->special;
    tagint *tag = atom->tag;

    int k = atom->map(i);
    double sum = 0.0;
    n_exp_count = 0;

    xi = x[k][0];
    yi = x[k][1];
    zi = x[k][2];
    itype = type[k]; 
    jlist = firstneigh[k];
    jnum = numneigh[k];
      
    double small = 1e-30;


    for (jj = 0; jj < jnum; jj++){
        j = jlist[jj];
        jtype = type[j];
        //std::cout << "itype :" << itype << std::endl;
        //std::cout << "jtype :" << jtype << std::endl;
        if (itype!=jtype){
            factor_lj = special_lj[sbmask(j)];
            j &= NEIGHMASK;        
            delx = xi - x[j][0];
            dely = yi - x[j][1];
            delz = zi - x[j][2];
            rsq = delx * delx + dely * dely + delz * delz;
            double distance = sqrt(rsq);
            const double rcut = sqrt(cutsq[itype][jtype]);
            double stretch_sq = (distance*distance);
            double delx_norm = delx/distance;
            double dely_norm = dely/distance;
            double delz_norm = delz/distance;

            if (stretch_sq < pow(rcut,2)){
                int atom_id_left = special[j][0];
                int atom_id_center = tag[j];
                int atom_id_right = special[j][1];
                //std::cout << "atom_id_left : "<< atom_id_left << ": atom_id_center :" << atom_id_center << ": atom_id_right :"<<atom_id_right << ": nspecial :"<< nspecial[j][0] <<std::endl;
                if(nspecial[j][0]==2) {

                    atom_k_left = atom->map(atom_id_left);
                    atom_k_center = atom->map(atom_id_center);
                    atom_k_right = atom->map(atom_id_right);

                    iquat1 = bonus[ellipsoid[atom_k_left]].quat;
                    MathExtra::quat_to_mat_trans(iquat1,m1);
                    iquat2 = bonus[ellipsoid[atom_k_center ]].quat;
                    MathExtra::quat_to_mat_trans(iquat2,m2);
                    iquat3 = bonus[ellipsoid[atom_k_right]].quat;
                    MathExtra::quat_to_mat_trans(iquat3,m3);

                    gx_center = m2[0][0];
                    gy_center = m2[1][0];
                    gz_center = m2[2][0];

                    hx_center = m2[0][1];
                    hy_center = m2[1][1];
                    hz_center = m2[2][1];

                    fx_left = m1[0][2];
                    fy_left = m1[1][2];
                    fz_left = m1[2][2];

                    fx_center = m2[0][2];
                    fy_center = m2[1][2];
                    fz_center = m2[2][2];

                    fx_right = m3[0][2];
                    fy_right = m3[1][2];
                    fz_right = m3[2][2];

                    g_mag_center = sqrt(gx_center * gx_center + gy_center * gy_center + gz_center * gz_center);
                    h_mag_center = sqrt(hx_center * hx_center + hy_center * hy_center + hz_center * hz_center);

                    f_mag_left = sqrt(fx_left * fx_left + fy_left * fy_left+ fz_left * fz_left);
                    f_mag_center = sqrt(fx_center * fx_center + fy_center * fy_center+ fz_center * fz_center);
                    f_mag_right = sqrt(fx_right * fx_right + fy_right * fy_right + fz_right * fz_right);

                    gx_center /= g_mag_center;
                    gy_center /= g_mag_center;
                    gz_center /= g_mag_center;

                    hx_center /= h_mag_center;
                    hy_center /= h_mag_center;
                    hz_center /= h_mag_center;
                    
                    fx_left /= f_mag_left;
                    fy_left /= f_mag_left;
                    fz_left /= f_mag_left;

                    fx_center /= f_mag_center;
                    fy_center /= f_mag_center;
                    fz_center /= f_mag_center;

                    fx_right /= f_mag_right;
                    fy_right /= f_mag_right;
                    fz_right /= f_mag_right;

                    a0 = delx*gx_center + dely*gy_center + delz*gz_center;
                    a1 = delx*hx_center + dely*hy_center + delz*hz_center;
                    a2 = delx*fx_center + dely*fy_center + delz*fz_center;

                    dot_f_left_center = (fx_left*fx_center + fy_left*fy_center + fz_left*fz_center);
                    dot_f_center_right = (fx_center*fx_right + fy_center*fy_right + fz_center*fz_right);

                    k_force_2 = force_constant_2[itype][jtype];
                    k_force_1 = force_constant_1[itype][jtype] - 0.5*force_constant_3[itype][jtype]*(dot_f_left_center + dot_f_center_right); 
                                                                         
                    if (k_force_1 >= 0 && k_force_2 >= 0){
                        double energy = factor_lj*0.5*k_force_1*(a0*a0 + a1*a1) + 0.5*k_force_2*a2*a2;
                        double exp_term = exp(-energy / (kb_constant * temp));
                        sum += exp_term;
                        n_exp_count +=1;
                        //std::cout << "energy: " << energy << std::endl;
                        //std::cout << "d1 : " << d1 << ": d2 : "<< d2 <<std::endl;
                        //std::cout << "sum: " << sum << std::endl;
                    }
                }   

                //std::cout << ": nspecial[j][0] :" << nspecial[j][0] << ": special[j][0] :"<< special[j][0] <<" : atom_id_location: "<< atom_id_location <<std::endl;
                if(nspecial[j][0]==1 ){ 

                    atom_k_other = atom->map(special[j][0]);
                    atom_k_center = atom->map(atom_id_center);

                    iquat1 = bonus[ellipsoid[atom_k_other]].quat;
                    MathExtra::quat_to_mat_trans(iquat1,m1);
                    iquat2 = bonus[ellipsoid[atom_k_center]].quat;
                    MathExtra::quat_to_mat_trans(iquat2,m2);

                    gx_center = m2[0][0];
                    gy_center = m2[1][0];
                    gz_center = m2[2][0];

                    hx_center = m2[0][1];
                    hy_center = m2[1][1];
                    hz_center = m2[2][1];
                        
                    fx_other = m1[0][2];
                    fy_other = m1[1][2];
                    fz_other = m1[2][2];

                    fx_center = m2[0][2];
                    fy_center = m2[1][2];
                    fz_center = m2[2][2];

                    g_mag_center = sqrt(gx_center * gx_center + gy_center * gy_center + gz_center * gz_center);
                    h_mag_center = sqrt(hx_center * hx_center + hy_center * hy_center + hz_center * hz_center);

                    f_mag_other = sqrt(fx_other * fx_other + fy_other * fy_other +  fz_other * fz_other);
                    f_mag_center = sqrt(fx_center * fx_center + fy_center * fy_center+ fz_center * fz_center);

                    gx_center /= g_mag_center;
                    gy_center /= g_mag_center;
                    gz_center /= g_mag_center;

                    hx_center /= h_mag_center;
                    hy_center /= h_mag_center;
                    hz_center /= h_mag_center;
                    
                    fx_other /= f_mag_other;
                    fy_other /= f_mag_other;
                    fz_other /= f_mag_other;

                    fx_center /= f_mag_center;
                    fy_center /= f_mag_center;
                    fz_center /= f_mag_center;


                    dot_f_center_other = (fx_center*fx_other + fy_center*fy_other + fz_center*fz_other);

                    k_force_2 = force_constant_2[itype][jtype];
                    k_force_1 = force_constant_1[itype][jtype] - force_constant_3[itype][jtype]*(dot_f_center_other); 

                    double x_other = x[atom_k_other][0];
                    double y_other = x[atom_k_other][1];
                    double z_other = x[atom_k_other][2];

                    double x_center = x[atom_k_center][0];
                    double y_center = x[atom_k_center][1];
                    double z_center = x[atom_k_center][2];

                    double V1_x = xi - x_center;
                    double V1_y = yi - y_center;
                    double V1_z = zi - z_center ;

                    double V2_x = x_other - x_center;
                    double V2_y = y_other - y_center;
                    double V2_z = z_other - z_center;

                    double magnitude_V2 = sqrt(V2_x * V2_x + V2_y * V2_y + V2_z * V2_z);
                    V2_x /= magnitude_V2;
                    V2_y /= magnitude_V2;
                    V2_z /= magnitude_V2;

                    double dot_product = V1_x * V2_x + V1_y * V2_y + V1_z * V2_z;

                    a0 = delx*gx_center + dely*gy_center + delz*gz_center;
                    a1 = delx*hx_center + dely*hy_center + delz*hz_center;
                    a2 = delx*fx_center + dely*fy_center + delz*fz_center;

                    if ( dot_product <= 0.0 ){
                        k_force_1 = force_constant_2[itype][jtype];
                    } 
                    
                    if (k_force_1 >= 0 && k_force_2 >= 0){
                        double energy = factor_lj*0.5*k_force_1*(a0*a0 + a1*a1) + 0.5*k_force_2*a2*a2;
                        double exp_term = exp(-energy / (kb_constant * temp));
                        sum += exp_term;
                        n_exp_count +=1;
                        //std::cout << "energy: " << energy << std::endl;
                        //std::cout << "d1 : " << d1 << ": d2 : "<< d2 <<": dot_product :"<< dot_product <<std::endl;
                        //std::cout << "sum: " << sum << std::endl;
                    }
                } 

            }

        }
    }
    
    if(n_exp_count==0 || sum ==0.0 ){
         n_exp_count = 1;
         sum = 1.0;
    }
    return sum;
}

int ElectronHopping::pack_forward_comm(int n, int *list, double *buf, int pbc_flag, int *pbc)
{
    int i, j, m;
    int **nspecial = atom->nspecial;
    tagint **special = atom->special;
    int *ellipsoid = atom->ellipsoid;
    m = 0;
    printf("Rank %d: pack_forward_comm n=%d, nlocal=%d, nghost=%d\n", comm->me, n, atom->nlocal, atom->nghost);
    if (n < 0) {
        error->all(FLERR, "Invalid n in pack_forward_comm");
    }
    for (i = 0; i < n; i++) {
        j = list[i];
        if (j < 0 || j >= atom->nlocal + atom->nghost) {
            printf("Rank %d: ERROR: invalid atom index j=%d\n", comm->me, j);
            error->all(FLERR, "Invalid atom index in pack_forward_comm");
        }
        buf[m++] = ubuf(nspecial[j][0]).d;
        buf[m++] = ubuf(special[j][0]).d;
        buf[m++] = ubuf(special[j][1]).d;
        buf[m++] = ubuf(ellipsoid[j]).d;
    }
    if (m != n * 4) {
        printf("Rank %d: ERROR: m=%d does not equal n*4=%d\n", comm->me, m, n * 4);
        error->all(FLERR, "Data mismatch in pack_forward_comm");
    }
    return m;
}

void ElectronHopping::unpack_forward_comm(int n, int first, double *buf)
{
    int i, m, last;
    int **nspecial = atom->nspecial;
    tagint **special = atom->special;
    int *ellipsoid = atom->ellipsoid;
    m = 0;
    last = first + n;
    printf("Rank %d: unpack_forward_comm n=%d, first=%d, nlocal=%d, nghost=%d\n", comm->me, n, first, atom->nlocal, atom->nghost);
    if (n < 0) {
        error->all(FLERR, "Invalid n in unpack_forward_comm");
    }
    for (i = first; i < last; i++) {
        if (i < 0 || i >= atom->nlocal + atom->nghost) {
            printf("Rank %d: ERROR: invalid atom index i=%d\n", comm->me, i);
            error->all(FLERR, "Invalid atom index in unpack_forward_comm");
        }
        nspecial[i][0] = (int) ubuf(buf[m++]).i;
        special[i][0] = (int) ubuf(buf[m++]).i;
        special[i][1] = (int) ubuf(buf[m++]).i;
        ellipsoid[i] = (int) ubuf(buf[m++]).i;
    }
    if (m != n * 4) {
        printf("Rank %d: ERROR: m=%d does not equal n*4=%d\n", comm->me, m, n * 4);
        error->all(FLERR, "Data mismatch in unpack_forward_comm");
    }
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void ElectronHopping::write_restart(FILE *fp)
{
  write_restart_settings(fp);

  int i, j;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      fwrite(&setflag[i][j], sizeof(int), 1, fp);
      if (setflag[i][j]) {
        fwrite(&force_constant_1[i][j], sizeof(double), 1, fp);
        fwrite(&force_constant_2[i][j], sizeof(double), 1, fp);
        fwrite(&force_constant_3[i][j], sizeof(double), 1, fp);
        fwrite(&cut[i][j], sizeof(double), 1, fp);
        fwrite(&cutsq[i][j], sizeof(double), 1, fp);
      }
    }
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void ElectronHopping::read_restart(FILE *fp)
{
  read_restart_settings(fp);
  allocate();

  int i, j;
  int me = comm->me;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      if (me == 0) utils::sfread(FLERR, &setflag[i][j], sizeof(int), 1, fp, nullptr, error);
      MPI_Bcast(&setflag[i][j], 1, MPI_INT, 0, world);
      if (setflag[i][j]) {
        if (me == 0) {
          utils::sfread(FLERR, &force_constant_1[i][j], sizeof(double), 1, fp, nullptr, error);
          utils::sfread(FLERR, &force_constant_2[i][j], sizeof(double), 1, fp, nullptr, error);
          utils::sfread(FLERR, &force_constant_3[i][j], sizeof(double), 1, fp, nullptr, error);
          utils::sfread(FLERR, &cut[i][j], sizeof(double), 1, fp, nullptr, error);
          utils::sfread(FLERR, &cutsq[i][j], sizeof(double), 1, fp, nullptr, error);
        }
        MPI_Bcast(&force_constant_1[i][j], 1, MPI_DOUBLE, 0, world);
        MPI_Bcast(&force_constant_2[i][j], 1, MPI_DOUBLE, 0, world);
        MPI_Bcast(&force_constant_3[i][j], 1, MPI_DOUBLE, 0, world);
        MPI_Bcast(&cut[i][j], 1, MPI_DOUBLE, 0, world);
        MPI_Bcast(&cutsq[i][j], 1, MPI_DOUBLE, 0, world);
      }
    }
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void ElectronHopping::write_restart_settings(FILE *fp)
{
  fwrite(&cut_global, sizeof(double), 1, fp);
}

void ElectronHopping::read_restart_settings(FILE *fp)
{
  int me = comm->me;
  if (me == 0) {
    utils::sfread(FLERR, &cut_global, sizeof(double), 1, fp, nullptr, error);
  }
  MPI_Bcast(&cut_global, 1, MPI_DOUBLE, 0, world);
}

/* ----------------------------------------------------------------------
   proc 0 writes to data file
------------------------------------------------------------------------- */

void ElectronHopping::write_data(FILE *fp)
{
  for (int i = 1; i <= atom->ntypes; i++) fprintf(fp, "%d %g %g %g\n", i, force_constant_1[i][i], force_constant_2[i][i], force_constant_3[i][i]);
}

/* ----------------------------------------------------------------------
   proc 0 writes all pairs to data file
------------------------------------------------------------------------- */

void ElectronHopping::write_data_all(FILE *fp)
{
  for (int i = 1; i <= atom->ntypes; i++)
    for (int j = i; j <= atom->ntypes; j++)
      fprintf(fp, "%d %d %g %g %g %g\n", i, j, force_constant_1[i][j], force_constant_2[i][j], force_constant_3[i][j], cut[i][j]);
}
