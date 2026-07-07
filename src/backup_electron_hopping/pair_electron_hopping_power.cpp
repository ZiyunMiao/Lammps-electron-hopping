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
    comm_forward = 7;
    //hybrid_overlay = nullptr;
}

ElectronHopping::~ElectronHopping() {
    if (allocated) {
        memory->destroy(setflag);
        memory->destroy(force_constant_1);
        memory->destroy(force_constant_2);
        memory->destroy(force_constant_3);
        memory->destroy(force_constant_4);
        memory->destroy(cut);
        memory->destroy(cutsq);
        memory->destroy(offset);
    }
}

void ElectronHopping::allocate() {
    allocated = 1;
    int n = atom->ntypes + 1;
    memory->create(force_constant_1, n, n, "pair:force_constant_1");
    memory->create(force_constant_2, n, n, "pair:force_constant_2");
    memory->create(force_constant_3, n, n, "pair:force_constant_3");
    memory->create(force_constant_4, n, n, "pair:force_constant_4");
    memory->create(cut, n, n, "pair:cut");
    memory->create(cutsq, n, n, "pair:cutsq");
    memory->create(setflag, n, n, "pair:setflag");
    memory->create(offset, n, n, "pair:offset");
    for (int i = 1; i < n; i++){
        for (int j = i; j < n; j++){
            setflag[i][j] = 0;
            offset[i][j] = 0;
        }
    }
}

// Parameter mapping for new model:
//   force_constant_1 = k1  : baseline intra-chain harmonic stiffness
//   force_constant_2 = alpha: inter-chain electronic coupling decay constant (inverse length)
//   force_constant_3 = rho_c: rounding length for inter-chain barrier
//   force_constant_4 = k4  : alignment-dependent stiffness reduction
//
// Pair energy: Uij = 0.5 * k_tilde1 * a0^2 + U_perp(rho)
//   where rho = sqrt(a1^2 + a2^2)
//
//   Smooth power-law inner core + linear outer tail:
//
//   U_perp = 2*alpha*kBT*(rho-rho_c)*(rho/rho_c)^(n-1)   if rho <= rho_c  (inner, n=6)
//          = 2*alpha*kBT*(rho - rho_c)                    if rho >  rho_c  (linear tail)
//
//   Properties at rho=rho_c:
//     - U_perp continuous: both sides = 0                              (C0) ✓
//     - dU/drho continuous: both sides = 2*alpha*kBT                   (C1) ✓
//   Properties at rho=0:
//     - U_perp = 0, dU/drho = 0  (charge fully free at chain axis)          ✓
//   Energy range within [0, rho_c] for n=6: ~0.064*kBT  (near-free core)   ✓
//
//   C_perp = (1/rho)*dU/drho, so grad_ri U_perp = C_perp*(a1*h_hat + a2*f_hat)
//
//   Inner (rho <= rho_c):
//     dU/drho = 2*alpha*kBT * [ (rho/rho_c)^(n-1)
//                              + (rho-rho_c)*(n-1)/rho_c*(rho/rho_c)^(n-2) ]
//     C_perp  = dU/drho / rho
//   Outer (rho > rho_c):
//     C_perp  = 2*alpha*kBT / rho

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

    if (narg != 10) error->all(FLERR, "Illegal pair_coeff command");
    if (!allocated) allocate();

    int ilo, ihi, jlo, jhi;
    utils::bounds(FLERR, arg[0], 1, atom->ntypes, ilo, ihi, error);
    utils::bounds(FLERR, arg[1], 1, atom->ntypes, jlo, jhi, error);

    double force_const_1 = utils::numeric(FLERR, arg[2], false, lmp);
    double force_const_2 = utils::numeric(FLERR, arg[3], false, lmp);
    double force_const_3 = utils::numeric(FLERR, arg[4], false, lmp);
    double force_const_4 = utils::numeric(FLERR, arg[5], false, lmp);
    double cutoff = utils::numeric(FLERR, arg[6], false, lmp);
    double kb_constant_in = utils::numeric(FLERR, arg[7], false, lmp);
    double temp_in= utils::numeric(FLERR, arg[8], false, lmp);
    double offset_in = utils::numeric(FLERR, arg[9], false, lmp);

    int count = 0;
    for (int i = ilo; i <= ihi; i++) {
        for (int j = MAX(jlo, i); j <= jhi; j++) {
            force_constant_1[i][j] = force_const_1;
            force_constant_2[i][j] = force_const_2;
            force_constant_3[i][j] = force_const_3;
            force_constant_4[i][j] = force_const_4;
            cutsq[i][j] = (cutoff)  * (cutoff);
            cut[i][j] = cutoff;
            offset[i][j] = offset_in;
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
      force_constant_4[i][j] = mix_energy(force_constant_4[i][i], force_constant_4[j][j], cut[i][i], cut[j][j]);
      offset[i][j] = 0;
  }
  force_constant_1[j][i] = force_constant_1[i][j];
  force_constant_2[j][i] = force_constant_2[i][j];
  force_constant_3[j][i] = force_constant_3[i][j];
  force_constant_4[j][i] = force_constant_4[i][j];
  offset[j][i] = offset[i][j];
  cut[j][i] = cut[i][j];
  return cut[i][j];
}

// Helper: compute rounded-linear inter-chain potential and coefficient C_perp
// rho   = sqrt(a1^2 + a2^2)
// alpha = force_constant_2, rho_c = force_constant_3
// Returns U_perp and sets C_perp such that grad_ri U_perp = C_perp*(a1*h + a2*f)
static inline void compute_interchain(double rho, double alpha, double rho_c,
                                      double kBT, double &U_perp, double &C_perp)
{
    // Power-law exponent for inner core.
    // n=6 gives energy range ~0.064*kBT within [0, rho_c]: near-free core.
    // Both U_perp and dU/drho are continuous at rho=rho_c (C1).
    const int n = 6;

    if (rho <= rho_c) {
        // Inner core: U_perp = 2*alpha*kBT*(rho - rho_c)*(rho/rho_c)^(n-1)
        //
        // Special case rho=0: both factors vanish, U_perp=0, C_perp=0.
        if (rho < 1e-14) {
            U_perp = 0.0;
            C_perp = 0.0;
            return;
        }
        double r_ratio  = rho / rho_c;                   // in (0,1]
        double r_ratio_n1 = 1.0;                          // (rho/rho_c)^(n-1)
        double r_ratio_n2 = 1.0;                          // (rho/rho_c)^(n-2)
        // Compute integer powers efficiently
        // n=6: need r_ratio^5 and r_ratio^4
        r_ratio_n2 = r_ratio*r_ratio*r_ratio*r_ratio;     // r^4
        r_ratio_n1 = r_ratio_n2 * r_ratio;                // r^5

        U_perp = 2.0 * alpha * kBT * (rho - rho_c) * r_ratio_n1;

        // dU/drho = 2*alpha*kBT * [ r_ratio^(n-1)
        //           + (rho - rho_c)*(n-1)/rho_c * r_ratio^(n-2) ]
        double dU_drho = 2.0 * alpha * kBT *
                         (r_ratio_n1
                          + (rho - rho_c) * (n - 1) / rho_c * r_ratio_n2);

        C_perp = dU_drho / rho;

    } else {
        // Outer linear tail: U_perp = 2*alpha*kBT*(rho - rho_c)
        // dU/drho = 2*alpha*kBT  →  C_perp = 2*alpha*kBT/rho
        U_perp = 2.0 * alpha * kBT * (rho - rho_c);
        C_perp = 2.0 * alpha * kBT / rho;
    }
}


void ElectronHopping::compute(int eflag, int vflag) {

    int i, j, ii, jj, inum, jnum, itype, jtype, allnum,me;
    double xtmp, ytmp, ztmp, fxtmp, fytmp, fztmp, force_x_pair, force_y_pair, force_z_pair;
    double delx, dely, delz, rsq, factor_lj, k_force_1, dot_f_left_center, dot_f_center_right, dot_f_center_other;
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

    comm->forward_comm(this);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Comm_rank(world, &me);
    double small_1 = 1e-100;
    double small_2 = 1e-300;
    double small_3 = 1e-10;

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
        double factor = 0.0;

        if (setflag[itype][itype]){
            double Z = partition_function(tag[i]);
            if(Z > small_1){
                for (jj = 0; jj < jnum; jj++){
                    j = jlist[jj];
                    jtype = type[j];
                    if (itype!=jtype && setflag[jtype][itype]){
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
                            if(nspecial[j][0]==1){

                                atom_k_other = atom->map(special[j][0]);
                                atom_k_center = atom->map(atom_id_center);

                                iquat1 = bonus[ellipsoid[atom_k_other]].quat;
                                MathExtra::quat_to_mat(iquat1,m1);
                                iquat2 = bonus[ellipsoid[atom_k_center]].quat;
                                MathExtra::quat_to_mat(iquat2,m2);

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

                                // Along-chain effective stiffness (chain-end: one neighbor)
                                k_force_1 = force_constant_1[itype][jtype] - force_constant_4[itype][jtype]*(dot_f_center_other);

                                double x_other = x[atom_k_other][0];
                                double y_other = x[atom_k_other][1];
                                double z_other = x[atom_k_other][2];

                                double x_center = x[atom_k_center][0];
                                double y_center = x[atom_k_center][1];
                                double z_center = x[atom_k_center][2];

                                double V1_x = xtmp - x_center;
                                double V1_y = ytmp - y_center;
                                double V1_z = ztmp - z_center;

                                double V3_x = x[atom_k_other][0] - x[atom_k_center][0];
                                double V3_y = x[atom_k_other][1] - x[atom_k_center][1];
                                double V3_z = x[atom_k_other][2] - x[atom_k_center][2];

                                double V2_x = x_other - x_center;
                                double V2_y = y_other - y_center;
                                double V2_z = z_other - z_center;

                                double v2[3] = {V2_x, V2_y, V2_z};
                                domain->minimum_image(v2);
                                V2_x = v2[0];
                                V2_y = v2[1];
                                V2_z = v2[2];

                                double magnitude_V2 = sqrt(V2_x * V2_x + V2_y * V2_y + V2_z * V2_z);
                                V2_x /= magnitude_V2;
                                V2_y /= magnitude_V2;
                                V2_z /= magnitude_V2;

                                double magnitude_V3 = sqrt(V3_x * V3_x + V3_y * V3_y + V3_z * V3_z);
                                V3_x /= magnitude_V3;
                                V3_y /= magnitude_V3;
                                V3_z /= magnitude_V3;

                                a0 = delx*gx_center + dely*gy_center + delz*gz_center;
                                a1 = delx*hx_center + dely*hy_center + delz*hz_center;
                                a2 = delx*fx_center + dely*fy_center + delz*fz_center;

                                double dot_product_2 = V2_x * gx_center + V2_y * gy_center + V2_z * gz_center;
                                double dot_product = a0 * dot_product_2;

                                double modulation = force_constant_4[itype][jtype];

                                if ( dot_product <= 0 ){
                                    k_force_1 = force_constant_1[itype][jtype];
                                    modulation = 0.0;
                                }

                                // --- New inter-chain rounded-linear term ---
                                // alpha = force_constant_2, rho_c = force_constant_3
                                double rho_ij = sqrt(a1*a1 + a2*a2);
                                double alpha_val = force_constant_2[itype][jtype];
                                double rho_c = force_constant_3[itype][jtype];
                                double U_perp, C_perp;
                                compute_interchain(rho_ij, alpha_val, rho_c, kb_constant*temp, U_perp, C_perp);

                                energy = 0.5*k_force_1*a0*a0 + U_perp + offset[itype][jtype];

                                double pij = exp(-energy/(kb_constant*temp)) * (1.0/Z);

                                force_x_pair = -pij*(k_force_1*a0*gx_center + C_perp*a1*hx_center + C_perp*a2*fx_center);
                                force_y_pair = -pij*(k_force_1*a0*gy_center + C_perp*a1*hy_center + C_perp*a2*fy_center);
                                force_z_pair = -pij*(k_force_1*a0*gz_center + C_perp*a1*hz_center + C_perp*a2*fz_center);

                                if(fabs(force_x_pair) < small_2){force_x_pair = 0.0;}
                                if(fabs(force_y_pair) < small_2){force_y_pair = 0.0;}
                                if(fabs(force_z_pair) < small_2){force_z_pair = 0.0;}

                                // Torque from along-chain harmonic (g direction)
                                torque_gi_x = pij*( -k_force_1*(a0*(gy_center*delz - gz_center*dely)) );
                                torque_gi_y = pij*( -k_force_1*(a0*(gz_center*delx - gx_center*delz)) );
                                torque_gi_z = pij*( -k_force_1*(a0*(gx_center*dely - gy_center*delx)) );
                                if(fabs(torque_gi_x) < small_2){torque_gi_x = 0.0;}
                                if(fabs(torque_gi_y) < small_2){torque_gi_y = 0.0;}
                                if(fabs(torque_gi_z) < small_2){torque_gi_z = 0.0;}

                                // Torque from transverse rounded-linear term (h direction)
                                torque_hi_x = pij*( -C_perp*(a1*(hy_center*delz - hz_center*dely)) );
                                torque_hi_y = pij*( -C_perp*(a1*(hz_center*delx - hx_center*delz)) );
                                torque_hi_z = pij*( -C_perp*(a1*(hx_center*dely - hy_center*delx)) );
                                if(fabs(torque_hi_x) < small_2){torque_hi_x = 0.0;}
                                if(fabs(torque_hi_y) < small_2){torque_hi_y = 0.0;}
                                if(fabs(torque_hi_z) < small_2){torque_hi_z = 0.0;}

                                // Torque from f direction: transverse part + alignment torque from k_tilde1
                                torque_fi_x = pij*(0.5*modulation*((fy_center*fz_other - fz_center*fy_other))*(a0*a0) - C_perp*a2*(fy_center*delz - fz_center*dely));
                                torque_fi_y = pij*(0.5*modulation*((fz_center*fx_other - fx_center*fz_other))*(a0*a0) - C_perp*a2*(fz_center*delx - fx_center*delz));
                                torque_fi_z = pij*(0.5*modulation*((fx_center*fy_other - fy_center*fx_other))*(a0*a0) - C_perp*a2*(fx_center*dely - fy_center*delx));
                                if(fabs(torque_fi_x) < small_2){torque_fi_x = 0.0;}
                                if(fabs(torque_fi_y) < small_2){torque_fi_y = 0.0;}
                                if(fabs(torque_fi_z) < small_2){torque_fi_z = 0.0;}

                                // Alignment torque on neighboring monomer (other)
                                torque_fi_x_other = pij*(0.5*modulation*(fy_other*fz_center-fz_other*fy_center)*(a0*a0));
                                torque_fi_y_other = pij*(0.5*modulation*(fz_other*fx_center-fx_other*fz_center)*(a0*a0));
                                torque_fi_z_other = pij*(0.5*modulation*(fx_other*fy_center-fy_other*fx_center)*(a0*a0));
                                if(fabs(torque_fi_x_other) < small_2){torque_fi_x_other = 0.0;}
                                if(fabs(torque_fi_y_other) < small_2){torque_fi_y_other = 0.0;}
                                if(fabs(torque_fi_z_other) < small_2){torque_fi_z_other = 0.0;}

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
                                if (evflag) {
                                    double n_bond_count = n_exp_count;
                                    if(n_bond_count > 0){
                                        factor =   -2*(factor_lj/n_bond_count)*kb_constant * temp * log(Z);
                                    }else{
                                        factor =  0.0;
                                    }
                                    ev_tally_xyz_full(i,factor,0.0,force_x_pair,force_y_pair,force_z_pair,delx, dely, delz);
                                }

                            }
                            if(nspecial[j][0]==2){

                                atom_k_left = atom->map(atom_id_left);
                                atom_k_center = atom->map(atom_id_center);
                                atom_k_right = atom->map(atom_id_right);

                                iquat1 = bonus[ellipsoid[atom_k_left]].quat;
                                MathExtra::quat_to_mat(iquat1,m1);
                                iquat2 = bonus[ellipsoid[atom_k_center ]].quat;
                                MathExtra::quat_to_mat(iquat2,m2);
                                iquat3 = bonus[ellipsoid[atom_k_right]].quat;
                                MathExtra::quat_to_mat(iquat3,m3);

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

                                // Along-chain effective stiffness (interior monomer: two neighbors)
                                k_force_1 = force_constant_1[itype][jtype] - 0.5*force_constant_4[itype][jtype]*(dot_f_left_center + dot_f_center_right);

                                // --- New inter-chain rounded-linear term ---
                                double rho_ij = sqrt(a1*a1 + a2*a2);
                                double alpha_val = force_constant_2[itype][jtype];
                                double rho_c = force_constant_3[itype][jtype];
                                double U_perp, C_perp;
                                compute_interchain(rho_ij, alpha_val, rho_c, kb_constant*temp, U_perp, C_perp);

                                energy = 0.5*k_force_1*a0*a0 + U_perp + offset[itype][jtype];

                                double pij = exp(-energy/(kb_constant*temp)) * (1.0/Z);

                                force_x_pair = -pij*(k_force_1*a0*gx_center + C_perp*a1*hx_center + C_perp*a2*fx_center);
                                force_y_pair = -pij*(k_force_1*a0*gy_center + C_perp*a1*hy_center + C_perp*a2*fy_center);
                                force_z_pair = -pij*(k_force_1*a0*gz_center + C_perp*a1*hz_center + C_perp*a2*fz_center);
                                if(fabs(force_x_pair) < small_2){force_x_pair = 0.0;}
                                if(fabs(force_y_pair) < small_2){force_y_pair = 0.0;}
                                if(fabs(force_z_pair) < small_2){force_z_pair = 0.0;}

                                // Torque from along-chain harmonic (g direction)
                                torque_gi_x = pij*( -k_force_1*(a0*(gy_center*delz - gz_center*dely)) );
                                torque_gi_y = pij*( -k_force_1*(a0*(gz_center*delx - gx_center*delz)) );
                                torque_gi_z = pij*( -k_force_1*(a0*(gx_center*dely - gy_center*delx)) );
                                if(fabs(torque_gi_x) < small_2){torque_gi_x = 0.0;}
                                if(fabs(torque_gi_y) < small_2){torque_gi_y = 0.0;}
                                if(fabs(torque_gi_z) < small_2){torque_gi_z = 0.0;}

                                // Torque from transverse rounded-linear term (h direction)
                                torque_hi_x = pij*( -C_perp*(a1*(hy_center*delz - hz_center*dely)) );
                                torque_hi_y = pij*( -C_perp*(a1*(hz_center*delx - hx_center*delz)) );
                                torque_hi_z = pij*( -C_perp*(a1*(hx_center*dely - hy_center*delx)) );
                                if(fabs(torque_hi_x) < small_2){torque_hi_x = 0.0;}
                                if(fabs(torque_hi_y) < small_2){torque_hi_y = 0.0;}
                                if(fabs(torque_hi_z) < small_2){torque_hi_z = 0.0;}

                                // Torque on f of center: transverse part + alignment torques from k_tilde1
                                torque_fi_x = pij*(0.25*force_constant_4[itype][jtype]*((fy_center*fz_left - fz_center*fy_left) + (fy_center*fz_right - fz_center*fy_right))*(a0*a0) - C_perp*a2*(fy_center*delz - fz_center*dely));
                                torque_fi_y = pij*(0.25*force_constant_4[itype][jtype]*((fz_center*fx_left - fx_center*fz_left) + (fz_center*fx_right - fx_center*fz_right))*(a0*a0) - C_perp*a2*(fz_center*delx - fx_center*delz));
                                torque_fi_z = pij*(0.25*force_constant_4[itype][jtype]*((fx_center*fy_left - fy_center*fx_left) + (fx_center*fy_right - fy_center*fx_right))*(a0*a0) - C_perp*a2*(fx_center*dely - fy_center*delx));
                                if(fabs(torque_fi_x) < small_2){torque_fi_x = 0.0;}
                                if(fabs(torque_fi_y) < small_2){torque_fi_y = 0.0;}
                                if(fabs(torque_fi_z) < small_2){torque_fi_z = 0.0;}

                                // Alignment torques on left/right neighbors
                                torque_fi_x_left = pij*(0.25*force_constant_4[itype][jtype]*(fy_left*fz_center-fz_left*fy_center)*(a0*a0));
                                torque_fi_y_left = pij*(0.25*force_constant_4[itype][jtype]*(fz_left*fx_center-fx_left*fz_center)*(a0*a0));
                                torque_fi_z_left = pij*(0.25*force_constant_4[itype][jtype]*(fx_left*fy_center-fy_left*fx_center)*(a0*a0));
                                if(fabs(torque_fi_x_left)  < small_2){torque_fi_x_left  = 0.0;}
                                if(fabs(torque_fi_y_left)  < small_2){torque_fi_y_left  = 0.0;}
                                if(fabs(torque_fi_z_left)  < small_2){torque_fi_z_left  = 0.0;}

                                torque_fi_x_right = pij*(0.25*force_constant_4[itype][jtype]*(fy_right*fz_center-fz_right*fy_center)*(a0*a0));
                                torque_fi_y_right = pij*(0.25*force_constant_4[itype][jtype]*(fz_right*fx_center-fx_right*fz_center)*(a0*a0));
                                torque_fi_z_right = pij*(0.25*force_constant_4[itype][jtype]*(fx_right*fy_center-fy_right*fx_center)*(a0*a0));
                                if(fabs(torque_fi_x_right)   < small_2){torque_fi_x_right   = 0.0;}
                                if(fabs(torque_fi_y_right)   < small_2){torque_fi_y_right   = 0.0;}
                                if(fabs(torque_fi_z_right)   < small_2){torque_fi_z_right   = 0.0;}


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

                                    tor[atom_k_left][0] += torque_fi_x_left;
                                    tor[atom_k_left][1] += torque_fi_y_left;
                                    tor[atom_k_left][2] += torque_fi_z_left;

                                    tor[atom_k_right][0] += torque_fi_x_right;
                                    tor[atom_k_right][1] += torque_fi_y_right;
                                    tor[atom_k_right][2] += torque_fi_z_right;
                                }
                                if (evflag) {
                                    double n_bond_count = n_exp_count;
                                    if(n_bond_count > 0){
                                        factor =   -2*(factor_lj/n_bond_count)*kb_constant * temp * log(Z);
                                    }else{
                                        factor =   0.0;
                                    }
                                    ev_tally_xyz_full(i,factor,0.0,force_x_pair,force_y_pair,force_z_pair,delx, dely, delz);
                                }
                            }
                        }
                    }else if (evflag){
                        factor =  0.0;
                        ev_tally_xyz_full(i,factor,0.0,0.0,0.0,0.0,delx, dely, delz);
                    }
                }
            }else if (evflag){
                factor =  0.0;
                ev_tally_xyz_full(i,factor,0.0,0.0,0.0,0.0,0.0, 0.0, 0.0);
            }
        }else if (evflag){
                factor =  0.0;
                ev_tally_xyz_full(i,factor,0.0,0.0,0.0,0.0,0.0, 0.0, 0.0);
        }
    }
    if (vflag_fdotr) virial_fdotr_compute();
}

void ElectronHopping::init_style()
{
  avec = dynamic_cast<AtomVecEllipsoid *>(atom->style_match("ellipsoid"));
  if (!avec) error->all(FLERR,"Pair electron_hopping requires atom style ellipsoid");

  int list_style = NeighConst::REQ_FULL;
  neighbor->add_request(this, list_style);

}


double ElectronHopping::partition_function(int i) {

    int j, ii, jj, inum, jnum, itype, jtype ;
    double xi, yi, zi;
    double delx, dely, delz, rsq, factor_lj, k_force_1, dot_f_left_center, dot_f_center_right, dot_f_center_other;
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

    bigint nellipsoids = atom->nellipsoids;
    bigint natoms = atom->natoms;

    double small_1 = 1e-300;
    double small_2 = 1e-300;
    double small_3 = 1e-10;

    if (setflag[itype][itype]){
        for (jj = 0; jj < jnum; jj++){
            j = jlist[jj];
            jtype = type[j];
            if (itype!=jtype && setflag[jtype][itype]){
                factor_lj = special_lj[sbmask(j)];
                j &= NEIGHMASK;
                delx = xi - x[j][0];
                dely = yi - x[j][1];
                delz = zi - x[j][2];
                rsq = delx * delx + dely * dely + delz * delz;
                double distance = sqrt(rsq);
                const double rcut = sqrt(cutsq[itype][jtype]);
                double stretch_sq = (distance*distance);

                if (stretch_sq < pow(rcut,2)){
                    int atom_id_left = special[j][0];
                    int atom_id_center = tag[j];
                    int atom_id_right = special[j][1];
                    if(nspecial[j][0]==2) {
                        atom_k_left = atom->map(atom_id_left);
                        atom_k_center = atom->map(atom_id_center);
                        atom_k_right = atom->map(atom_id_right);

                        iquat1 = bonus[ellipsoid[atom_k_left]].quat;
                        MathExtra::quat_to_mat(iquat1,m1);
                        iquat2 = bonus[ellipsoid[atom_k_center ]].quat;
                        MathExtra::quat_to_mat(iquat2,m2);
                        iquat3 = bonus[ellipsoid[atom_k_right]].quat;
                        MathExtra::quat_to_mat(iquat3,m3);

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

                        k_force_1 = force_constant_1[itype][jtype] - 0.5*force_constant_4[itype][jtype]*(dot_f_left_center + dot_f_center_right);

                        // Inter-chain rounded-linear contribution
                        double rho_ij = sqrt(a1*a1 + a2*a2);
                        double alpha_val = force_constant_2[itype][jtype];
                        double rho_c = force_constant_3[itype][jtype];
                        double U_perp, C_perp;
                        compute_interchain(rho_ij, alpha_val, rho_c, kb_constant*temp, U_perp, C_perp);

                        double energy = 0.5*k_force_1*a0*a0 + U_perp + offset[itype][jtype];
                        double exp_term = exp(-energy / (kb_constant * temp));
                        sum += exp_term;
                        n_exp_count +=1;

                    }

                    if(nspecial[j][0]==1 ){
                        atom_k_other = atom->map(special[j][0]);
                        atom_k_center = atom->map(atom_id_center);

                        iquat1 = bonus[ellipsoid[atom_k_other]].quat;
                        MathExtra::quat_to_mat(iquat1,m1);
                        iquat2 = bonus[ellipsoid[atom_k_center]].quat;
                        MathExtra::quat_to_mat(iquat2,m2);

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

                        k_force_1 = force_constant_1[itype][jtype] - force_constant_4[itype][jtype]*(dot_f_center_other);

                        double x_other = x[atom_k_other][0];
                        double y_other = x[atom_k_other][1];
                        double z_other = x[atom_k_other][2];

                        double x_center = x[atom_k_center][0];
                        double y_center = x[atom_k_center][1];
                        double z_center = x[atom_k_center][2];

                        double V1_x = xi - x_center;
                        double V1_y = yi - y_center;
                        double V1_z = zi - z_center;

                        double V2_x = x_other - x_center;
                        double V2_y = y_other - y_center;
                        double V2_z = z_other - z_center;

                        double V3_x = x[atom_k_other][0] - x[atom_k_center][0];
                        double V3_y = x[atom_k_other][1] - x[atom_k_center][1];
                        double V3_z = x[atom_k_other][2] - x[atom_k_center][2];

                        double v2[3] = {V2_x, V2_y, V2_z};
                        domain->minimum_image(v2);
                        V2_x = v2[0];
                        V2_y = v2[1];
                        V2_z = v2[2];

                        double magnitude_V2 = sqrt(V2_x * V2_x + V2_y * V2_y + V2_z * V2_z);
                        V2_x /= magnitude_V2;
                        V2_y /= magnitude_V2;
                        V2_z /= magnitude_V2;

                        double magnitude_V3 = sqrt(V3_x * V3_x + V3_y * V3_y + V3_z * V3_z);
                        V3_x /= magnitude_V3;
                        V3_y /= magnitude_V3;
                        V3_z /= magnitude_V3;

                        a0 = delx*gx_center + dely*gy_center + delz*gz_center;
                        a1 = delx*hx_center + dely*hy_center + delz*hz_center;
                        a2 = delx*fx_center + dely*fy_center + delz*fz_center;

                        double dot_product_2 = V2_x * gx_center + V2_y * gy_center + V2_z * gz_center;
                        double dot_product = a0 * dot_product_2;

                        if ( dot_product <= 0 ){
                            k_force_1 = force_constant_1[itype][jtype];
                        }

                        // Inter-chain rounded-linear contribution
                        double rho_ij = sqrt(a1*a1 + a2*a2);
                        double alpha_val = force_constant_2[itype][jtype];
                        double rho_c = force_constant_3[itype][jtype];
                        double U_perp, C_perp;
                        compute_interchain(rho_ij, alpha_val, rho_c, kb_constant*temp, U_perp, C_perp);

                        double energy = 0.5*k_force_1*a0*a0 + U_perp + offset[itype][jtype];
                        double exp_term = exp(-energy / (kb_constant * temp));
                        sum += exp_term;
                        n_exp_count +=1;

                    }
                }

            }
        }
    }
    return sum;
}

int ElectronHopping::pack_forward_comm(int n, int *list, double *buf, int pbc_flag, int *pbc)
{
    int i,j,m,me;
    double *quat;
    int **nspecial = atom->nspecial;
    tagint **special = atom->special;
    bigint natoms = atom->natoms;
    int *ellipsoid = atom->ellipsoid;
    AtomVecEllipsoid::Bonus *bonus = avec->bonus;
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
        }
    }
  return m;
}

void ElectronHopping::unpack_forward_comm(int n, int first, double *buf)
{
    int i,m,last,me;
    double *quat;
    int **nspecial = atom->nspecial;
    tagint **special = atom->special;
    int *ellipsoid = atom->ellipsoid;
    AtomVecEllipsoid::Bonus *bonus = avec->bonus;
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
        }

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
        fwrite(&force_constant_4[i][j], sizeof(double), 1, fp);
        fwrite(&offset[i][j], sizeof(double), 1, fp);
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
          utils::sfread(FLERR, &force_constant_4[i][j], sizeof(double), 1, fp, nullptr, error);
          utils::sfread(FLERR, &offset[i][j], sizeof(double), 1, fp, nullptr, error);
          utils::sfread(FLERR, &cut[i][j], sizeof(double), 1, fp, nullptr, error);
          utils::sfread(FLERR, &cutsq[i][j], sizeof(double), 1, fp, nullptr, error);
        }
        MPI_Bcast(&force_constant_1[i][j], 1, MPI_DOUBLE, 0, world);
        MPI_Bcast(&force_constant_2[i][j], 1, MPI_DOUBLE, 0, world);
        MPI_Bcast(&force_constant_3[i][j], 1, MPI_DOUBLE, 0, world);
        MPI_Bcast(&force_constant_4[i][j], 1, MPI_DOUBLE, 0, world);
        MPI_Bcast(&offset[i][j], 1, MPI_DOUBLE, 0, world);
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
  for (int i = 1; i <= atom->ntypes; i++) fprintf(fp, "%d %g %g %g %g %g\n", i, force_constant_1[i][i], force_constant_2[i][i], force_constant_3[i][i],force_constant_4[i][i], offset[i][i]);
}

/* ----------------------------------------------------------------------
   proc 0 writes all pairs to data file
------------------------------------------------------------------------- */

void ElectronHopping::write_data_all(FILE *fp)
{
  for (int i = 1; i <= atom->ntypes; i++)
    for (int j = i; j <= atom->ntypes; j++)
      fprintf(fp, "%d %d %g %g %g %g %g %g\n", i, j, force_constant_1[i][j], force_constant_2[i][j], force_constant_3[i][j],force_constant_4[i][j], offset[i][j], cut[i][j]);
}