#include "electron_hopping.h"
#include "atom.h"
#include "force.h"
#include "math_extra.h"
#include "update.h"
#include "error.h" // Add the missing header
#include <cmath>

using namespace LAMMPS_NS;

ElectronHopping::ElectronHopping(LAMMPS *lmp) : Pair(lmp) {
    force_constant = 1.0; // Default value for the force constant
    cutoff_sq = 0.0;      // Default value for the cutoff distance
    boltzmann_constant = 1.0; // Default value for the Boltzmann constant
    nstep = 1000;         // Default value for the number of steps in the numerical integration
    temperature = 1.0;   // Default value for the temperature
}

ElectronHopping::~ElectronHopping() {}

void ElectronHopping::init_style() {
    // Set the pair style as an electron hopping potential
    if (force->newton_pair == 0)
        error->all(FLERR, "Electron hopping requires pair_modify newton yes");
}

void ElectronHopping::init_list(int, char **) {}

void ElectronHopping::settings(int narg, char **arg) {
    if (narg != 7)
        error->all(FLERR, "Illegal pair_style command");

    // Read user-defined parameters from the input script
    force_constant = atof(arg[3]);
    cutoff_sq = atof(arg[4]);
    boltzmann_constant = atof(arg[5]);
    nstep = atoi(arg[6]);
}

void ElectronHopping::compute(int eflag, int vflag) {
    int nlocal = atom->nlocal;
    double **x = atom->x;
    double **f = atom->f;
    double **v = atom->v;

    double ke = 0.0;
    for (int i = 0; i < nlocal; i++) {
        double vmag = 0.0;
        for (int k = 0; k < 3; k++) {
            vmag += v[i][k] * v[i][k];
        }
        ke += 0.5 * vmag;
    }

    temperature = 2.0 * ke / (3.0 * static_cast<double>(nlocal));

    // Loop over all pairs of atoms
    for (int i = 0; i < nlocal - 1; i++) {
        for (int j = i + 1; j < nlocal; j++) {
            double rsq = 0.0;
            for (int k = 0; k < 3; k++) {
                double del = x[i][k] - x[j][k];
                rsq += del * del;
            }

            // Check if the distance is within the cutoff
            if (rsq < cutoff_sq) {
                // Compute the force between the pair of atoms
                double force = calculate_force(i, j, sqrt(rsq));

                // Apply the force to each atom in opposite directions
                for (int k = 0; k < 3; k++) {
                    double del = x[i][k] - x[j][k];
                    f[i][k] += force * del;
                    f[j][k] -= force * del;
                }

                // Optionally, accumulate the potential energy and virial
                if (eflag)
                    ev_tally5(i, j, eng_vdwl, 0.0, virial, 0, vflag, 0.0, nullptr, 0, 0, nullptr);
                if (vflag)
                    ev_tally5(i, j, 0.0, 0.0, virial, 0, vflag, 0.0, nullptr, 0, 0, nullptr);
            }
        }
    }
}


double ElectronHopping::calculate_force(int i, int j, double distance) {
    // Calculate the partition function for the pair of atoms
    double Z = partition_function(i, j);

    // Calculate the energy of the pair of atoms at the given distance
    double energy = -boltzmann_constant * temperature * log(Z);

    // Calculate the force as the negative gradient of the energy
    double force = -force_constant * distance * exp(-energy / (boltzmann_constant * temperature));
    return force;
}

double ElectronHopping::partition_function(int i, int j) {
    // Numerical integration using trapezoidal rule
    double integral = 0.0;
    double dx = 1.0 / static_cast<double>(nstep);
    double x_prev = 0.0;

    for (int step = 0; step <= nstep; step++) {
        double x = static_cast<double>(step) * dx;
        double x_avg = 0.5 * (x_prev + x);
        double energy = 0.5 * force_constant * x_avg * x_avg;
        double exp_term = exp(-energy / (boltzmann_constant * temperature));
        integral += exp_term;

        x_prev = x;
    }

    return dx * integral;
}
