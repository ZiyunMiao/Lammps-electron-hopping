#include "pair_electron_hopping.h"
#include "atom.h"
#include "force.h"
#include "math_extra.h"
#include "update.h"
#include "error.h"
#include <cmath>
#include <iostream>

using namespace LAMMPS_NS;

ElectronHopping::ElectronHopping(LAMMPS *lmp) : Pair(lmp) {
    force_constant = 1.0;         // Default value for the force constant
    equilibrium_distance = 1.0;   // Default value for the equilibrium distance
    cutoff_sq = 0.0;              // Default value for the cutoff distance
    boltzmann_constant = 8.617333262145e-5;  // Boltzmann constant in eV/K
    nstep = 1000;                 // Default value for the number of steps in the numerical integration
    temperature = 300.0;         // Default value for the temperature in Kelvin
    hybrid_overlay = nullptr;    // Initialize the pointer to PairHybridOverlay as nullptr
}

ElectronHopping::~ElectronHopping() {}

void ElectronHopping::init_style() {
    // Set the pair style as an electron hopping potential
    if (force->newton_pair == 0)
        error->all(FLERR, "Electron hopping requires pair_modify newton yes");
}

void ElectronHopping::init_list(int, char **) {}

void ElectronHopping::settings(int narg, char **arg) {
    //std::cout << "Number of arguments: " << narg << std::endl;
    //std::cout << "Coeff arguments: ";
    //for (int i = 0; i < narg; i++) {
    //    std::cout << arg[i] << " ";
    //}
    //std::cout << std::endl;
    if (narg != 1)
        error->all(FLERR, "Illegal pair_style command");
    cutoff_sq = atof(arg[0]);
    // Calculate temperature from kinetic energy
    temperature = compute_temperature();
}

double ElectronHopping::compute_temperature() {
    double kinetic_energy = 0.0;
    int nlocal = atom->nlocal;
    double **v = atom->v;

    for (int i = 0; i < nlocal; i++) {
        double mass = atom->mass[atom->type[i]];
        for (int k = 0; k < 3; k++) {
            kinetic_energy += 0.5 * mass * v[i][k] * v[i][k];
        }
    }

    // The total kinetic energy is summed across all processors
    MPI_Allreduce(MPI_IN_PLACE, &kinetic_energy, 1, MPI_DOUBLE, MPI_SUM, world);

    // Compute the temperature from kinetic energy
    double num_dofs = 3 * nlocal;
    double kb_temperature = 2.0 * kinetic_energy / (num_dofs * boltzmann_constant);

    return kb_temperature;
}

void ElectronHopping::compute(int eflag, int vflag) {
    int nlocal = atom->nlocal;
    double **x = atom->x;
    double **f = atom->f;
    int newton_pair = force->newton_pair;

    // Optionally, accumulate the potential energy and virial
    if (eflag) {
        double eng_vdwl = 0.0;
        double virial[6] = {0.0}; // Initialize virial tensor to zeros

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

                    // Optionally, accumulate the potential energy and virial
                    eng_vdwl += 0.5 * force * sqrt(rsq);

                    // Apply the force to each atom in opposite directions
                    for (int k = 0; k < 3; k++) {
                        double del = x[i][k] - x[j][k];
                        f[i][k] += force * del;
                        f[j][k] -= force * del;

                        // Optionally, accumulate the virial tensor
                        virial[k] += del * del * force;
                        virial[k + 3] -= del * del * force;
                    }
                }
            }
        }

        // Accumulate the energy and virial
        eng_vdwl *= 2.0; // Double the energy since each pair is counted twice
        ev_tally(1, 0, nlocal, newton_pair, eng_vdwl, 0.0, virial[0], virial[1], virial[2], virial[3]);
    }

    if (vflag) {
        double virial[6] = {0.0}; // Initialize virial tensor to zeros

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

                        // Optionally, accumulate the virial tensor
                        virial[k] += del * del * force;
                        virial[k + 3] -= del * del * force;
                    }
                }
            }
        }

        // Accumulate the virial
        ev_tally(0, 0, nlocal, newton_pair, 0.0, 0.0, virial[0], virial[1], virial[2], virial[3]);
    }
}

void ElectronHopping::coeff(int narg, char **arg) {
    //std::cout << "Number of arguments: " << narg << std::endl;
    //std::cout << "Coeff arguments: ";
    //for (int i = 0; i < narg; i++) {
    //    std::cout << arg[i] << " ";
    //}
    //std::cout << std::endl;

    if (narg != 7)
        error->all(FLERR, "Illegal pair_coeff command");

    int ilo, ihi, jlo, jhi;

    // Set the parameters for ilo, ihi, jlo, and jhi based on user input
    utils::bounds(FLERR, arg[0], 1, atom->ntypes, ilo, ihi, error);
    utils::bounds(FLERR, arg[1], 1, atom->ntypes, jlo, jhi, error);


    // Set the parameters for the ElectronHopping interactions
    force_constant = atof(arg[2]);
    equilibrium_distance = atof(arg[3]); // Read the equilibrium distance from the input
    cutoff_sq = atof(arg[4]); // Assuming the cutoff_sq is the same as the force_constant for electron_hopping
    boltzmann_constant = atof(arg[5]);   // Read the boltzmann constant from the input
    nstep = atoi(arg[6]);                // Read the nstep from the input

    // Additional processing of coefficients if required.
    // You can set the force_constant, equilibrium_distance, boltzmann_constant, and nstep class variables here.
}

double ElectronHopping::calculate_force(int i, int j, double distance) {
    // Calculate the partition function for the pair of atoms
    double Z = partition_function(i, j);

    // Calculate the energy of the pair of atoms at the given distance
    double energy = -boltzmann_constant * temperature * log(Z);

    // Calculate the force as the negative gradient of the energy
    double force = -force_constant * (distance - equilibrium_distance) * exp(-energy / (boltzmann_constant * temperature));
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
        double energy = 0.5 * force_constant * (x_avg - equilibrium_distance) * (x_avg - equilibrium_distance);
        double exp_term = exp(-energy / (boltzmann_constant * temperature));
        integral += exp_term;

        x_prev = x;
    }

    return dx * integral;
}
