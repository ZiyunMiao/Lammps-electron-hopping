#ifdef PAIR_CLASS
// clang-format off
PairStyle(electron_hopping,ElectronHopping);
// clang-format on
#else

#ifndef ELECTRON_HOPPING_H
#define ELECTRON_HOPPING_H

#include "pair.h"

namespace LAMMPS_NS {

class ElectronHopping : public Pair {
public:
    ElectronHopping(class LAMMPS *lmp);
    virtual ~ElectronHopping();

    virtual void init_style();
    virtual void init_list(int, char **);
    virtual void settings(int, char **);
    virtual void compute(int, int);

    void set_nstep(int n) { nstep = n; }

private:
    double force_constant, cutoff_sq, boltzmann_constant;
    int nstep;
    double temperature;

    double partition_function(int, int);
    double calculate_force(int, int, double);
};

}

#endif
#endif