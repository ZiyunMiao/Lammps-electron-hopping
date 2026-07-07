#ifdef PAIR_CLASS
PairStyle(electron_hopping, ElectronHopping);
#else

#ifndef ELECTRON_HOPPING_H
#define ELECTRON_HOPPING_H

#include "pair.h"
#include "pair_hybrid_overlay.h"

namespace LAMMPS_NS {

class ElectronHopping : public Pair {
public:
    ElectronHopping(class LAMMPS *);
    ~ElectronHopping() override;

    void init_style() override;
    void init_list(int, char **);
    void settings(int, char **) override;
    void compute(int, int) override;
    void set_nstep(int n) { nstep = n; }
    void coeff(int narg, char **arg) override;

private:
    double force_constant, equilibrium_distance, cutoff_sq, boltzmann_constant;
    int nstep;
    double temperature;

    double partition_function(int, int);
    double calculate_force(int, int, double);
    double compute_temperature();
    virtual void allocate();

    PairHybridOverlay *hybrid_overlay;
};

} // namespace LAMMPS_NS

#endif // ELECTRON_HOPPING_H

#endif // PAIR_CLASS

