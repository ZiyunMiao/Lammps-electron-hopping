#ifdef PAIR_CLASS
PairStyle(electron_hopping, ElectronHopping);
#else
#ifndef PAIR_ELECTRON_HOPPING_H
#define PAIR_ELECTRON_HOPPING_H

#include "pair.h"
//#include "pair_hybrid_overlay.h"

namespace LAMMPS_NS {

class ElectronHopping : public Pair {
 public:
  ElectronHopping(class LAMMPS *);
  ~ElectronHopping() override;
  void compute(int, int)  override;
  void coeff(int, char **) override;
  void init_style() override;
  //virtual void init_list(int, char **);
  double init_one(int, int) override;
  void settings(int, char **) override;
  int pack_forward_comm(int , int *, double *, int , int *) override;
  void unpack_forward_comm(int , int , double *) override;
  void write_restart(FILE *) override;
  void read_restart(FILE *) override;
  void write_restart_settings(FILE *) override;
  void read_restart_settings(FILE *) override;
  void write_data(FILE *) override;
  void write_data_all(FILE *) override;

 protected:
  double **force_constant_1, **force_constant_2, **force_constant_3, **force_constant_4,**cut;
  double cut_global;
  double kb_constant, temp;
  int  n_exp_count;
  //double **hybrid_overlay;

  virtual void allocate();
  double partition_function(int);
  class AtomVecEllipsoid *avec;
  //double number_of_bonds();
};

} // namespace LAMMPS_NS

#endif // PAIR_ELECTRON_HOPPING_H
#endif // PAIR_CLASS

