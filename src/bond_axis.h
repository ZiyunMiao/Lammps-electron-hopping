
#ifdef BOND_CLASS
// clang-format off
BondStyle(axis,BondAxis);
// clang-format on
#else

#ifndef LMP_BOND_AXIS_H
#define LMP_BOND_AXIS_H

#include "bond.h"
#include "atom_vec_ellipsoid.h"
#include <vector>

namespace LAMMPS_NS {

class BondAxis : public Bond {
 public:
  BondAxis(class LAMMPS *);
  ~BondAxis() override;
  void compute(int, int) override;
  void coeff(int, char **) override;
  double equilibrium_distance(int) override;
  void write_restart(FILE *) override;
  void read_restart(FILE *) override;
  void write_data(FILE *) override;
  double single(int, double, int, int, double &) override;
  //void born_matrix(int, double, int, int, double &, double &) override;
  void *extract(const char *, int &) override;
  int pack_forward_comm(int , int *, double *, int , int *) override;
  void unpack_forward_comm(int , int , double *) override;

 protected:
  double *kr, *kR, *kalpha, *kphi, *R , *r0;
  class AtomVecEllipsoid *avec;
  virtual void allocate();
  double roundToPrecision(double, int);
  double dotProduct(const std::vector<double>& v1, const std::vector<double>& v2);
  std::vector<double> crossProduct(const std::vector<double>& v1, const std::vector<double>& v2); 
  double magnitude(const std::vector<double>& v);

};

}    // namespace LAMMPS_NS

#endif
#endif
