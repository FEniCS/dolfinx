// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2006-05-30
// Last changed: 2006-08-24
//
// At this point, all this demo does is to assemble
// the tensor-weighted Poisson matrix.

#include <dolfin.h>
#include "Bidomain.h"
  
using namespace dolfin;

// Tensor-valued anisotropic conductivity
class Conductivity : public Function
{
public:
  
  Conductivity(unsigned int i, unsigned int j) : i(i), j(j) {}

  real eval(const Point& p, unsigned int component)
  { 
    if ( i == j )
      return 0.001;
    else
      return 0.0;
  }

private:

  unsigned int i, j;

};

int main()
{
  Conductivity M00(0, 0), M01(0, 1), M02(0, 2);
  Conductivity M10(1, 0), M11(1, 1), M12(1, 2);
  Conductivity M20(2, 0), M21(2, 1), M22(2, 2);
  
  UnitCube mesh(16, 16, 16);
  Bidomain::BilinearForm mi(M00, M01, M02, M10, M11, M12, M20, M21, M22);
  Matrix Mi;
  FEM::assemble(mi, Mi, mesh);

  dolfin_info("\nThis demo just assembles the tensor-weighted Poisson matrix. It does not (yet) solve the bidomain equations.");
    
  return 0;
}
