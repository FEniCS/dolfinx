// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __POISSON_H
#define __POISSON_H

#include <dolfin/NewPDE.h>

namespace dolfin
{

  /// EXPERIMENTAL: Redesign of the evaluation of variational forms
  
  class Poisson : public NewPDE
  {
  public:
    
    void lhs(NewArray< NewArray<real> >& A)
    {
      real tmp = det / 6.0;
      
      real G00 = tmp*(g00*g00 + g01*g01 + g02*g02);
      real G01 = tmp*(g00*g10 + g01*g11 + g02*g12);
      real G02 = tmp*(g00*g20 + g01*g21 + g02*g22);
      real G11 = tmp*(g10*g10 + g11*g11 + g12*g12);
      real G12 = tmp*(g10*g20 + g11*g21 + g12*g22);
      real G22 = tmp*(g20*g20 + g21*g21 + g22*g22);
      
      A[1][1] = G00;
      A[1][2] = G01;
      A[1][3] = G02;
      A[2][2] = G11;
      A[2][3] = G12;
      A[3][3] = G22;
      A[0][1] = - G00 - G01 - G02;
      A[0][2] = - G01 - G11 - G12;
      A[0][3] = - G02 - G12 - G22;
      A[0][0] = - A[0][1] - A[0][2] - A[0][3];
      
      A[1][0] = A[0][1];
      A[2][0] = A[0][2];
      A[2][1] = G01;
      
    }
    
    void rhs(NewArray<real>& b)
    {
      b[0] = 1.0;
    }
    
  };
  
}

#endif
