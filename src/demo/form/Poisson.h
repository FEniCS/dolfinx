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

    Poisson(Function& f) : NewPDE(4, true, false), w0(4)
    {
      // Add functions
      add(w0, f);

      // Using default (full) nonzero pattern
    }

    unsigned int dim() const
    {
      return 1;
    }

    unsigned int dof(unsigned int i, const Cell& cell) const
    {
      return cell.nodeID(i);
    }

    const Point& coord(unsigned int i, const Cell& cell) const
    {
      return cell.node(i).coord();
    }

    void interiorElementMatrix(NewArray<NewArray<real> >& A) const
    {
      real tmp0 = det / 6.0;
      
      real G00 = tmp0*(g00*g00 + g01*g01 + g02*g02);
      real G01 = tmp0*(g00*g10 + g01*g11 + g02*g12);
      real G02 = tmp0*(g00*g20 + g01*g21 + g02*g22);
      real G11 = tmp0*(g10*g10 + g11*g11 + g12*g12);
      real G12 = tmp0*(g10*g20 + g11*g21 + g12*g22); 
      real G22 = tmp0*(g20*g20 + g21*g21 + g22*g22);
      
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
      A[3][0] = A[0][3];
      A[3][1] = A[1][3];
      A[3][2] = A[2][3];
    }
    
    void interiorElementVector(NewArray<real>& b) const
    {
      real tmp0 = det / 120.0;

      real G0 = tmp0*w0[0];
      real G1 = tmp0*w0[1];
      real G2 = tmp0*w0[2];
      real G3 = tmp0*w0[3];
      
      real tmp1 = G0 + G1 + G2 + G3;

      b[0] = tmp1 + G0;
      b[1] = tmp1 + G1;
      b[2] = tmp1 + G2;
      b[3] = tmp1 + G3;
    }

  private:

    NewArray<real> w0;
    
  };
  
}

#endif
