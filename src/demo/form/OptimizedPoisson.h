// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __OPTIMIZED_POISSON_H
#define __OPTIMIZED_POISSON_H

#include <dolfin/NewFiniteElement.h>
#include <dolfin/BilinearForm.h>

using namespace dolfin;

/// The finite element for which the form is generated.

class OptimizedPoissonFiniteElement : public NewFiniteElement
{
public:
  
  OptimizedPoissonFiniteElement() : NewFiniteElement() {}

  unsigned int spacedim() const
  {
    return 4;
  }

  unsigned int shapedim() const
  {
    return 3;
  }

  unsigned int vectordim() const
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

};

/// This is a hand-optimized version of the bilinear form for
/// Poisson's equation using first-order Lagrange elements on
/// tetrahedrons. Eventually, these optimizations will be automated
/// by FErari and built into the form compiler FFC.

class OptimizedPoissonBilinearForm : public BilinearForm
{
public:
  
  OptimizedPoissonBilinearForm(const NewFiniteElement& element) : BilinearForm(element) {}
  
  bool interior(real** A) const
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
    
    return true;
  }
  
};

#endif
