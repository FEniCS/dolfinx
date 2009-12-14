// Copyright (C) 2009 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-12-08
// Last changed: 2009-12-14

#ifndef __RECONSTRUCTION_H
#define __RECONSTRUCTION_H

#include <vector>

namespace dolfin
{

  class Function;
  class Cell;
  class LAPACKMatrix;
  class LAPACKVector;

  /// This class implements an algorithm for reconstructing a function
  /// on a given function space from an approximation of that function
  /// on a possibly lower-order function space.
  ///
  /// This can be used to obtain a higher-order approximation of a
  /// computed dual solution, which is necessary when the computed
  /// dual approximation is in the test space of the primal problem,
  /// thereby being orthogonal to the residual.
  ///
  /// It is assumed that the reconstruction is computed on the same
  /// mesh as the original function.

  class Reconstruction
  {
  public:

    /// Compute reconstruction w from v
    static void reconstruct(Function& w, const Function& v);

  private:

    // Add equations for current cell
    static uint add_equations(LAPACKMatrix& A,
                              LAPACKVector& b,
                              const Cell& cell0,
                              const Cell& cell1,
                              const ufc::cell& c0,
                              const ufc::cell& c1,
                              const FunctionSpace& V,
                              const FunctionSpace& W,
                              const Function& v,
                              uint offset);

  };

}

#endif
