// Copyright (C) 2009 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-12-08
// Last changed: 2010-02-08

#ifndef __EXTRAPOLATION_H
#define __EXTRAPOLATION_H

#include <vector>

namespace dolfin
{

  class Function;
  class Cell;
  class FacetCell;
  class LAPACKMatrix;
  class LAPACKVector;

  /// This class implements an algorithm for extrapolating a function
  /// on a given function space from an approximation of that function
  /// on a possibly lower-order function space.
  ///
  /// This can be used to obtain a higher-order approximation of a
  /// computed dual solution, which is necessary when the computed
  /// dual approximation is in the test space of the primal problem,
  /// thereby being orthogonal to the residual.
  ///
  /// It is assumed that the extrapolation is computed on the same
  /// mesh as the original function.

  class Extrapolation
  {
  public:

    /// Compute extrapolation w from v
    static void extrapolate(Function& w, const Function& v,
                            bool facet_extrapolation=true);

  private:

    // Extrapolate over interior (including boundary dofs)
    static void extrapolate_interior(Function& w, const Function& v);

    // Extrapolate over boundary (overwriting earlier boundary dofs)
    static void extrapolate_boundary(Function& w, const Function& v);

    // Add equations for current cell
    static uint add_cell_equations(LAPACKMatrix& A,
                                   LAPACKVector& b,
                                   const Cell& cell0,
                                   const Cell& cell1,
                                   const ufc::cell& c0,
                                   const ufc::cell& c1,
                                   const FunctionSpace& V,
                                   const FunctionSpace& W,
                                   const Function& v,
                                   uint offset);

    // Add equations for current facet
    static uint add_facet_equations(LAPACKMatrix& A,
                                    LAPACKVector& b,
                                    const FacetCell& cell0,
                                    const FacetCell& cell1,
                                    const ufc::cell& c0,
                                    const ufc::cell& c1,
                                    const FunctionSpace& V,
                                    const FunctionSpace& W,
                                    const Function& v,
                                    const uint* facet_dofs0,
                                    const uint* facet_dofs1,
                                    uint offset);

  };

}

#endif
