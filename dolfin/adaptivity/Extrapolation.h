// Copyright (C) 2009 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-12-08
// Last changed: 2010-04-20
//
// Modified by Marie E. Rognes (meg@simula.no) 2010.

#ifndef __EXTRAPOLATION_H
#define __EXTRAPOLATION_H

#include <vector>

namespace dolfin
{

  class Function;
  class Cell;
  class DirichletBC;
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
    static void extrapolate(Function& w, const Function& v);

    /// Compute extrapolation w from v and apply boundary conditions to w
    static void extrapolate(Function& w, const Function& v,
                            const std::vector<const DirichletBC*>& bcs);

  private:

    // Extrapolate over interior (including boundary dofs)
    static void extrapolate_interior(Function& w, const Function& v);

    // Add equations for current cell
    static void add_cell_equations(LAPACKMatrix& A,
                                   LAPACKVector& b,
                                   const Cell& cell0,
                                   const Cell& cell1,
                                   const ufc::cell& c0,
                                   const ufc::cell& c1,
                                   const FunctionSpace& V,
                                   const FunctionSpace& W,
                                   const Function& v,
                                   std::map<uint, uint>& dof2row);

    // Compute unique dofs in given cell
    static std::map<uint, uint> compute_unique_dofs(const Cell& cell, const ufc::cell& c,
                                                    const FunctionSpace& V,
                                                    uint& row, std::set<uint>& unique_dofs);

  };

}

#endif
