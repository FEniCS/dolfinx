// Copyright (C) 2009 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-12-08
// Last changed: 2010-02-04

#include <vector>
#include <boost/scoped_array.hpp>

#include <dolfin/log/log.h>
#include <dolfin/la/GenericVector.h>
#include <dolfin/la/LAPACKMatrix.h>
#include <dolfin/la/LAPACKVector.h>
#include <dolfin/la/LAPACKSolvers.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/fem/BasisFunction.h>
#include <dolfin/fem/DofMap.h>
#include <dolfin/function/Function.h>
#include <dolfin/function/FunctionSpace.h>
#include "Extrapolation.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
void Extrapolation::extrapolate(Function& w, const Function& v)
{
  // Using set_local for simplicity here
  not_working_in_parallel("Extrapolation");
  warning("Extrapolation not fully implemented yet.");

  // Check that the meshes are the same
  if (&w.function_space().mesh() != &v.function_space().mesh())
    error("Extrapolation must be computed on the same mesh.");

  // Extract mesh and function spaces
  const FunctionSpace& V(v.function_space());
  const FunctionSpace& W(w.function_space());
  const Mesh& mesh(V.mesh());

  // Initialize cell-cell connectivity
  const uint D = mesh.topology().dim();
  mesh.init(D, D);

  // UFC cell views of center and patch cells
  UFCCell c0(mesh);
  UFCCell c1(mesh);

  // List of values for each dof of w (multivalued until we average)
  std::vector<std::vector<double> > dof_values_multi;
  dof_values_multi.resize(W.dim());

  // Local array for dof indices
  boost::scoped_array<uint> dofs(new uint[V.dofmap().max_local_dimension()]);

  // Iterate over cells in mesh of extrapolation space
  for (CellIterator cell0(w.function_space().mesh()); !cell0.end(); ++cell0)
  {
    // Update UFC view
    c0.update(*cell0);

    // Create linear system
    const uint M = V.element().space_dimension() * (1 + cell0->num_entities(D));
    const uint N = W.element().space_dimension();
    LAPACKMatrix A(M, N);
    LAPACKVector b(M);

    // Add equations for cell and its neighbors
    uint offset = 0;
    offset += add_equations(A, b, *cell0, *cell0, c0, c0, V, W, v, offset);
    for (CellIterator cell1(*cell0); !cell1.end(); ++cell1)
    {
      c1.update(*cell1);
      offset += add_equations(A, b, *cell0, *cell1, c0, c1, V, W, v, offset);
    }

    // Solve least squares system
    LAPACKSolvers::solve_least_squares(A, b);

    // Tabulate dofs for w on cell and store values
    W.dofmap().tabulate_dofs(dofs.get(), c0, cell0->index());
    for (uint i = 0; i < W.dofmap().local_dimension(c0); ++i)
      dof_values_multi[dofs[i]].push_back(b[i]);
  }

  // Compute average of dof values
  boost::scoped_array<double> dof_values_single(new double[W.dim()]);
  for (uint i = 0; i < W.dim(); i++)
  {
    double s = 0.0;
    for (uint j = 0; j < dof_values_multi[i].size(); ++j)
      s += dof_values_multi[i][j];
    s /= static_cast<double>(dof_values_multi[i].size());
    dof_values_single[i] = s;
  }

  // Update dofs for w
  w.vector().set_local(dof_values_single.get());
}
//-----------------------------------------------------------------------------
dolfin::uint Extrapolation::add_equations(LAPACKMatrix& A,
                                          LAPACKVector& b,
                                          const Cell& cell0,
                                          const Cell& cell1,
                                          const ufc::cell& c0,
                                          const ufc::cell& c1,
                                          const FunctionSpace& V,
                                          const FunctionSpace& W,
                                          const Function& v,
                                          uint offset)
{
  // Iterate over dofs for V on patch cell
  for (uint i = 0; i < V.element().space_dimension(); ++i)
  {
    // Iterate over basis functions for W on center cell
    for (uint j = 0; j < W.element().space_dimension(); ++j)
    {
      // Create basis function
      BasisFunction phi(j, W.element(), c0);

      // Evaluate dof on basis function
      const double dof_value = V.element().evaluate_dof(i, phi, c1);

      // Insert into matrix
      A(offset + i, j) = dof_value;
    }
  }

  // Extract dof values for v on patch cell
  boost::scoped_array<double> dof_values(new double(V.element().space_dimension()));
  v.restrict(dof_values.get(), V.element(), cell1, c1, -1);

  // Insert into vector
  for (uint i = 0; i < V.element().space_dimension(); ++i)
    b[offset + i] = dof_values[i];

  return V.element().space_dimension();
}
//-----------------------------------------------------------------------------
