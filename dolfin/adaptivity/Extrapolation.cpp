// Copyright (C) 2009 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-12-08
// Last changed: 2010-02-09

#include <vector>
#include <boost/scoped_array.hpp>

#include <dolfin/log/log.h>
#include <dolfin/la/GenericVector.h>
#include <dolfin/la/LAPACKMatrix.h>
#include <dolfin/la/LAPACKVector.h>
#include <dolfin/la/LAPACKSolvers.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/BoundaryMesh.h>
#include <dolfin/mesh/FacetCell.h>
#include <dolfin/fem/BasisFunction.h>
#include <dolfin/fem/DofMap.h>
#include <dolfin/function/Function.h>
#include <dolfin/function/FunctionSpace.h>
#include "Extrapolation.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
void Extrapolation::extrapolate(Function& w, const Function& v,
                                bool facet_extrapolation)
{
  // Using set_local for simplicity here
  not_working_in_parallel("Extrapolation");
  warning("Extrapolation not fully implemented yet.");

  // Check that the meshes are the same
  if (&w.function_space().mesh() != &v.function_space().mesh())
    error("Extrapolation must be computed on the same mesh.");

  // Extrapolate over interior (including boundary dofs)
  extrapolate_interior(w, v);

  // Extrapolate over boundary (overwriting earlier boundary dofs)
  if (facet_extrapolation)
    extrapolate_boundary(w, v);
}
//-----------------------------------------------------------------------------
void Extrapolation::extrapolate_interior(Function& w, const Function& v)
{
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
  boost::scoped_array<uint> dofs(new uint[W.dofmap().max_local_dimension()]);

  // Iterate over cells in mesh
  for (CellIterator cell0(mesh); !cell0.end(); ++cell0)
  {
    // Update UFC view
    c0.update(*cell0);

    // Create linear system (note that size varies over the mesh)
    const uint M = V.element().space_dimension() * (1 + cell0->num_entities(D));
    const uint N = W.element().space_dimension();
    LAPACKMatrix A(M, N);
    LAPACKVector b(M);

    // Check dimension of system
    if (M < N)
      error("Extrapolation failed, not enough equations.");

    // Add equations on cell
    uint offset = 0;
    offset += add_cell_equations(A, b, *cell0, *cell0, c0, c0, V, W, v, offset);

    // Add equations on neighboring cells
    for (CellIterator cell1(*cell0); !cell1.end(); ++cell1)
    {
      c1.update(*cell1);
      offset += add_cell_equations(A, b, *cell0, *cell1, c0, c1, V, W, v, offset);
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
void Extrapolation::extrapolate_boundary(Function& w, const Function& v)
{
  // Extract mesh and function spaces
  const FunctionSpace& V(v.function_space());
  const FunctionSpace& W(w.function_space());
  const Mesh& mesh(V.mesh());

  // Create boundary mesh
  BoundaryMesh boundary(mesh);

  // Initialize cell-cell connectivity for boundary mesh
  const uint D = mesh.topology().dim();
  boundary.init(D - 1, D - 1);

  // UFC cell views of center and patch cells
  UFCCell c0(mesh);
  UFCCell c1(mesh);

  // List of values for each dof of w (multivalued until we average)
  std::vector<std::vector<double> > dof_values_multi;
  dof_values_multi.resize(W.dim());

  // Local arrays for dof indices
  boost::scoped_array<uint> facet_dofs0(new uint[W.dofmap().num_facet_dofs()]);
  boost::scoped_array<uint> facet_dofs1(new uint[V.dofmap().num_facet_dofs()]);
  boost::scoped_array<uint> dofs(new uint[W.dofmap().max_local_dimension()]);

  // Iterate over facets (cells) in boundary
  for (CellIterator facet0(boundary); !facet0.end(); ++facet0)
  {
    // Get corresponding cell and update UFC view
    FacetCell cell0(mesh, *facet0);
    c0.update(cell0);

    // Create linear system (note that size varies over the mesh)
    const uint M = V.dofmap().num_facet_dofs() * (1 + facet0->num_entities(D - 1));
    const uint N = W.dofmap().num_facet_dofs();
    LAPACKMatrix A(M, N);
    LAPACKVector b(M);

    // Check dimension of system
    if (M < N)
      error("Extrapolation failed, not enough equations.");

    // Tabulate facet dofs
    W.dofmap().tabulate_facet_dofs(facet_dofs0.get(), cell0.facet_index());
    V.dofmap().tabulate_facet_dofs(facet_dofs1.get(), cell0.facet_index());

    // Compute non-facet dofs
    std::set<uint> non_facet_dofs0;
    for (uint i = 0; i < W.dofmap().local_dimension(c0); i++)
      non_facet_dofs0.insert(i);
    for (uint i = 0; i < W.dofmap().num_facet_dofs(); i++)
      non_facet_dofs0.erase(facet_dofs0[i]);

    // Add equations on facet
    uint offset = 0;
    offset += add_facet_equations(A, b, cell0, cell0, c0, c0, V, W, v, w,
                                  facet_dofs0.get(), facet_dofs1.get(),
                                  non_facet_dofs0, offset);

    // Add equations on neighboring facets
    for (CellIterator facet1(*facet0); !facet1.end(); ++facet1)
    {
      FacetCell cell1(mesh, *facet1);
      c1.update(cell1);
      V.dofmap().tabulate_facet_dofs(facet_dofs1.get(), cell1.facet_index());
      offset += add_facet_equations(A, b, cell0, cell1, c0, c1, V, W, v, w,
                                    facet_dofs0.get(), facet_dofs1.get(),
                                    non_facet_dofs0, offset);
    }

    // Solve least squares system
    LAPACKSolvers::solve_least_squares(A, b);

    // Tabulate dofs for w on cell and store values
    W.dofmap().tabulate_dofs(dofs.get(), c0, cell0.index());
    for (uint i = 0; i < W.dofmap().num_facet_dofs(); ++i)
      dof_values_multi[dofs[facet_dofs0[i]]].push_back(b[i]);
  }

  // Compute average of dof values (only touching facet dofs)
  boost::scoped_array<double> dof_values_single(new double[W.dim()]);
  w.vector().get_local(dof_values_single.get());
  for (uint i = 0; i < W.dim(); i++)
  {
    if (dof_values_multi[i].size() == 0)
      continue;
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
dolfin::uint Extrapolation::add_cell_equations(LAPACKMatrix& A,
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
      const BasisFunction phi(j, W.element(), c0);

      // Evaluate dof on basis function
      const double dof_value = V.element().evaluate_dof(i, phi, c1);

      // Insert into matrix
      A(offset + i, j) = dof_value;
    }
  }

  // Extract dof values for v on patch cell
  boost::scoped_array<double> dof_values(new double[V.element().space_dimension()]);
  v.restrict(dof_values.get(), V.element(), cell1, c1, -1);

  // Insert into vector
  for (uint i = 0; i < V.element().space_dimension(); ++i)
    b[offset + i] = dof_values[i];

  return V.element().space_dimension();
}
//-----------------------------------------------------------------------------
dolfin::uint Extrapolation::add_facet_equations(LAPACKMatrix& A,
                                                LAPACKVector& b,
                                                const FacetCell& cell0,
                                                const FacetCell& cell1,
                                                const ufc::cell& c0,
                                                const ufc::cell& c1,
                                                const FunctionSpace& V,
                                                const FunctionSpace& W,
                                                const Function& v,
                                                const Function& w,
                                                const uint* facet_dofs0,
                                                const uint* facet_dofs1,
                                                std::set<uint>& non_facet_dofs0,
                                                uint offset)
{
  // Iterate over facet dofs for V on patch cell
  for (uint i = 0; i < V.dofmap().num_facet_dofs(); ++i)
  {
    // Iterate over facet basis functions for W on center cell
    for (uint j = 0; j < W.dofmap().num_facet_dofs(); ++j)
    {
      // Create basis function
      const BasisFunction phi(facet_dofs0[j], W.element(), c0);

      // Evaluate dof on basis function
      const double dof_value = V.element().evaluate_dof(facet_dofs1[i], phi, c1);

      // Insert into matrix
      A(offset + i, j) = dof_value;
    }
  }

  // Extract dof values for v on patch cell
  boost::scoped_array<double> vdofs(new double[V.element().space_dimension()]);
  v.restrict(vdofs.get(), V.element(), cell1, c1, cell1.facet_index());

  // Extract dof values for w on center cell
  boost::scoped_array<double> wdofs(new double[W.element().space_dimension()]);
  w.restrict(wdofs.get(), W.element(), cell0, c0, cell0.facet_index());

  // Compute right-hand side
  for (uint i = 0; i < V.dofmap().num_facet_dofs(); ++i)
  {
    // Insert dof value for v
    b[offset + i] = vdofs[facet_dofs1[i]];

    // Subtract dof value of non-facet part of w
    for (std::set<uint>::iterator j = non_facet_dofs0.begin(); j != non_facet_dofs0.end(); ++j)
    {
      const BasisFunction phi(*j, W.element(), c0);
      b[offset + i] -= wdofs[*j] * V.element().evaluate_dof(facet_dofs1[i], phi, c1);
    }
  }

  return V.dofmap().num_facet_dofs();
}
//-----------------------------------------------------------------------------
