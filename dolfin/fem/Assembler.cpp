// Copyright (C) 2018 Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "Assembler.h"
#include "DirichletBC.h"
#include "Form.h"
#include "GenericDofMap.h"
#include "UFC.h"
#include "utils.h"
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/la/PETScMatrix.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshIterator.h>
#include <string>

using namespace dolfin;

using EigenMatrixD
    = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

//-----------------------------------------------------------------------------
fem::Assembler::Assembler(
    std::vector<std::vector<std::shared_ptr<const Form>>> a,
    std::vector<std::shared_ptr<const Form>> L,
    std::vector<std::shared_ptr<const DirichletBC>> bcs)
    : _a(a), _l(L), _bcs(bcs)
{
  // Check shape of a and L

  // Check rank of forms
  /*
  if (a and a->rank() != 2)
  {
    throw std::runtime_error(
        "Expecting bilinear form (rank 2), but form has rank \'"
        + std::to_string(a->rank()) + "\'");
  }
  if (L and L->rank() != 1)
  {
    throw std::runtime_error(
        "Expecting linear form (rank 1), but form has rank \'"
        + std::to_string(L->rank()) + "\'");
  }
  */
}
//-----------------------------------------------------------------------------
void fem::Assembler::assemble(PETScMatrix& A, PETScVector& b)
{
  // Assemble A and b
  this->assemble(A, *_a[0][0]);
  this->assemble(b, *_l[0]);

  // const int mpi_size = dolfin::MPI::size(A.mpi_comm());

  // Build bcs and . . ..
  DirichletBC::Map boundary_values;
  for (std::size_t i = 0; i < _bcs.size(); ++i)
  {
    assert(_bcs[i]);
    assert(_bcs[i]->function_space());
    _bcs[i]->get_boundary_values(boundary_values);

    // FIXME: this probably isn't required with MatZeroRowsColumnsLocal - check
    // if (mpi_size > 1 and _bcs[i]->method() != "pointwise")
    //  _bcs[i]->gather(boundary_values);
  }

  // Create x vector with bcs (could this be a local vector?)
  PETScVector x(A.mpi_comm());
  A.init_vector(x, 1);
  x.zero();
  std::vector<double> _x;
  std::vector<la_index_t> dofs;
  for (auto bc : boundary_values)
  {
    dofs.push_back(bc.first);
    _x.push_back(bc.second);
  }
  x.set_local(_x.data(), dofs.size(), dofs.data());
  x.apply();

  // Apply Dirichlet boundary conditions
  MatZeroRowsColumnsLocal(A.mat(), dofs.size(), dofs.data(), 1.0, x.vec(),
                          b.vec());
}
//-----------------------------------------------------------------------------
void fem::Assembler::assemble(PETScMatrix& A, const Form& a)
{
  if (A.empty())
    fem::init(A, a);

  // Get mesh from form
  assert(a.mesh());
  const Mesh& mesh = *a.mesh();

  // FIXME: Remove UFC
  // Create data structures for local assembly data
  UFC ufc(a);

  const std::size_t gdim = mesh.geometry().dim();
  const std::size_t tdim = mesh.topology().dim();
  mesh.init(tdim);

  // Function spaces for each axis
  std::array<const FunctionSpace*, 2> spaces
      = {{a.function_space(0).get(), a.function_space(1).get()}};

  // Collect pointers to dof maps
  std::array<const GenericDofMap*, 2> dofmaps
      = {{spaces[0]->dofmap().get(), spaces[1]->dofmap().get()}};

  // Data structures used in assembly
  ufc::cell ufc_cell;
  EigenMatrixD coordinate_dofs;
  EigenMatrixD Ae;

  // Get cell integral
  auto cell_integral = ufc.default_cell_integral;

  // Iterate over all cells
  for (auto& cell : MeshRange<Cell>(mesh))
  {
    // Check that cell is not a ghost
    assert(!cell.is_ghost());

    // Get cell vertex coordinates
    coordinate_dofs.resize(cell.num_vertices(), gdim);
    cell.get_coordinate_dofs(coordinate_dofs);

    // Get UFC cell data
    cell.get_cell_data(ufc_cell);

    // Update UFC data to current cell
    ufc.update(cell, coordinate_dofs, ufc_cell,
               cell_integral->enabled_coefficients());

    // Get dof maps for cell
    auto dmap0 = dofmaps[0]->cell_dofs(cell.index());
    auto dmap1 = dofmaps[1]->cell_dofs(cell.index());

    // Size data structure for assembly
    Ae.resize(dmap0.size(), dmap1.size());
    Ae.setZero();

    // Compute cell matrix
    cell_integral->tabulate_tensor(Ae.data(), ufc.w(), coordinate_dofs.data(),
                                   ufc_cell.orientation);

    // Add to matrix
    A.add_local(Ae.data(), dmap0.size(), dmap0.data(), dmap1.size(),
                dmap1.data());
  }

  // FIXME: Put this elsewhere?
  // Finalise matrix
  A.apply(PETScMatrix::AssemblyType::FINAL);
}
//-----------------------------------------------------------------------------
void fem::Assembler::assemble(PETScVector& b, const Form& L)
{
  if (b.empty())
    fem::init(b, L);

  // Get mesh from form
  assert(L.mesh());
  const Mesh& mesh = *L.mesh();

  // FIXME: Remove UFC
  // Create data structures for local assembly data
  UFC ufc(L);

  const std::size_t gdim = mesh.geometry().dim();
  const std::size_t tdim = mesh.topology().dim();
  mesh.init(tdim);

  // Collect pointers to dof maps
  auto dofmap = L.function_space(0)->dofmap();

  // Data structures used in assembly
  ufc::cell ufc_cell;
  EigenMatrixD coordinate_dofs;
  Eigen::VectorXd be;

  // Get cell integral
  auto cell_integral = ufc.default_cell_integral;

  // Iterate over all cells
  for (auto& cell : MeshRange<Cell>(mesh))
  {
    // Check that cell is not a ghost
    assert(!cell.is_ghost());

    // Get cell vertex coordinates
    coordinate_dofs.resize(cell.num_vertices(), gdim);
    cell.get_coordinate_dofs(coordinate_dofs);

    // Get UFC cell data
    cell.get_cell_data(ufc_cell);

    // Update UFC data to current cell
    ufc.update(cell, coordinate_dofs, ufc_cell,
               cell_integral->enabled_coefficients());

    // Get dof maps for cell
    auto dmap = dofmap->cell_dofs(cell.index());
    // auto dmap1 = dofmaps[1]->cell_dofs(cell.index());

    // Size data structure for assembly
    be.resize(dmap.size());
    be.setZero();

    // Compute cell matrix
    cell_integral->tabulate_tensor(be.data(), ufc.w(), coordinate_dofs.data(),
                                   ufc_cell.orientation);

    // Add to vector
    b.add_local(be.data(), dmap.size(), dmap.data());
  }

  // FIXME: Put this elsewhere?
  // Finalise matrix
  b.apply();
}
//-----------------------------------------------------------------------------
