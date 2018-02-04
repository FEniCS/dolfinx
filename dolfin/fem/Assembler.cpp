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
  // Check if matrix should be nested
  const bool block_matrix = _a.size() > 1 or _a[0].size() > 1;

  if (A.empty())
  {
    // Initialise matrix
    if (block_matrix)
    {
      std::cout << "Have block matrix" << std::endl;
      // Loop over each form
      std::vector<std::shared_ptr<PETScMatrix>> mats;
      for (auto a_row : _a)
      {
        for (auto a : a_row)
        {
          if (a)
          {
            mats.push_back(std::make_shared<PETScMatrix>(MPI_COMM_WORLD));
            std::cout << "-- Init matrix block" << std::endl;
            fem::init(*mats.back(), *a);
            std::cout << "-- Mat size: " << mats.back()->size(0) << ", "
                      << mats.back()->size(1) << std::endl;
          }
          else
            mats.push_back(NULL);
        }
      }

      // Build list of PETSc Mat objects
      std::vector<Mat> petsc_mats;
      for (auto mat : mats)
      {
        if (mat)
          petsc_mats.push_back(mat->mat());
        else
          petsc_mats.push_back(nullptr);
      }

      // Intitialise block (MatNest) matrix
      std::cout << "Set PETSc mat type" << std::endl;
      MatSetType(A.mat(), MATNEST);
      std::cout << "Set submats" << std::endl;
      MatNestSetSubMats(A.mat(), _a.size(), NULL, _a[0].size(), NULL,
                        petsc_mats.data());
      std::cout << "End set submats" << std::endl;
      // A.apply(PETScMatrix::AssemblyType::FINAL);
    }
    else
      fem::init(A, *_a[0][0]);
  }
  else
  {
    // Matrix already intialised
    throw std::runtime_error("Not implemented");
    // Extract block
    // MatNestGetSubMat(Mat A,PetscInt idxm,PetscInt jdxm,Mat *sub)
  }

  // Assemble blocks (A)
  for (std::size_t i = 0; i < _a.size(); ++i)
  {
    // MatNestGetSubMat(Mat A,PetscInt idxm,PetscInt jdxm,Mat *sub)
    for (std::size_t j = 0; j < _a[i].size(); ++j)
    {
      if (_a[i][j])
      {
        Mat subA;
        MatNestGetSubMat(A.mat(), i, j, &subA);
        PETScMatrix mat(subA);
        std::cout << "Assembling into matrix:" << i << ", " << j << std::endl;
        this->assemble(mat, *_a[i][j]);
        std::cout << "End assembling into matrix:" << i << ", " << j
                  << std::endl;
      }
    }
  }

  A.apply(PETScMatrix::AssemblyType::FINAL);

  // ------

  // Build bcs and . . ..
  DirichletBC::Map boundary_values;
  for (std::size_t i = 0; i < _bcs.size(); ++i)
  {
    assert(_bcs[i]);
    assert(_bcs[i]->function_space());
    _bcs[i]->get_boundary_values(boundary_values);

    // FIXME: this probably isn't required with MatZeroRowsColumnsLocal -
    // check
    // if (mpi_size > 1 and _bcs[i]->method() != "pointwise")
    //  _bcs[i]->gather(boundary_values);
  }

  // Create x vector with bcs (could this be a local vector?)
  PETScVector x(A.mpi_comm());
  A.init_vector(x, 1);

  PETScVector _b(A.mpi_comm());
  A.init_vector(_b, 0);
  _b.zero();

  x.zero();
  std::cout << "Testing x size: " << x.size() << std::endl;
  std::vector<double> _x;
  std::vector<la_index_t> dofs;
  for (auto bc : boundary_values)
  {
    dofs.push_back(bc.first);
    _x.push_back(bc.second);
  }
  x.set_local(_x.data(), dofs.size(), dofs.data());
  x.apply();

  std::cout << x.str(true) << std::endl;
  std::cout << _b.str(true) << std::endl;

  // std::cout << "Apply bcs to matrix" << std::endl;
  // MatZeroRowsColumnsLocal(A.mat(), dofs.size(), dofs.data(), 1.0, x.vec(),
  //                             _b.vec());
  // std::cout << "End apply bcs to matrix" << std::endl;

  return;

  // // Assemble blocks (b)
  // for (auto row : _l)
  // {
  //   this->assemble(b, *row);
  // }

  // // const int mpi_size = dolfin::MPI::size(A.mpi_comm());

  // // Build bcs and . . ..
  // DirichletBC::Map boundary_values;
  // for (std::size_t i = 0; i < _bcs.size(); ++i)
  // {
  //   assert(_bcs[i]);
  //   assert(_bcs[i]->function_space());
  //   _bcs[i]->get_boundary_values(boundary_values);

  //   // FIXME: this probably isn't required with MatZeroRowsColumnsLocal -
  //   // check
  //   // if (mpi_size > 1 and _bcs[i]->method() != "pointwise")
  //   //  _bcs[i]->gather(boundary_values);
  // }

  // // Create x vector with bcs (could this be a local vector?)
  // PETScVector x(A.mpi_comm());
  // A.init_vector(x, 1);
  // x.zero();
  // std::vector<double> _x;
  // std::vector<la_index_t> dofs;
  // for (auto bc : boundary_values)
  // {
  //   dofs.push_back(bc.first);
  //   _x.push_back(bc.second);
  // }
  // x.set_local(_x.data(), dofs.size(), dofs.data());
  // x.apply();

  // // Apply Dirichlet boundary conditions
  // MatZeroRowsColumnsLocal(A.mat(), dofs.size(), dofs.data(), 1.0, x.vec(),
  //                         b.vec());
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

    if (dmap0.size() != dmap1.size())
    {
      for (int i = 0; i < dmap0.size(); ++i)
        std::cout << dmap0[i] << std::endl;
      std::cout << "-------------------" << std::endl;
      for (int j = 0; j < dmap1.size(); ++j)
        std::cout << dmap1[j] << std::endl;
    }

    // Add to matrix
    std::cout << "add to mat" << std::endl;
    A.add_local(Ae.data(), dmap0.size(), dmap0.data(), dmap1.size(),
                dmap1.data());
    std::cout << "post add to mat" << std::endl;
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
