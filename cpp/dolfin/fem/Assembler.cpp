// Copyright (C) 2018 Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "Assembler.h"
#include "DirichletBC.h"
#include "Form.h"
#include "GenericDofMap.h"
#include "SparsityPatternBuilder.h"
#include "UFC.h"
#include "utils.h"
#include <dolfin/common/types.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/la/PETScMatrix.h>
#include <dolfin/la/SparsityPattern.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshIterator.h>
#include <string>

using namespace dolfin;
using namespace dolfin::fem;

//-----------------------------------------------------------------------------
Assembler::Assembler(std::vector<std::vector<std::shared_ptr<const Form>>> a,
                     std::vector<std::shared_ptr<const Form>> L,
                     std::vector<std::shared_ptr<const DirichletBC>> bcs)
    : _a(a), _l(L), _bcs(bcs)
{
  assert(!a.empty());
  assert(!a[0].empty());

  // FIXME: check that a is square

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
void Assembler::assemble(la::PETScMatrix& A, BlockType type)
{
  // Check if matrix should be nested
  assert(!_a.empty());
  const bool use_nested_matrix = false;
  const bool block_matrix = _a.size() > 1 or _a[0].size() > 1;

  if (A.empty())
  {
    // Initialise matrix if empty

    // Build array of pointers to forms
    std::vector<std::vector<const Form*>> forms(
        _a.size(), std::vector<const Form*>(_a[0].size()));
    for (std::size_t i = 0; i < _a.size(); ++i)
      for (std::size_t j = 0; j < _a[i].size(); ++j)
        forms[i][j] = _a[i][j].get();

    // Initialise matrix
    if (use_nested_matrix)
      fem::init_nest(A, forms);
    else if (block_matrix)
      fem::init_monolithic(A, forms);
    else
      init(A, *_a[0][0]);
  }

  // Assemble matrix

  if (use_nested_matrix)
  {
    for (std::size_t i = 0; i < _a.size(); ++i)
    {
      for (std::size_t j = 0; j < _a[i].size(); ++j)
      {
        if (_a[i][j])
        {
          Mat subA;
          MatNestGetSubMat(A.mat(), i, j, &subA);
          la::PETScMatrix mat(subA);
          //std::cout << "Assembling into matrix:" << i << ", " << j << std::endl;
          this->assemble(mat, *_a[i][j], _bcs);
          //std::cout << "End assembling into matrix:" << i << ", " << j
          //          << std::endl;
        }
        else
        {
          // Null block, do nothing
        }
      }
    }
  }
  else if (block_matrix)
  {
    std::cout << "Assembling block matrix (non-nested)" << std::endl;
    std::int64_t offset_row = 0;
    for (std::size_t i = 0; i < _a.size(); ++i)
    {
      // Loop over columns
      std::int64_t offset_col = 0;
      for (std::size_t j = 0; j < _a[i].size(); ++j)
      {
        if (_a[i][j])
        {
          auto map0 = _a[i][j]->function_space(0)->dofmap()->index_map();
          auto map1 = _a[i][j]->function_space(1)->dofmap()->index_map();
          auto map0_size = map0->size(common::IndexMap::MapSize::ALL);
          auto map1_size = map1->size(common::IndexMap::MapSize::ALL);

          std::vector<PetscInt> index0(map0_size);
          std::vector<PetscInt> index1(map1_size);
          std::iota(index0.begin(), index0.end(), offset_row);
          std::iota(index1.begin(), index1.end(), offset_col);

          std::cout << "Block: " << i << ", " << j << std::endl;
          std::cout << "***** start index and size 0: " << offset_row << ", "
                    << offset_row + index0.size() << std::endl;
          std::cout << "***** start index and size 1: " << offset_col << ", "
                    << offset_col + index1.size() << std::endl;

          IS is0, is1;
          ISCreateBlock(A.mpi_comm(), map0->block_size(), index0.size(),
                        index0.data(), PETSC_COPY_VALUES, &is0);
          ISCreateBlock(A.mpi_comm(), map1->block_size(), index1.size(),
                        index1.data(), PETSC_COPY_VALUES, &is1);

          Mat subA;
          MatGetLocalSubMatrix(A.mat(), is0, is1, &subA);
          la::PETScMatrix mat(subA);
          std::cout << "Mat size: " << mat.size(0) << ", " << mat.size(1)
                    << std::endl;

          double one = 10000.0;
          PetscInt zero = 0;
          std::cout << "   Add single entry" << std::endl;
          mat.add_local(&one, 1, &zero, 1, &zero);

          PetscInt onei = 1;
          std::cout << "   Add single entry" << std::endl;
          mat.add_local(&one, 1, &zero, 1, &onei);

          // A.str(true);
          // std::cout << "Assembling into matrix (non-nested):" << i << ", "
          // <<
          // j
          //          << std::endl;
          this->assemble(mat, *_a[i][j], _bcs);
          // std::cout << "End assembling into matrix:" << i << ", " << j
          //         << std::endl;

          MatRestoreLocalSubMatrix(A.mat(), is0, is1, &subA);
          ISDestroy(&is0);
          ISDestroy(&is1);

          offset_col += map1_size;
        }
      }
      auto map0 = _a[i][0]->function_space(0)->dofmap()->index_map();
      auto map0_size = map0->size(common::IndexMap::MapSize::ALL);
      offset_row += map0_size;
    }
  }
  else
  {
    this->assemble(A, *_a[0][0], _bcs);
  }

  A.apply(la::PETScMatrix::AssemblyType::FINAL);

  // return;
}
//-----------------------------------------------------------------------------
void Assembler::assemble(la::PETScVector& b)
{
  // Assemble vector
  this->assemble(b, *_l[0]);

  // Apply bcs to RHS of vector
  for (std::size_t i = 0; i < _l.size(); ++i)
    for (std::size_t j = 0; j < _a[i].size(); ++j)
      apply_bc(b, *_a[i][j], _bcs);

  // Set bc values
  set_bc(b, *_l[0], _bcs);

  // // Assemble blocks (b)
  // for (auto row : _l)
  // {
  //   this->assemble(b, *row);
  // }
}
//-----------------------------------------------------------------------------
void Assembler::assemble(la::PETScMatrix& A, la::PETScVector& b)
{
  // TODO: pre common boundary condition data

  // Assemble matrix
  assemble(A);

  // Assemble vector
  assemble(b);
}
//-----------------------------------------------------------------------------
void Assembler::assemble(la::PETScMatrix& A, const Form& a,
                         std::vector<std::shared_ptr<const DirichletBC>> bcs)
{
  if (A.empty())
    init(A, a);

  // Get mesh from form
  assert(a.mesh());
  const mesh::Mesh& mesh = *a.mesh();

  // FIXME: Remove UFC
  // Create data structures for local assembly data
  UFC ufc(a);

  const std::size_t gdim = mesh.geometry().dim();
  const std::size_t tdim = mesh.topology().dim();
  mesh.init(tdim);

  // Function spaces for each axis
  std::array<const function::FunctionSpace*, 2> spaces
      = {{a.function_space(0).get(), a.function_space(1).get()}};

  // Collect pointers to dof maps
  std::array<const GenericDofMap*, 2> dofmaps
      = {{spaces[0]->dofmap().get(), spaces[1]->dofmap().get()}};

  // FIXME: Move out of this function
  // FIXME: For the matrix, we only need to know if there is a boundary
  // condition on the entry. The value is not required.
  // FIXME: Avoid duplication when spaces[0] == spaces[1]
  // Collect boundary conditions by matrix axis
  std::array<DirichletBC::Map, 2> boundary_values;
  for (std::size_t i = 0; i < bcs.size(); ++i)
  {
    assert(bcs[i]);
    assert(bcs[i]->function_space());
    for (std::size_t axis = 0; axis < 2; ++axis)
    {
      if (spaces[axis]->contains(*bcs[i]->function_space()))
      {
        // FIXME: find way to avoid gather, or perform with a single
        // gather
        bcs[i]->get_boundary_values(boundary_values[axis]);
        if (MPI::size(mesh.mpi_comm()) > 1
            and bcs[i]->method() != DirichletBC::Method::pointwise)
        {
          bcs[i]->gather(boundary_values[axis]);
        }
      }
    }
  }

  // Data structures used in assembly
  ufc::cell ufc_cell;
  EigenRowArrayXXd coordinate_dofs;
  EigenRowArrayXXd Ae;

  // Get cell integral
  auto cell_integral = a.integrals().cell_integral();

  // Iterate over all cells
  for (auto& cell : mesh::MeshRange<mesh::Cell>(mesh))
  {
    std::cout << "Iterate over cells" << std::endl;
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

    // FIXME: Pass in list  of cells, and list of local dofs, with
    // Dirichlet conditions
    // Note: could use zero dof indices to have PETSc do this
    // Zero rows/columns for Dirichlet bcs
    /*
    for (int i = 0; i < Ae.rows(); ++i)
    {
      const std::size_t ii = dmap0[i];
      DirichletBC::Map::const_iterator bc_value = boundary_values[0].find(ii);
      if (bc_value != boundary_values[0].end())
        Ae.row(i).setZero();
    }
    // Loop over columns
    for (int j = 0; j < Ae.cols(); ++j)
    {
      const std::size_t jj = dmap1[j];
      DirichletBC::Map::const_iterator bc_value = boundary_values[1].find(jj);
      if (bc_value != boundary_values[1].end())
        Ae.col(j).setZero();
    }
    */

    // Add to matrix
    /*
    std::cout << "Add to matrix: " << std::endl;
    for (std::size_t i = 0; i < dmap0.size(); ++i)
      std::cout << "  0: " << dmap0[i] << std::endl;
    for (std::size_t i = 0; i < dmap1.size(); ++i)
      std::cout << "  1: " << dmap1[i] << std::endl;
  */

    A.add_local(Ae.data(), dmap0.size(), dmap0.data(), dmap1.size(),
                dmap1.data());
    // std::cout << "Post add to matrix: " << std::endl;
  }

  // FIXME: Put this elsewhere?
  // Finalise matrix
  // A.apply(la::PETScMatrix::AssemblyType::FINAL);

  // FIXME: Move this outside of function
  // Place '1' on diagonal for bc entries
  /*
  if (spaces[0] == spaces[1])
  {
    std::vector<la_index_t> rows;
    for (auto bc : boundary_values[0])
      rows.push_back(bc.first);
    A.zero_local(rows.size(), rows.data(), 1.0);
  }
  */
}
//-----------------------------------------------------------------------------
void Assembler::assemble(la::PETScVector& b, const Form& L)
{
  if (b.empty())
    init(b, L);

  // Get mesh from form
  assert(L.mesh());
  const mesh::Mesh& mesh = *L.mesh();

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
  EigenRowArrayXXd coordinate_dofs;
  EigenArrayXd be;

  // Get cell integral
  auto cell_integral = L.integrals().cell_integral();

  // Iterate over all cells
  for (auto& cell : mesh::MeshRange<mesh::Cell>(mesh))
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
void Assembler::apply_bc(la::PETScVector& b, const Form& a,
                         std::vector<std::shared_ptr<const DirichletBC>> bcs)
{
  // Get mesh from form
  assert(a.mesh());
  const mesh::Mesh& mesh = *a.mesh();

  const std::size_t gdim = mesh.geometry().dim();

  // Get bcs
  DirichletBC::Map boundary_values;
  for (std::size_t i = 0; i < bcs.size(); ++i)
  {
    assert(bcs[i]);
    assert(bcs[i]->function_space());
    if (a.function_space(1)->contains(*bcs[i]->function_space()))
    {
      bcs[i]->get_boundary_values(boundary_values);
      if (MPI::size(mesh.mpi_comm()) > 1
          and bcs[i]->method() != DirichletBC::Method::pointwise)
      {
        bcs[i]->gather(boundary_values);
      }
    }
  }

  // std::array<const function::FunctionSpace*, 2> spaces
  //    = {{a.function_space(0).get(), a.function_space(1).get()}};

  // Get dofmap for columns a a[i]
  auto dofmap0 = a.function_space(0)->dofmap();
  auto dofmap1 = a.function_space(1)->dofmap();

  ufc::cell ufc_cell;
  EigenRowArrayXXd Ae;
  EigenArrayXd be;
  EigenRowArrayXXd coordinate_dofs;

  // Create data structures for local assembly data
  UFC ufc(a);

  // Get cell integral
  auto cell_integral = a.integrals().cell_integral();

  // Iterate over all cells
  for (auto& cell : mesh::MeshRange<mesh::Cell>(mesh))
  {
    // Check that cell is not a ghost
    assert(!cell.is_ghost());

    // Get dof maps for cell
    auto dmap1 = dofmap1->cell_dofs(cell.index());

    // Check if bc is applied to cell
    bool has_bc = false;
    for (int i = 0; i < dmap1.size(); ++i)
    {
      const std::size_t ii = dmap1[i];
      if (boundary_values.find(ii) != boundary_values.end())
      {
        has_bc = true;
        break;
      }
    }

    // std::cout << "Applying bcs" << std::endl;
    if (!has_bc)
      continue;
    // std::cout << "  has bc" << std::endl;

    // Get cell vertex coordinates
    coordinate_dofs.resize(cell.num_vertices(), gdim);
    cell.get_coordinate_dofs(coordinate_dofs);

    // Get UFC cell data
    cell.get_cell_data(ufc_cell);

    // Update UFC data to current cell
    ufc.update(cell, coordinate_dofs, ufc_cell,
               cell_integral->enabled_coefficients());

    // Size data structure for assembly
    auto dmap0 = dofmap1->cell_dofs(cell.index());
    Ae.resize(dmap0.size(), dmap1.size());
    Ae.setZero();
    cell_integral->tabulate_tensor(Ae.data(), ufc.w(), coordinate_dofs.data(),
                                   ufc_cell.orientation);

    // FIXME: Is this required?
    // Zero Dirichlet rows in Ae
    /*
    if (spaces[0] == spaces[1])
    {
      for (int i = 0; i < dmap0.size(); ++i)
      {
        const std::size_t ii = dmap0[i];
        auto bc = boundary_values.find(ii);
        if (bc != boundary_values.end())
          Ae.row(i).setZero();
      }
    }
    */

    // Size data structure for assembly
    be.resize(dmap0.size());
    be.setZero();

    for (int j = 0; j < dmap1.size(); ++j)
    {
      const std::size_t jj = dmap1[j];
      auto bc = boundary_values.find(jj);
      if (bc != boundary_values.end())
      {
        be -= Ae.col(j) * bc->second;
      }
    }

    // Add to vector
    b.add_local(be.data(), dmap0.size(), dmap0.data());
  }

  // FIXME: Put this elsewhere?
  // Finalise matrix
  b.apply();
}
//-----------------------------------------------------------------------------
void Assembler::set_bc(la::PETScVector& b, const Form& L,
                       std::vector<std::shared_ptr<const DirichletBC>> bcs)
{
  // Get mesh from form
  assert(L.mesh());
  const mesh::Mesh& mesh = *L.mesh();

  auto V = L.function_space(0);

  // Get bcs
  DirichletBC::Map boundary_values;
  for (std::size_t i = 0; i < bcs.size(); ++i)
  {
    assert(bcs[i]);
    assert(bcs[i]->function_space());
    if (V->contains(*bcs[i]->function_space()))
    {
      bcs[i]->get_boundary_values(boundary_values);
      if (MPI::size(mesh.mpi_comm()) > 1
          and bcs[i]->method() != DirichletBC::Method::pointwise)
      {
        bcs[i]->gather(boundary_values);
      }
    }
  }

  std::vector<double> values;
  values.reserve(boundary_values.size());
  std::vector<la_index_t> rows;
  rows.reserve(boundary_values.size());
  for (auto bc : boundary_values)
  {
    rows.push_back(bc.first);
    values.push_back(bc.second);
  }

  b.set_local(values.data(), values.size(), rows.data());
  b.apply();
}
//-----------------------------------------------------------------------------
