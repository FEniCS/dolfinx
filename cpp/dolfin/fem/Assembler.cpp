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
void Assembler::assemble(la::PETScMatrix& A, BlockType block_type)
{
  // Check if matrix should be nested
  assert(!_a.empty());
  const bool block_matrix = _a.size() > 1 or _a[0].size() > 1;

  if (A.empty())
  {
    // std::cout << "Init matrix" << std::endl;

    // Initialise matrix if empty

    // Build array of pointers to forms
    std::vector<std::vector<const Form*>> forms(
        _a.size(), std::vector<const Form*>(_a[0].size()));
    for (std::size_t i = 0; i < _a.size(); ++i)
      for (std::size_t j = 0; j < _a[i].size(); ++j)
        forms[i][j] = _a[i][j].get();

    // Initialise matrix
    if (block_type == BlockType::nested)
    {
      fem::init_nest(A, forms);
    }
    else if (block_matrix and block_type == BlockType::monolithic)
    {
      // Initialise matrix
      fem::init_monolithic(A, forms);

      // Create local-to-global maps and attach to matrix
      std::vector<PetscInt> _map0, _map1;

      // Build list of index maps
      std::vector<const common::IndexMap*> maps0, maps1;
      for (std::size_t i = 0; i < _a.size(); ++i)
      {
        auto map = _a[i][0]->function_space(0)->dofmap()->index_map();
        maps0.push_back(map.get());
      }
      for (std::size_t i = 0; i < _a[0].size(); ++i)
      {
        auto map = _a[0][i]->function_space(1)->dofmap()->index_map();
        maps1.push_back(map.get());
      }

      // Row local-to-global map
      for (std::size_t i = 0; i < _a.size(); ++i)
      {
        auto map = _a[i][0]->function_space(0)->dofmap()->index_map();
        for (std::size_t k = 0; k < map->size(common::IndexMap::MapSize::ALL);
             ++k)
        {
          auto index_k = map->local_to_global(k);
          std::size_t index = get_global_index(maps0, i, index_k);
          _map0.push_back(index);
        }
      }

      // Column local-to-global map
      for (std::size_t i = 0; i < _a[0].size(); ++i)
      {
        auto map = _a[0][i]->function_space(1)->dofmap()->index_map();
        for (std::size_t k = 0; k < map->size(common::IndexMap::MapSize::ALL);
             ++k)
        {
          auto index_k = map->local_to_global(k);
          std::size_t index = get_global_index(maps1, i, index_k);
          _map1.push_back(index);
        }
      }

      // Create PETSc local-to-global map/index sets and attach to matrix
      ISLocalToGlobalMapping petsc_local_to_global0, petsc_local_to_global1;
      ISLocalToGlobalMappingCreate(MPI_COMM_SELF, 1, _map0.size(), _map0.data(),
                                   PETSC_COPY_VALUES, &petsc_local_to_global0);
      ISLocalToGlobalMappingCreate(MPI_COMM_SELF, 1, _map1.size(), _map1.data(),
                                   PETSC_COPY_VALUES, &petsc_local_to_global1);
      MatSetLocalToGlobalMapping(A.mat(), petsc_local_to_global0,
                                 petsc_local_to_global1);

      // Clean up local-to-global maps
      ISLocalToGlobalMappingDestroy(&petsc_local_to_global0);
      ISLocalToGlobalMappingDestroy(&petsc_local_to_global1);
    }
    else
    {
      init(A, *_a[0][0]);
    }
  }

  // Get matrix type
  MatType mat_type;
  MatGetType(A.mat(), &mat_type);
  const bool is_matnest = strcmp(mat_type, MATNEST) == 0 ? true : false;

  // Assemble matrix

  if (is_matnest)
  {
    for (std::size_t i = 0; i < _a.size(); ++i)
    {
      for (std::size_t j = 0; j < _a[i].size(); ++j)
      {
        // Get submatrix
        Mat subA;
        MatNestGetSubMat(A.mat(), i, j, &subA);
        if (_a[i][j])
        {
          la::PETScMatrix mat(subA);
          this->assemble(mat, *_a[i][j], _bcs);
        }
        else
        {
          // FIXME: Figure out how to check that matrix block is null
          // Null block, do nothing
        }
      }
    }
  }
  else if (block_matrix)
  {
    std::int64_t offset_row = 0;
    for (std::size_t i = 0; i < _a.size(); ++i)
    {
      // Loop over columns
      std::int64_t offset_col = 0;
      for (std::size_t j = 0; j < _a[i].size(); ++j)
      {
        if (_a[i][j])
        {
          // Build index set for block
          auto map0 = _a[i][j]->function_space(0)->dofmap()->index_map();
          auto map1 = _a[i][j]->function_space(1)->dofmap()->index_map();
          auto map0_size = map0->size(common::IndexMap::MapSize::ALL);
          auto map1_size = map1->size(common::IndexMap::MapSize::ALL);

          std::vector<PetscInt> index0(map0_size);
          std::vector<PetscInt> index1(map1_size);
          std::iota(index0.begin(), index0.end(), offset_row);
          std::iota(index1.begin(), index1.end(), offset_col);

          IS is0, is1;
          ISCreateBlock(MPI_COMM_SELF, map0->block_size(), index0.size(),
                        index0.data(), PETSC_COPY_VALUES, &is0);
          ISCreateBlock(MPI_COMM_SELF, map1->block_size(), index1.size(),
                        index1.data(), PETSC_COPY_VALUES, &is1);

          // Get sub-matrix
          Mat subA;
          MatGetLocalSubMatrix(A.mat(), is0, is1, &subA);

          // Assemble block
          la::PETScMatrix mat(subA);
          this->assemble(mat, *_a[i][j], _bcs);

          // Restore sub-matrix and destroy index sets
          MatRestoreLocalSubMatrix(A.mat(), is0, is1, &subA);
          ISDestroy(&is0);
          ISDestroy(&is1);

          offset_col += map1_size;
        }
        else
        {
          // FIXME: Figure out how to check that matrix block is null
          // Null block, do nothing
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
}
//-----------------------------------------------------------------------------
void Assembler::assemble(la::PETScVector& b, BlockType block_type)
{
  // Check if matrix should be nested
  assert(!_l.empty());
  const bool block_vector = _l.size() > 1;

  if (b.empty())
  {
    // Initialise matrix if empty

    // Build array of pointers to forms
    std::vector<const Form*> forms(_a.size());
    for (std::size_t i = 0; i < _l.size(); ++i)
      forms[i] = _l[i].get();

    // Initialise vector
    if (block_type == BlockType::nested)
    {
      // std::cout << "Init block vector (nested)" << std::endl;
      fem::init_nest(b, forms);
      // std::cout << "End init block vector (nested)" << std::endl;
    }
    else if (block_vector and block_type == BlockType::monolithic)
    {
      // std::cout << "Init block vector (non-nested)" << std::endl;
      fem::init_monolithic(b, forms);
    }
    else
      init(b, *_l[0]);
  }

  // Get vector type
  VecType vec_type;
  VecGetType(b.vec(), &vec_type);
  bool is_vecnest = strcmp(vec_type, VECNEST) == 0 ? true : false;

  if (is_vecnest)
  {
    for (std::size_t i = 0; i < _l.size(); ++i)
    {
      // Get subvector
      Vec sub_b;
      VecNestGetSubVec(b.vec(), i, &sub_b);
      if (_l[i])
      {
        la::PETScVector vec(sub_b);
        // std::cout << "Assemble RHS (nest)" << std::endl;
        this->assemble(vec, *_l[i]);
      }
      else
      {
        // FIXME: Figure out how to check that vector block is null
        // Null block, do nothing
      }
    }
  }
  else if (block_vector)
  {
    // std::cout << "Assembling block vector (non-nested)" << std::endl;
    std::int64_t offset = 0;
    for (std::size_t i = 0; i < _l.size(); ++i)
    {
      if (_l[i])
      {
        auto map = _l[i]->function_space(0)->dofmap()->index_map();
        auto map_size = map->size(common::IndexMap::MapSize::ALL);

        std::vector<PetscInt> index(map_size);
        std::iota(index.begin(), index.end(), offset);

        IS is;
        ISCreateBlock(b.mpi_comm(), map->block_size(), index.size(),
                      index.data(), PETSC_COPY_VALUES, &is);

        Vec sub_b;
        // std::cout << "*** get subvector" << std::endl;
        VecGetSubVector(b.vec(), is, &sub_b);
        // std::cout << "*** end get subvector" << std::endl;
        la::PETScVector vec(sub_b);

        // FIXME: Does it pick up the block size?

        // FIXME: Update for parallel
        // Attach local-to-global map

        // Fill vector with [i0 + 0, i0 + 1, i0 +2, . . .]
        std::vector<PetscInt> local_to_global_map(vec.size());
        std::iota(local_to_global_map.begin(), local_to_global_map.end(), 1);

        // Create PETSc local-to-global map
        ISLocalToGlobalMapping petsc_local_to_global;
        // PetscErrorCode ierr = 0;
        ISLocalToGlobalMappingCreate(PETSC_COMM_SELF, 1,
                                     local_to_global_map.size(),
                                     local_to_global_map.data(),
                                     PETSC_COPY_VALUES, &petsc_local_to_global);
        // CHECK_ERROR("ISLocalToGlobalMappingCreate");

        // Apply local-to-global map to vector
        VecSetLocalToGlobalMapping(sub_b, petsc_local_to_global);
        // CHECK_ERROR("VecSetLocalToGlobalMapping");

        // Clean-up PETSc local-to-global map
        ISLocalToGlobalMappingDestroy(&petsc_local_to_global);
        // CHECK_ERROR("ISLocalToGlobalMappingDestroy");

        this->assemble(vec, *_l[i]);

        VecRestoreSubVector(b.vec(), is, &sub_b);
        ISDestroy(&is);

        offset += map_size;
      }
    }
  }
  else
  {
    this->assemble(b, *_l[0]);
  }

  /*
  // Assemble vector
  this->assemble(b, *_l[0]);

  // Apply bcs to RHS of vector
  for (std::size_t i = 0; i < _l.size(); ++i)
    for (std::size_t j = 0; j < _a[i].size(); ++j)
      apply_bc(b, *_a[i][j], _bcs);

  // Set bc values
  set_bc(b, *_l[0], _bcs);
  `*/

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
  assert(!A.empty());

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
      // else
      // {
      //   std::cout << "     No spaces match: " << std::endl;
      // }
    }
  }

  // Data structures used in assembly
  EigenRowArrayXXd coordinate_dofs;
  EigenRowMatrixXd Ae;

  // Get cell integral
  auto cell_integral = a.integrals().cell_integral();

  // Iterate over all cells
  for (auto& cell : mesh::MeshRange<mesh::Cell>(mesh))
  {
    // std::cout << "Iterate over cells" << std::endl;
    // Check that cell is not a ghost
    assert(!cell.is_ghost());

    // Get cell vertex coordinates
    coordinate_dofs.resize(cell.num_vertices(), gdim);
    cell.get_coordinate_dofs(coordinate_dofs);

    // Update UFC data to current cell
    ufc.update(cell, coordinate_dofs, cell_integral->enabled_coefficients());

    // Get dof maps for cell
    auto dmap0 = dofmaps[0]->cell_dofs(cell.index());
    auto dmap1 = dofmaps[1]->cell_dofs(cell.index());

    // Size data structure for assembly
    Ae.resize(dmap0.size(), dmap1.size());
    Ae.setZero();

    // Compute cell matrix
    cell_integral->tabulate_tensor(Ae.data(), ufc.w(), coordinate_dofs.data(),
                                   1);

    // FIXME: Pass in list  of cells, and list of local dofs, with
    // Dirichlet conditions
    // Note: could use zero dof indices to have PETSc do this
    // Zero rows/columns for Dirichlet bcs
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

  // Flush matrix
  A.apply(la::PETScMatrix::AssemblyType::FLUSH);

  // FIXME: Move this outside of function?
  // Place '1' on diagonal for bc entries
  if (spaces[0] == spaces[1])
  {
    // Note: set diagonal using PETScMatrix::set_local since other functions,
    // e.g. PETScMatrix::set_local, do not work for all PETSc Mat types
    for (auto bc : boundary_values[0])
    {
      la_index_t row = bc.first;
      double one = 1.0;
      A.set_local(&one, 1, &row, 1, &row);
    }
  }

  // Finalise matrix
  A.apply(la::PETScMatrix::AssemblyType::FINAL);
}
//-----------------------------------------------------------------------------
void Assembler::assemble(la::PETScVector& b, const Form& L)
{
  // if (b.empty())
  //  init(b, L);

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
  EigenRowArrayXXd coordinate_dofs;
  EigenVectorXd be;

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

    // Update UFC data to current cell
    ufc.update(cell, coordinate_dofs, cell_integral->enabled_coefficients());

    // Get dof maps for cell
    auto dmap = dofmap->cell_dofs(cell.index());
    // auto dmap1 = dofmaps[1]->cell_dofs(cell.index());

    // Size data structure for assembly
    be.resize(dmap.size());
    be.setZero();

    // Compute cell matrix
    cell_integral->tabulate_tensor(be.data(), ufc.w(), coordinate_dofs.data(),
                                   1);

    // Add to vector
    // std::cout << "Adding to vector: " << be(0) << ", " << dmap[0] <<
    // std::endl;
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

  EigenRowMatrixXd Ae;
  EigenVectorXd be;
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

    // Update UFC data to current cell
    ufc.update(cell, coordinate_dofs, cell_integral->enabled_coefficients());

    // Size data structure for assembly
    auto dmap0 = dofmap1->cell_dofs(cell.index());
    Ae.resize(dmap0.size(), dmap1.size());
    Ae.setZero();
    cell_integral->tabulate_tensor(Ae.data(), ufc.w(), coordinate_dofs.data(),
                                   1);

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
