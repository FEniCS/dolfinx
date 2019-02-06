// Copyright (C) 2018 Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "assemble_matrix_impl.h"
#include "Form.h"
#include "GenericDofMap.h"
#include <dolfin/function/Function.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/la/utils.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/Facet.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshIterator.h>
#include <petscsys.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
void fem::impl::assemble_matrix(Mat A, const Form& a,
                                const std::vector<bool>& bc0,
                                const std::vector<bool>& bc1)
{
  assert(a.mesh());
  const mesh::Mesh& mesh = *a.mesh();
  const fem::GenericDofMap& dofmap0 = *a.function_space(0)->dofmap();
  const fem::GenericDofMap& dofmap1 = *a.function_space(1)->dofmap();

  // Prepare coefficients
  const FormCoefficients& coefficients = a.coeffs();
  std::vector<const function::Function*> coeff_fn(coefficients.size());
  for (int i = 0; i < coefficients.size(); ++i)
    coeff_fn[i] = coefficients.get(i).get();
  std::vector<int> c_offsets = coefficients.offsets();

  if (a.integrals().num_integrals(fem::FormIntegrals::Type::exterior_facet) > 1)
  {
    throw std::runtime_error(
        "Multiple cell integrals in bilinear form not yet supported.");
  }
  if (a.integrals().num_integrals(fem::FormIntegrals::Type::cell) > 0)
  {
    const std::function<void(PetscScalar*, const PetscScalar*, const double*,
                             int)>& fn
        = a.integrals().get_tabulate_tensor_fn_cell(0);
    fem::impl::assemble_cells(A, mesh, dofmap0, dofmap1, bc0, bc1, fn, coeff_fn,
                              c_offsets);
  }

  if (a.integrals().num_integrals(fem::FormIntegrals::Type::exterior_facet) > 1)
  {
    throw std::runtime_error("Multiple exterior facet integrals in bilinear "
                             "form not yet supported.");
  }
  if (a.integrals().num_integrals(fem::FormIntegrals::Type::exterior_facet) > 0)
  {
    const std::function<void(PetscScalar*, const PetscScalar*, const double*,
                             int, int)>& fn
        = a.integrals().get_tabulate_tensor_fn_exterior_facet(0);
    fem::impl::assemble_exterior_facets(A, mesh, dofmap0, dofmap1, bc0, bc1, fn,
                                        coeff_fn, c_offsets);
  }

  if (a.integrals().num_integrals(fem::FormIntegrals::Type::interior_facet) > 0)
  {
    throw std::runtime_error(
        "Interior facet integrals in bilinear forms not yet supported.");
  }
}
//-----------------------------------------------------------------------------
void fem::impl::assemble_cells(
    Mat A, const mesh::Mesh& mesh, const GenericDofMap& dofmap0,
    const GenericDofMap& dofmap1, const std::vector<bool>& bc0,
    const std::vector<bool>& bc1,
    const std::function<void(PetscScalar*, const PetscScalar*, const double*,
                             int)>& fn,
    std::vector<const function::Function*> coefficients,
    const std::vector<int>& offsets)
{
  assert(A);

  // Data structures used in assembly
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      coordinate_dofs;
  Eigen::Matrix<PetscScalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      Ae;
  Eigen::Array<PetscScalar, Eigen::Dynamic, 1> coeff_array(offsets.back());

  // Iterate over all cells
  PetscErrorCode ierr;
  for (auto& cell : mesh::MeshRange<mesh::Cell>(mesh))
  {
    // Check that cell is not a ghost
    assert(!cell.is_ghost());

    // Get cell vertex coordinates
    cell.get_coordinate_dofs(coordinate_dofs);

    // Get dof maps for cell
    const std::size_t cell_index = cell.index();
    Eigen::Map<const Eigen::Array<PetscInt, Eigen::Dynamic, 1>> dmap0
        = dofmap0.cell_dofs(cell_index);
    Eigen::Map<const Eigen::Array<PetscInt, Eigen::Dynamic, 1>> dmap1
        = dofmap1.cell_dofs(cell_index);

    // Update coefficients
    for (std::size_t i = 0; i < coefficients.size(); ++i)
    {
      coefficients[i]->restrict(coeff_array.data() + offsets[i], cell,
                                coordinate_dofs);
    }

    // Tabulate tensor
    Ae.setZero(dmap0.size(), dmap1.size());
    fn(Ae.data(), coeff_array.data(), coordinate_dofs.data(), 1);

    // Zero rows/columns for essential bcs
    if (!bc0.empty())
    {
      for (Eigen::Index i = 0; i < Ae.rows(); ++i)
      {
        if (bc0[dmap0[i]])
          Ae.row(i).setZero();
      }
    }
    if (!bc1.empty())
    {
      for (Eigen::Index j = 0; j < Ae.cols(); ++j)
      {
        if (bc1[dmap1[j]])
          Ae.col(j).setZero();
      }
    }

    ierr = MatSetValuesLocal(A, dmap0.size(), dmap0.data(), dmap1.size(),
                             dmap1.data(), Ae.data(), ADD_VALUES);
#ifdef DEBUG
    if (ierr != 0)
      la::petsc_error(ierr, __FILE__, "MatSetValuesLocal");
#endif
  }
}
//-----------------------------------------------------------------------------
void fem::impl::assemble_exterior_facets(
    Mat A, const mesh::Mesh& mesh, const GenericDofMap& dofmap0,
    const GenericDofMap& dofmap1, const std::vector<bool>& bc0,
    const std::vector<bool>& bc1,
    const std::function<void(PetscScalar*, const PetscScalar*, const double*,
                             int, int)>& fn,
    std::vector<const function::Function*> coefficients,
    const std::vector<int>& offsets)
{
  const std::size_t tdim = mesh.topology().dim();
  mesh.init(tdim - 1);
  mesh.init(tdim - 1, tdim);

  // Data structures used in assembly
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      coordinate_dofs;
  Eigen::Matrix<PetscScalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      Ae;
  Eigen::Array<PetscScalar, Eigen::Dynamic, 1> coeff_array(offsets.back());

  // Iterate over all facets
  PetscErrorCode ierr;
  for (const mesh::Facet& facet : mesh::MeshRange<mesh::Facet>(mesh))
  {
    if (facet.num_global_entities(tdim) != 1)
      continue;

    // TODO: check ghosting sanity?

    // Create attached cell
    mesh::Cell cell(mesh, facet.entities(tdim)[0]);

    // Get local index of facet with respect to the cell
    const int local_facet = cell.index(facet);

    // Get cell vertex coordinates
    cell.get_coordinate_dofs(coordinate_dofs);

    // Get dof maps for cell
    const std::size_t cell_index = cell.index();
    Eigen::Map<const Eigen::Array<PetscInt, Eigen::Dynamic, 1>> dmap0
        = dofmap0.cell_dofs(cell_index);
    Eigen::Map<const Eigen::Array<PetscInt, Eigen::Dynamic, 1>> dmap1
        = dofmap1.cell_dofs(cell_index);

    // Update coefficients
    for (std::size_t i = 0; i < coefficients.size(); ++i)
    {
      coefficients[i]->restrict(coeff_array.data() + offsets[i], cell,
                                coordinate_dofs);
    }

    // Tabulate tensor
    Ae.setZero(dmap0.size(), dmap1.size());
    fn(Ae.data(), coeff_array.data(), coordinate_dofs.data(), local_facet, 1);

    // Zero rows/columns for essential bcs
    if (!bc0.empty())
    {
      for (Eigen::Index i = 0; i < Ae.rows(); ++i)
      {
        if (bc0[dmap0[i]])
          Ae.row(i).setZero();
      }
    }
    if (!bc1.empty())
    {
      for (Eigen::Index j = 0; j < Ae.cols(); ++j)
      {
        if (bc1[dmap1[j]])
          Ae.col(j).setZero();
      }
    }

    ierr = MatSetValuesLocal(A, dmap0.size(), dmap0.data(), dmap1.size(),
                             dmap1.data(), Ae.data(), ADD_VALUES);
#ifdef DEBUG
    if (ierr != 0)
      la::petsc_error(ierr, __FILE__, "MatSetValuesLocal");
#endif
  }
}
//-----------------------------------------------------------------------------
