// Copyright (C) 2018-2019 Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "assemble_scalar_impl.h"
#include "Form.h"
#include <dolfin/common/IndexMap.h>
#include <dolfin/common/types.h>
#include <dolfin/function/Function.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/Facet.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshIterator.h>
#include <petscsys.h>

using namespace dolfin;
using namespace dolfin::fem;

//-----------------------------------------------------------------------------
PetscScalar dolfin::fem::impl::assemble(const dolfin::fem::Form& M)
{
  assert(M.mesh());
  const mesh::Mesh& mesh = *M.mesh();

  // Prepare coefficients
  const FormCoefficients& coefficients = M.coeffs();
  std::vector<const function::Function*> coeff_fn(coefficients.size());
  for (int i = 0; i < coefficients.size(); ++i)
    coeff_fn[i] = coefficients.get(i).get();
  std::vector<int> c_offsets = coefficients.offsets();

  PetscScalar value = 0.0;
  if (M.integrals().num_integrals(fem::FormIntegrals::Type::cell) > 0)
  {
    const std::function<void(PetscScalar*, const PetscScalar*, const double*,
                             int)>& fn
        = M.integrals().get_tabulate_tensor_fn_cell(0);
    value += fem::impl::assemble_cells(mesh, fn, coeff_fn, c_offsets);
  }
  if (M.integrals().num_integrals(fem::FormIntegrals::Type::cell) > 1)
  {
    throw std::runtime_error("Multiple cell integrals not supported yet.");
  }

  if (M.integrals().num_integrals(fem::FormIntegrals::Type::exterior_facet) > 0)
  {
    const std::function<void(PetscScalar*, const PetscScalar*, const double*,
                             int, int)>& fn
        = M.integrals().get_tabulate_tensor_fn_exterior_facet(0);
    value += fem::impl::assemble_exterior_facets(mesh, fn, coeff_fn, c_offsets);
  }
  if (M.integrals().num_integrals(fem::FormIntegrals::Type::exterior_facet) > 1)
  {
    throw std::runtime_error(
        "Multiple exterior facet integrals not supported yet.");
  }

  if (M.integrals().num_integrals(fem::FormIntegrals::Type::interior_facet) > 0)
    value += fem::impl::assemble_interior_facets(M);

  return value;
}
//-----------------------------------------------------------------------------
PetscScalar fem::impl::assemble_cells(
    const mesh::Mesh& mesh,
    const std::function<void(PetscScalar*, const PetscScalar*, const double*,
                             int)>& fn,
    std::vector<const function::Function*> coefficients,
    const std::vector<int>& offsets)
{
  const std::size_t tdim = mesh.topology().dim();
  mesh.init(tdim);

  // Create data structures used in assembly
  Eigen::Array<PetscScalar, Eigen::Dynamic, 1> coeff_array(offsets.back());
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      coordinate_dofs;

  // Iterate over all cells
  PetscScalar cell_value, value(0);
  for (const mesh::Cell& cell : mesh::MeshRange<mesh::Cell>(mesh))
  {
    // Check that cell is not a ghost
    assert(!cell.is_ghost());

    // Get cell vertex coordinates
    cell.get_coordinate_dofs(coordinate_dofs);

    // Update coefficients
    for (std::size_t i = 0; i < coefficients.size(); ++i)
    {
      coefficients[i]->restrict(coeff_array.data() + offsets[i], cell,
                                coordinate_dofs);
    }

    fn(&cell_value, coeff_array.data(), coordinate_dofs.data(), 1);
    value += cell_value;
  }

  return value;
}
//-----------------------------------------------------------------------------
PetscScalar fem::impl::assemble_exterior_facets(
    const mesh::Mesh& mesh,
    const std::function<void(PetscScalar*, const PetscScalar*, const double*,
                             int, int)>& fn,
    std::vector<const function::Function*> coefficients,
    const std::vector<int>& offsets)
{
  const std::size_t tdim = mesh.topology().dim();
  mesh.init(tdim - 1);
  mesh.init(tdim - 1, tdim);

  // Creat data structures used in assembly
  Eigen::Array<PetscScalar, Eigen::Dynamic, 1> coeff_array(offsets.back());
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      coordinate_dofs;

  // Iterate over all facets
  PetscScalar cell_value, value(0);
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

    // Update coefficients
    for (std::size_t i = 0; i < coefficients.size(); ++i)
    {
      coefficients[i]->restrict(coeff_array.data() + offsets[i], cell,
                                coordinate_dofs);
    }

    fn(&cell_value, coeff_array.data(), coordinate_dofs.data(), local_facet, 1);
    value += cell_value;
  }

  return value;
}
//-----------------------------------------------------------------------------
PetscScalar fem::impl::assemble_interior_facets(const Form& M)
{
  throw std::runtime_error("Interior facet integrals not supported yet.");
  return 0.0;
}
//-----------------------------------------------------------------------------
