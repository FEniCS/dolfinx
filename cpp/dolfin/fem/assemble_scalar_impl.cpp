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

  PetscScalar value = 0.0;

  if (M.integrals().num_integrals(fem::FormIntegrals::Type::cell) > 0)
  {
    const std::function<void(PetscScalar*, const PetscScalar*, const double*,
                             int)>& fn
        = M.integrals().tabulate_tensor_fn_cell(0);

    const Eigen::Array<bool, Eigen::Dynamic, 1> enabled_coefficients
        = M.integrals().enabled_coefficients_cell(0);
    const FormCoefficients& coefficients = M.coeffs();
    std::vector<std::uint32_t> n = {0};
    std::vector<const function::Function*> coefficients_ptr(
        coefficients.size());
    std::vector<const FiniteElement*> elements_ptr(coefficients.size());
    for (std::uint32_t i = 0; i < coefficients.size(); ++i)
    {
      coefficients_ptr[i] = coefficients.get(i).get();
      elements_ptr[i] = &coefficients.element(i);
      const FiniteElement& element = coefficients.element(i);
      n.push_back(n.back() + element.space_dimension());
    }
    Eigen::Array<PetscScalar, Eigen::Dynamic, 1> coeff_array(n.back());

    value += fem::impl::assemble_cells(M, mesh, fn, coefficients_ptr);
  }
  if (M.integrals().num_integrals(fem::FormIntegrals::Type::cell) > 1)
  {
    throw std::runtime_error("Multiple cell integrals not supported yet.");
  }

  if (M.integrals().num_integrals(fem::FormIntegrals::Type::exterior_facet) > 0)
  {
    const std::function<void(PetscScalar*, const PetscScalar*, const double*,
                             int, int)>& fn
        = M.integrals().tabulate_tensor_fn_exterior_facet(0);
    value += fem::impl::assemble_exterior_facets(M, mesh, fn);
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
    const Form& M, const mesh::Mesh& mesh,
    const std::function<void(PetscScalar*, const PetscScalar*, const double*,
                             int)>& fn,
    std::vector<const function::Function*> coefficients)
{
  const std::size_t tdim = mesh.topology().dim();
  mesh.init(tdim);

  // TODO: simplify and move elsewhere
  // Manage coefficients
  const Eigen::Array<bool, Eigen::Dynamic, 1> enabled_coefficients
      = M.integrals().enabled_coefficients_cell(0);
  const FormCoefficients& coeffs = M.coeffs();
  std::vector<std::uint32_t> n = {0};
  std::vector<const function::Function*> coefficients_ptr(coefficients.size());
  for (std::uint32_t i = 0; i < coeffs.size(); ++i)
  {
    coefficients_ptr[i] = coeffs.get(i).get();
    n.push_back(
        n.back()
        + coefficients_ptr[i]->function_space()->element()->space_dimension());
  }
  Eigen::Array<PetscScalar, Eigen::Dynamic, 1> coeff_array(n.back());

  // Create data structures used in assembly
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

    // TODO: Move gathering of coefficients outside of main assembly
    // loop
    // Update coefficients
    for (std::size_t i = 0; i < coefficients.size(); ++i)
    {
      if (enabled_coefficients[i])
      {
        coefficients[i]->restrict(coeff_array.data() + n[i], cell,
                                  coordinate_dofs);
      }
    }
    fn(&cell_value, coeff_array.data(), coordinate_dofs.data(), 1);
    value += cell_value;
  }

  return value;
}
//-----------------------------------------------------------------------------
PetscScalar fem::impl::assemble_exterior_facets(
    const Form& M, const mesh::Mesh& mesh,
    const std::function<void(PetscScalar*, const PetscScalar*, const double*,
                             int, int)>& fn)
{
  const std::size_t tdim = mesh.topology().dim();
  mesh.init(tdim - 1);
  mesh.init(tdim - 1, tdim);

  // Creat data structures used in assembly
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      coordinate_dofs;

  const Eigen::Array<bool, Eigen::Dynamic, 1> enabled_coefficients
      = M.integrals().enabled_coefficients_exterior_facet(0);
  const FormCoefficients& coefficients = M.coeffs();
  std::vector<std::uint32_t> n = {0};

  std::vector<const function::Function*> coefficients_ptr(coefficients.size());
  for (std::uint32_t i = 0; i < coefficients.size(); ++i)
  {
    coefficients_ptr[i] = coefficients.get(i).get();
    n.push_back(
        n.back()
        + coefficients_ptr[i]->function_space()->element()->space_dimension());
  }
  Eigen::Array<PetscScalar, Eigen::Dynamic, 1> coeff_array(n.back());

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

    // TODO: Move gathering of coefficients outside of main assembly
    // loop
    // Update coefficients
    for (std::size_t i = 0; i < coefficients.size(); ++i)
    {
      if (enabled_coefficients[i])
      {
        coefficients_ptr[i]->restrict(coeff_array.data() + n[i], cell,
                                      coordinate_dofs);
      }
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
